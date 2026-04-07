"""
Connect Four AI: Minimax (baseline), Alpha-Beta pruning, and iterative deepening.
"""

import time
import random
import numpy as np
from typing import Optional, List, Tuple
from connect_four import ConnectFourBoard, COLS, ROWS, EMPTY, PLAYER_1, PLAYER_2

# Transposition Table
TRANSPOSITION_TABLE = {}

WIN_SCORE = 100_000
LOSS_SCORE = -100_000

_CENTER_ORDER = [3, 2, 4, 1, 5, 0, 6]


def _opponent(player: int) -> int:
    return PLAYER_2 if player == PLAYER_1 else PLAYER_1


def _order_moves(valid_moves: List[int]) -> List[int]:
    """Return moves sorted center-first for better alpha-beta cutoffs."""
    return [c for c in _CENTER_ORDER if c in valid_moves]


# ---------------------------------------------------------------------------
# Search statistics
# ---------------------------------------------------------------------------

class SearchStats:
    """Mutable counter object threaded through search calls."""
    __slots__ = ("nodes",)

    def __init__(self) -> None:
        self.nodes: int = 0

    def reset(self) -> None:
        self.nodes = 0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

POSITION_WEIGHTS = np.array([
    [3, 4,  5,  7,  5, 4, 3],
    [4, 6,  8, 10,  8, 6, 4],
    [5, 8, 11, 13, 11, 8, 5],
    [5, 8, 11, 13, 11, 8, 5],
    [4, 6,  8, 10,  8, 6, 4],
    [3, 4,  5,  7,  5, 4, 3],
], dtype=np.float64)


def _score_window(window: np.ndarray, player: int, opp: int) -> float:
    """Score a window of 4 cells."""
    p = int(np.sum(window == player))
    o = int(np.sum(window == opp))
    empty = int(np.sum(window == EMPTY))

    if p == 4:
        return 1000
    if p == 3 and empty == 1:
        return 50
    if p == 2 and empty == 2:
        return 10
    if o == 3 and empty == 1:
        return -80
    if o == 2 and empty == 2:
        return -8
    return 0


def evaluate(board: ConnectFourBoard, player: int) -> float:
    """
    Improved heuristic: positional weight table + sliding window-of-4
    across rows, columns, and both diagonals.
    """
    opp = _opponent(player)
    b = board.board
    score = 0.0

    score += float(np.sum(POSITION_WEIGHTS * (b == player)))
    score -= float(np.sum(POSITION_WEIGHTS * (b == opp)))

    for r in range(ROWS):
        for c in range(COLS - 3):
            score += _score_window(b[r, c:c + 4], player, opp)

    for c in range(COLS):
        for r in range(ROWS - 3):
            score += _score_window(b[r:r + 4, c], player, opp)

    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            window = np.array([b[r + i, c + i] for i in range(4)])
            score += _score_window(window, player, opp)

    for r in range(3, ROWS):
        for c in range(COLS - 3):
            window = np.array([b[r - i, c + i] for i in range(4)])
            score += _score_window(window, player, opp)

    return score


def simple_evaluate(board: ConnectFourBoard, player: int) -> float:
    """
    Simple heuristic: piece count difference + center bias.
    Positive score is good for `player`, negative for opponent.
    """
    opp = _opponent(player)
    score = 0.0

    # Piece count: more of my pieces is better
    my_count = np.sum(board.board == player)
    opp_count = np.sum(board.board == opp)
    score += (my_count - opp_count) * 1.0

    for c in (2, 3, 4):
        for r in range(ROWS):
            if board.board[r, c] == player:
                score += 0.5
            elif board.board[r, c] == opp:
                score -= 0.5

    return score


def minimax(
    board: ConnectFourBoard,
    depth: int,
    maximizing: bool,
    player: int,
    stats: Optional[SearchStats] = None,
) -> Tuple[float, Optional[int]]:
    """
    Minimax (no pruning). Returns (score, best_column).
    Uses simple_evaluate for a fair baseline comparison.
    """
    if stats is not None:
        stats.nodes += 1

    winner = board.get_winner()
    if winner is not None:
        return (WIN_SCORE if winner == player else LOSS_SCORE, None)
    if board.is_draw():
        return (0.0, None)

    valid = board.get_valid_moves()
    if not valid or depth == 0:
        return (simple_evaluate(board, player), None)

    opp = _opponent(player)
    best_col: Optional[int] = None

    if maximizing:
        best_score = float("-inf")
        for col in valid:
            child = board.copy()
            child.drop(col, player)
            score, _ = minimax(child, depth - 1, False, player, stats)
            if score > best_score:
                best_score = score
                best_col = col
        return (best_score, best_col)
    else:
        best_score = float("inf")
        for col in valid:
            child = board.copy()
            child.drop(col, opp)
            score, _ = minimax(child, depth - 1, True, player, stats)
            if score < best_score:
                best_score = score
                best_col = col
        return (best_score, best_col)


def get_ai_move(
    board: ConnectFourBoard,
    player: int,
    depth: int = 4,
    stats: Optional[SearchStats] = None,
) -> Optional[int]:
    """
    Return the best column for `player` using plain Minimax at fixed depth.
    Kept for backward-compatibility with play.py and baseline experiments.
    """
    valid = board.get_valid_moves()
    if not valid:
        return None
    _, col = minimax(board, depth, True, player, stats)
    return col


# ---------------------------------------------------------------------------
# Alpha-Beta Pruning
# ---------------------------------------------------------------------------

def alphabeta(
    board: ConnectFourBoard,
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
    player: int,
    stats: Optional[SearchStats] = None,
) -> Tuple[float, Optional[int]]:
    """
    Minimax with alpha-beta pruning. Returns (score, best_column).
    """
    if stats is not None:
        stats.nodes += 1

    # --- Transposition lookup ---
    key = _board_key(board, player, maximizing)
    if key in TRANSPOSITION_TABLE:
        stored_depth, stored_score = TRANSPOSITION_TABLE[key]
        if stored_depth >= depth:
            return stored_score, None
        
    winner = board.get_winner()
    if winner is not None:
        return (WIN_SCORE if winner == player else LOSS_SCORE, None)
    if board.is_draw():
        return (0.0, None)

    valid = _order_moves(board.get_valid_moves())
    if not valid or depth == 0:
        return (evaluate(board, player), None)

    opp = _opponent(player)
    best_col: Optional[int] = valid[0]

    if maximizing:
        best_score = float("-inf")
        for col in valid:
            child = board.copy()
            child.drop(col, player)
            score, _ = alphabeta(child, depth - 1, alpha, beta, False, player, stats)
            if score > best_score:
                best_score = score
                best_col = col
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break
        return (best_score, best_col)
    else:
        best_score = float("inf")
        for col in valid:
            child = board.copy()
            child.drop(col, opp)
            score, _ = alphabeta(child, depth - 1, alpha, beta, True, player, stats)
            if score < best_score:
                best_score = score
                best_col = col
            beta = min(beta, best_score)
            if alpha >= beta:
                break

            # --- Store result ---
        TRANSPOSITION_TABLE[key] = (depth, best_score)
        return (best_score, best_col)


def get_ab_move(
    board: ConnectFourBoard,
    player: int,
    depth: int = 6,
    stats: Optional[SearchStats] = None,
) -> Optional[int]:
    """Return the best column using alpha-beta at fixed depth."""
    valid = board.get_valid_moves()
    if not valid:
        return None
    _, col = alphabeta(board, depth, float("-inf"), float("inf"), True, player, stats)
    return col


# ---------------------------------------------------------------------------
# Iterative Deepening
# ---------------------------------------------------------------------------

def iterative_deepening(
    board: ConnectFourBoard,
    player: int,
    max_depth: int = 20,
    time_limit: float = 5.0,
    stats: Optional[SearchStats] = None,
) -> Tuple[Optional[int], int]:
    """
    Iterative deepening over alpha-beta. Searches depth 1, 2, ... up to
    max_depth or until time_limit seconds elapse.

    Returns (best_column, depth_reached).
    """
    best_col: Optional[int] = None
    depth_reached = 0
    start = time.perf_counter()

    for depth in range(1, max_depth + 1):
        if time.perf_counter() - start >= time_limit:
            break
        score, col = alphabeta(
            board, depth, float("-inf"), float("inf"), True, player, stats
        )
        if col is not None:
            best_col = col
        depth_reached = depth
        if abs(score) >= WIN_SCORE:
            break
        if time.perf_counter() - start >= time_limit:
            break

    return best_col, depth_reached


def get_id_move(
    board: ConnectFourBoard,
    player: int,
    max_depth: int = 20,
    time_limit: float = 5.0,
    stats: Optional[SearchStats] = None,
) -> Optional[int]:
    """Return the best column using iterative-deepening alpha-beta."""
    valid = board.get_valid_moves()
    if not valid:
        return None
    col, _ = iterative_deepening(board, player, max_depth, time_limit, stats)
    return col


# ---------------------------------------------------------------------------
# Random agent
# ---------------------------------------------------------------------------

def get_random_move(board: ConnectFourBoard, player: int, **_kwargs) -> Optional[int]:
    """Pick a uniformly random valid column."""
    valid = board.get_valid_moves()
    if not valid:
        return None
    return random.choice(valid)

# ---------------------------------------------------------------------------
# Create a hashable key for the board state
# ---------------------------------------------------------------------------

def _board_key(board: ConnectFourBoard, player: int, maximizing: bool) -> tuple:
    """
    Create a hashable key for the board state.
    Includes player + turn to avoid incorrect reuse.
    """
    return (tuple(board.board.flatten()), player, maximizing)