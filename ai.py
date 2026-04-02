"""
Connect Four AI: Minimax (baseline) and Alpha-Beta pruning.
"""

import numpy as np
from typing import Optional, List, Tuple
from connect_four import ConnectFourBoard, COLS, ROWS, EMPTY, PLAYER_1, PLAYER_2

WIN_SCORE = 100_000
LOSS_SCORE = -100_000

_CENTER_ORDER = [3, 2, 4, 1, 5, 0, 6]


def _opponent(player: int) -> int:
    return PLAYER_2 if player == PLAYER_1 else PLAYER_1


def _order_moves(valid_moves: List[int]) -> List[int]:
    """Return moves sorted center-first for better alpha-beta cutoffs."""
    return [c for c in _CENTER_ORDER if c in valid_moves]


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
) -> Tuple[float, Optional[int]]:
    """
    Minimax (no pruning). Returns (score, best_column).
    best_column is None only at terminal state when no move is possible.
    """
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
            score, _ = minimax(child, depth - 1, False, player)
            if score > best_score:
                best_score = score
                best_col = col
        return (best_score, best_col)
    else:
        best_score = float("inf")
        for col in valid:
            child = board.copy()
            child.drop(col, opp)
            score, _ = minimax(child, depth - 1, True, player)
            if score < best_score:
                best_score = score
                best_col = col
        return (best_score, best_col)


def get_ai_move(board: ConnectFourBoard, player: int, depth: int = 4) -> Optional[int]:
    """
    Return the best column for `player` using plain Minimax at fixed depth.
    Kept for backward-compatibility with play.py and baseline experiments.
    """
    valid = board.get_valid_moves()
    if not valid:
        return None
    _, col = minimax(board, depth, True, player)
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
) -> Tuple[float, Optional[int]]:
    """
    Minimax with alpha-beta pruning. Returns (score, best_column).
    """
    winner = board.get_winner()
    if winner is not None:
        return (WIN_SCORE if winner == player else LOSS_SCORE, None)
    if board.is_draw():
        return (0.0, None)

    valid = _order_moves(board.get_valid_moves())
    if not valid or depth == 0:
        return (simple_evaluate(board, player), None)

    opp = _opponent(player)
    best_col: Optional[int] = valid[0]

    if maximizing:
        best_score = float("-inf")
        for col in valid:
            child = board.copy()
            child.drop(col, player)
            score, _ = alphabeta(child, depth - 1, alpha, beta, False, player)
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
            score, _ = alphabeta(child, depth - 1, alpha, beta, True, player)
            if score < best_score:
                best_score = score
                best_col = col
            beta = min(beta, best_score)
            if alpha >= beta:
                break
        return (best_score, best_col)


def get_ab_move(board: ConnectFourBoard, player: int, depth: int = 6) -> Optional[int]:
    """Return the best column using alpha-beta at fixed depth."""
    valid = board.get_valid_moves()
    if not valid:
        return None
    _, col = alphabeta(board, depth, float("-inf"), float("inf"), True, player)
    return col
