"""
Basic AI: Minimax with depth 4 and simple heuristic (piece count + center bias).
"""

import numpy as np
from typing import Optional, List, Tuple
from connect_four import ConnectFourBoard, COLS, ROWS, EMPTY, PLAYER_1, PLAYER_2


def simple_evaluate(board: ConnectFourBoard, player: int) -> float:
    """
    Simple heuristic: piece count difference + center bias.
    Positive score is good for `player`, negative for opponent.
    """
    opponent = PLAYER_2 if player == PLAYER_1 else PLAYER_1
    score = 0.0

    # Piece count: more of my pieces is better
    my_count = np.sum(board.board == player)
    opp_count = np.sum(board.board == opponent)
    score += (my_count - opp_count) * 1.0

    # Center bias: pieces in center columns (2,3,4) are more valuable
    center_cols = [2, 3, 4]
    for c in center_cols:
        for r in range(ROWS):
            if board.board[r, c] == player:
                score += 0.5
            elif board.board[r, c] == opponent:
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
        if winner == player:
            return (1000.0, None)
        return (-1000.0, None)
    if board.is_draw():
        return (0.0, None)

    valid = board.get_valid_moves()
    if not valid or depth == 0:
        return (simple_evaluate(board, player), None)

    opponent = PLAYER_2 if player == PLAYER_1 else PLAYER_1
    best_col: Optional[int] = None

    if maximizing:
        best_score = -np.inf
        for col in valid:
            child = board.copy()
            child.drop(col, player)
            score, _ = minimax(child, depth - 1, False, player)
            if score > best_score:
                best_score = score
                best_col = col
        return (best_score, best_col)
    else:
        best_score = np.inf
        for col in valid:
            child = board.copy()
            child.drop(col, opponent)
            score, _ = minimax(child, depth - 1, True, player)
            if score < best_score:
                best_score = score
                best_col = col
        return (best_score, best_col)


def get_ai_move(board: ConnectFourBoard, player: int, depth: int = 4) -> Optional[int]:
    """
    Return the best column for `player` using Minimax with given depth.
    Returns None if no valid move.
    """
    valid = board.get_valid_moves()
    if not valid:
        return None
    _, col = minimax(board, depth, True, player)
    return col
