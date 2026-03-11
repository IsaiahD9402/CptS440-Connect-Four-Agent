"""
Connect Four game environment.
Board: 7 columns x 6 rows, NumPy array. 0=empty, 1=player1, 2=player2.
Pieces drop in columns (0-6) and fall to the lowest empty row.
"""

import numpy as np
from typing import Optional, List, Tuple

# Board dimensions (columns x rows)
COLS = 7
ROWS = 6

# Players
EMPTY = 0
PLAYER_1 = 1
PLAYER_2 = 2


class ConnectFourBoard:
    """Connect Four board with move validation and win/draw detection."""

    def __init__(self, board: Optional[np.ndarray] = None):
        if board is not None:
            self.board = np.array(board, dtype=np.int8)
            assert self.board.shape == (ROWS, COLS), "Board must be 6x7 (rows x cols)"
        else:
            self.board = np.zeros((ROWS, COLS), dtype=np.int8)

    def copy(self) -> "ConnectFourBoard":
        return ConnectFourBoard(self.board.copy())

    def get_valid_moves(self) -> List[int]:
        """Return list of column indices (0..6) that are not full."""
        return [c for c in range(COLS) if self.board[0, c] == EMPTY]

    def is_valid_move(self, col: int) -> bool:
        if col < 0 or col >= COLS:
            return False
        return self.board[0, col] == EMPTY

    def drop(self, col: int, player: int) -> bool:
        """
        Drop a piece in column `col` for `player`. Returns True if successful.
        Piece lands in the lowest empty row (row index 5 at bottom).
        """
        if not self.is_valid_move(col):
            return False
        # Find lowest empty row (rows are 0=top, 5=bottom)
        for row in range(ROWS - 1, -1, -1):
            if self.board[row, col] == EMPTY:
                self.board[row, col] = player
                return True
        return False

    def remove(self, col: int) -> None:
        """Remove the top piece in column (for undoing / search)."""
        for row in range(ROWS):
            if self.board[row, col] != EMPTY:
                self.board[row, col] = EMPTY
                return
        raise ValueError(f"Column {col} is empty")

    def _check_line(self, r: int, c: int, dr: int, dc: int, player: int) -> bool:
        """Check if player has 4 in a row starting at (r,c) in direction (dr,dc)."""
        for _ in range(4):
            if r < 0 or r >= ROWS or c < 0 or c >= COLS or self.board[r, c] != player:
                return False
            r, c = r + dr, c + dc
        return True

    def get_winner(self) -> Optional[int]:
        """
        Return PLAYER_1 or PLAYER_2 if that player has 4 in a row, else None.
        Does not consider draw (use is_draw() or is_terminal() for that).
        """
        for player in (PLAYER_1, PLAYER_2):
            for r in range(ROWS):
                for c in range(COLS):
                    if self._check_line(r, c, 0, 1, player):   # horizontal
                        return player
                    if self._check_line(r, c, 1, 0, player):   # vertical
                        return player
                    if self._check_line(r, c, 1, 1, player):   # diagonal \
                        return player
                    if self._check_line(r, c, 1, -1, player): # diagonal /
                        return player
        return None

    def is_draw(self) -> bool:
        """True if board is full and there is no winner."""
        if self.get_winner() is not None:
            return False
        return np.all(self.board != EMPTY)

    def is_terminal(self) -> bool:
        """True if game is over (win or draw)."""
        return self.get_winner() is not None or self.is_draw()

    def __repr__(self) -> str:
        return f"ConnectFourBoard(\n{self.board}\n)"

    def to_string(self) -> str:
        """Human-readable board for printing."""
        lines = []
        for r in range(ROWS):
            line = "|"
            for c in range(COLS):
                v = self.board[r, c]
                line += " X" if v == PLAYER_1 else " O" if v == PLAYER_2 else " ."
            line += " |"
            lines.append(line)
        lines.append("+---------------+" + "\n " + " ".join(str(i) for i in range(COLS)))
        return "\n".join(lines)
