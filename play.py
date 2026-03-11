#!/usr/bin/env python3
"""
Simple CLI to play Connect Four vs the basic Minimax AI (depth 4).
Human is Player 1 (X), AI is Player 2 (O). Human moves first.
"""

from connect_four import ConnectFourBoard, PLAYER_1, PLAYER_2
from ai import get_ai_move


def main() -> None:
    board = ConnectFourBoard()
    human = PLAYER_1
    ai_player = PLAYER_2
    print("Connect Four — You are X, AI is O. Enter column 0–6.")
    print(board.to_string())

    while not board.is_terminal():
        # Human move
        valid = board.get_valid_moves()
        while True:
            try:
                raw = input("\nYour move (column 0–6): ").strip()
                col = int(raw)
                if col in valid:
                    break
                print(f"Column {col} is full or invalid. Valid: {valid}")
            except ValueError:
                print("Enter a number 0–6.")
        board.drop(col, human)
        print(board.to_string())
        if board.get_winner() == human:
            print("You win!")
            return
        if board.is_draw():
            print("Draw.")
            return

        # AI move
        col = get_ai_move(board, ai_player, depth=4)
        if col is None:
            print("No valid move for AI.")
            break
        print(f"\nAI plays column {col}")
        board.drop(col, ai_player)
        print(board.to_string())
        if board.get_winner() == ai_player:
            print("AI wins!")
            return
        if board.is_draw():
            print("Draw.")
            return

    print("Game over.")


if __name__ == "__main__":
    main()
