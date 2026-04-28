"""
Arena: pit two Connect Four agents against each other for N games and
report win/draw rates plus per-move search statistics.

Usage examples:
    python arena.py                         # alpha-beta vs random, 100 games
    python arena.py --p1 minimax --p2 random --games 50 --depth 4
    python arena.py --p1 id --p2 ab --games 20
"""

import argparse
import time
from typing import Callable, Optional, Dict, List

from connect_four import ConnectFourBoard, PLAYER_1, PLAYER_2
from ai import (
    get_ai_move,
    get_ab_move,
    get_id_move,
    get_random_move,
    SearchStats,
    clear_transposition_table,
)

AGENT_REGISTRY: Dict[str, Callable] = {
    "minimax": get_ai_move,
    "ab": get_ab_move,
    "id": get_id_move,
    "random": get_random_move,
}


def play_game(
    agent1_fn: Callable,
    agent2_fn: Callable,
    agent1_kwargs: dict,
    agent2_kwargs: dict,
    verbose: bool = False,
) -> Dict:
    """
    Play a single game. agent1 is PLAYER_1 (moves first), agent2 is PLAYER_2.

    Returns a dict with:
        winner: 1, 2, or 0 (draw)
        moves: total number of moves played
        p1_nodes: total nodes expanded by player 1
        p2_nodes: total nodes expanded by player 2
        p1_time: total seconds spent by player 1
        p2_time: total seconds spent by player 2
    """
    board = ConnectFourBoard()
    clear_transposition_table()
    stats1, stats2 = SearchStats(), SearchStats()
    total_time1, total_time2 = 0.0, 0.0
    move_count = 0

    agents = [
        (PLAYER_1, agent1_fn, agent1_kwargs, stats1),
        (PLAYER_2, agent2_fn, agent2_kwargs, stats2),
    ]

    turn = 0
    while not board.is_terminal():
        player, fn, kwargs, stats = agents[turn]
        stats_before = stats.nodes

        t0 = time.perf_counter()
        col = fn(board, player, stats=stats, **kwargs)
        elapsed = time.perf_counter() - t0

        if turn == 0:
            total_time1 += elapsed
        else:
            total_time2 += elapsed

        if col is None:
            break
        board.drop(col, player)
        move_count += 1

        if verbose:
            nodes_this = stats.nodes - stats_before
            print(
                f"Move {move_count}: P{player} -> col {col}  "
                f"({nodes_this} nodes, {elapsed:.3f}s)"
            )
            print(board.to_string())
            print()

        turn = 1 - turn

    winner = board.get_winner()
    return {
        "winner": winner if winner else 0,
        "moves": move_count,
        "p1_nodes": stats1.nodes,
        "p2_nodes": stats2.nodes,
        "p1_time": total_time1,
        "p2_time": total_time2,
    }


def run_arena(
    agent1_name: str,
    agent2_name: str,
    num_games: int,
    agent1_kwargs: dict,
    agent2_kwargs: dict,
    verbose: bool = False,
) -> List[Dict]:
    """Run num_games matches and return list of per-game result dicts."""
    agent1_fn = AGENT_REGISTRY[agent1_name]
    agent2_fn = AGENT_REGISTRY[agent2_name]
    results: List[Dict] = []

    for i in range(1, num_games + 1):
        result = play_game(agent1_fn, agent2_fn, agent1_kwargs, agent2_kwargs, verbose)
        results.append(result)

        tag = (
            f"P1 ({agent1_name}) wins"
            if result["winner"] == 1
            else f"P2 ({agent2_name}) wins"
            if result["winner"] == 2
            else "Draw"
        )
        print(f"Game {i:>4}/{num_games}: {tag}  ({result['moves']} moves)")

    return results


def print_summary(
    results: List[Dict],
    agent1_name: str,
    agent2_name: str,
) -> None:
    n = len(results)
    p1_wins = sum(1 for r in results if r["winner"] == 1)
    p2_wins = sum(1 for r in results if r["winner"] == 2)
    draws = sum(1 for r in results if r["winner"] == 0)
    total_p1_nodes = sum(r["p1_nodes"] for r in results)
    total_p2_nodes = sum(r["p2_nodes"] for r in results)
    total_p1_time = sum(r["p1_time"] for r in results)
    total_p2_time = sum(r["p2_time"] for r in results)
    total_moves = sum(r["moves"] for r in results)

    print("\n" + "=" * 50)
    print(f"Results: {n} games  |  P1={agent1_name}  P2={agent2_name}")
    print("=" * 50)
    print(f"  P1 wins : {p1_wins:>4}  ({100 * p1_wins / n:.1f}%)")
    print(f"  P2 wins : {p2_wins:>4}  ({100 * p2_wins / n:.1f}%)")
    print(f"  Draws   : {draws:>4}  ({100 * draws / n:.1f}%)")
    print(f"  Total moves played : {total_moves}")
    print(f"  Avg moves per game : {total_moves / n:.1f}")
    print(f"  P1 total nodes     : {total_p1_nodes:>12,}")
    print(f"  P2 total nodes     : {total_p2_nodes:>12,}")
    print(f"  P1 total time      : {total_p1_time:>9.2f}s")
    print(f"  P2 total time      : {total_p2_time:>9.2f}s")
    if total_moves > 0:
        avg_p1_time = total_p1_time / (total_moves / 2) if total_moves > 1 else 0
        avg_p2_time = total_p2_time / (total_moves / 2) if total_moves > 1 else 0
        print(f"  P1 avg time/move   : {avg_p1_time:>9.4f}s")
        print(f"  P2 avg time/move   : {avg_p2_time:>9.4f}s")
    print("=" * 50)


def _build_kwargs(agent_name: str, args: argparse.Namespace) -> dict:
    """Build agent-specific keyword arguments from CLI args."""
    if agent_name == "random":
        return {}
    if agent_name == "id":
        return {"max_depth": args.max_depth, "time_limit": args.time_limit}
    return {"depth": args.depth}


def main() -> None:
    parser = argparse.ArgumentParser(description="Connect Four arena")
    parser.add_argument(
        "--p1", choices=AGENT_REGISTRY.keys(), default="ab", help="Player 1 agent"
    )
    parser.add_argument(
        "--p2", choices=AGENT_REGISTRY.keys(), default="random", help="Player 2 agent"
    )
    parser.add_argument("--games", type=int, default=100, help="Number of games")
    parser.add_argument("--depth", type=int, default=6, help="Search depth (minimax/ab)")
    parser.add_argument("--max-depth", type=int, default=20, help="Max depth (id)")
    parser.add_argument("--time-limit", type=float, default=5.0, help="Time limit per move (id)")
    parser.add_argument("--verbose", action="store_true", help="Print every move")
    args = parser.parse_args()

    p1_kwargs = _build_kwargs(args.p1, args)
    p2_kwargs = _build_kwargs(args.p2, args)

    results = run_arena(args.p1, args.p2, args.games, p1_kwargs, p2_kwargs, args.verbose)
    print_summary(results, args.p1, args.p2)


if __name__ == "__main__":
    main()
