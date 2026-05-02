# Connect Four Game-Playing Agent
**CptS 440 — Adversarial Search Project**

This repository contains a game-playing agent capable of playing a two-player, perfect-information, zero-sum board game (Connect Four) using adversarial search techniques. 

By iteratively enhancing a baseline algorithm with state-of-the-art search optimizations, our agent is capable of playing competitively in real-time.

## Features & Implementation
Our agent (`ai.py`) utilizes the following search algorithms and enhancements:
- **Minimax (Baseline):** Exhaustive game tree evaluation to a fixed depth.
- **Alpha-Beta Pruning:** Skips mathematically irrelevant branches to reduce the effective branching factor.
- **Move Ordering:** Evaluates center columns first to force earlier Alpha-Beta cutoffs.
- **Transposition Tables:** Hashes and caches previously evaluated board states to prevent redundant computation.
- **Iterative Deepening:** Provides an "anytime" response, allowing the agent to search as deeply as possible within a strict time budget (e.g., 2 seconds per move).
- **Advanced Heuristics:** Uses a positional weight matrix combined with a sliding window-of-4 to evaluate non-terminal states, prioritizing center control and blocking opponent threats.

## Repository Structure
- `connect_four.py`: The core NumPy-based game environment, board representation, and win detection.
- `ai.py`: Contains the search algorithms (Minimax, Alpha-Beta, Iterative Deepening) and the heuristic evaluation functions.
- `play.py`: An interactive CLI script to play a game against the AI yourself.
- `arena.py`: A benchmarking script to pit different AI algorithms against each other to measure win rates, search time, and node expansion efficiency.
- `connect_four_demo.ipynb`: A Jupyter Notebook demonstrating the agent's capabilities and systematically presenting our experimental results and performance graphs.

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Interactive Play (Human vs. AI)
Play against the baseline Minimax AI in your terminal:
```bash
python play.py
```

### 3. Automated Arena (AI vs. AI)
Watch the agents play each other and view performance statistics. By default, this runs 100 games of Alpha-Beta vs. Random:
```bash
python arena.py
```
You can customize the matchups using command line arguments. For example, to run Iterative Deepening vs Alpha-Beta for 20 games and print every move:
```bash
python arena.py --p1 id --p2 ab --games 20 --verbose
```

### 4. View the Experiments
Open the `connect_four_demo.ipynb` notebook in Jupyter or VS Code and run the cells to reproduce the experimental results (Win Rate, Node Expansion Reduction, Execution Time) and generate the corresponding charts.