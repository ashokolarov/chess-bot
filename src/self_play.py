from typing import List, Tuple

import numpy as np

from .environment import ChessEnvironment
from .mcts import MCTS
from .network import AlphaZeroNet


class SelfPlayGame:
    """Represents a single self-play game and its training examples."""

    def __init__(self):
        self.states = []  # Board states
        self.policies = []  # MCTS improved policies
        self.rewards = []  # Game outcomes from each position
        self.move_count = 0

    def add_example(self, state: np.ndarray, policy: np.ndarray):
        """Add a training example (state, policy) to the game."""
        self.states.append(state.copy())
        self.policies.append(policy.copy())
        self.move_count += 1

    def set_rewards(self, final_result: float):
        """
        Set rewards for all positions based on game outcome.

        Args:
            final_result: Game result (+1 for win, -1 for loss, 0 for draw)
        """
        self.rewards = []
        for i in range(len(self.states)):
            # Alternate perspective for each move
            reward = final_result if i % 2 == 0 else -final_result
            self.rewards.append(reward)

    def get_training_examples(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Get all training examples from this game."""
        return list(zip(self.states, self.policies, self.rewards))


class SelfPlay:
    """Self-play data generation for AlphaZero."""

    def __init__(
        self,
        neural_net: AlphaZeroNet,
        mcts_simulations: int = 100,
        temperature_threshold: int = 15,
        c_puct: float = 1.0,
        dirichlet_alpha: float = 0.25,
        dirichlet_epsilon: float = 0.25,
        mcts_batch_size: int = 8,
    ):
        self.neural_net = neural_net
        self.mcts = MCTS(
            neural_net,
            c_puct=c_puct,
            num_simulations=mcts_simulations,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            batch_size=mcts_batch_size,
        )
        self.temperature_threshold = (
            temperature_threshold  # Move number after which temp=0
        )

    def play_game(self, verbose: bool = False) -> SelfPlayGame:
        """
        Play a complete self-play game.

        Args:
            verbose: Whether to print game progress

        Returns:
            SelfPlayGame object with training examples
        """
        env = ChessEnvironment()
        game = SelfPlayGame()

        move_count = 0

        while not env.is_game_over():
            # Get current state
            state = env.get_state()

            # Determine temperature (high early in game, low later)
            temperature = 1.0 if move_count < self.temperature_threshold else 0.1

            # Run MCTS to get improved policy (with noise for exploration)
            policy, root = self.mcts.search(
                env, temperature=temperature, add_noise=True
            )

            # Check for resignation based on value estimate
            # Get value estimate from root node (after MCTS expansion)
            if (
                hasattr(root, "mean_value") and move_count > 20
            ):  # Don't resign too early
                position_value = root.mean_value
                # Resign if position is very bad (threshold: -0.9)
                if position_value < -0.9:
                    if verbose:
                        print(
                            f"Resignation at move {move_count}, position value: {position_value:.3f}"
                        )
                    # Set result: if current player resigns, opponent wins
                    final_result = -1.0  # Current player loses
                    game.set_rewards(final_result)
                    return game

            # Add training example
            game.add_example(state, policy)

            # Select move based on policy
            legal_moves = env.get_legal_moves()
            if len(legal_moves) == 0:
                break

            # Sample move from policy
            move_probs = []
            moves = []
            for move in legal_moves:
                move_idx = env.move_to_index(move)
                move_probs.append(policy[move_idx])
                moves.append(move)

            # Normalize probabilities
            move_probs = np.array(move_probs)
            if move_probs.sum() > 0:
                move_probs = move_probs / move_probs.sum()
                chosen_idx = np.random.choice(len(moves), p=move_probs)
            else:
                chosen_idx = np.random.choice(len(moves))

            chosen_move = moves[chosen_idx]

            # Make the move
            env.make_move(chosen_move)
            move_count += 1

            if verbose and move_count % 10 == 0:
                print(f"Move {move_count}, FEN: {env.get_fen()}")

            # Reduced move limit to prevent endless games
            if move_count > 150:  # Reduced from 200 to 150
                if verbose:
                    print("Game terminated due to move limit")
                break

        # Get game result and set rewards
        final_result = env.get_result()
        if final_result is None:
            final_result = 0.0  # Draw if game limit reached

        game.set_rewards(final_result)

        if verbose:
            result_str = (
                "White wins"
                if final_result > 0
                else "Black wins"
                if final_result < 0
                else "Draw"
            )
            print(f"Game finished: {result_str}, Moves: {move_count}")

        return game

    def generate_training_data(
        self, num_games: int, verbose: bool = False
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Generate training data by playing multiple self-play games.

        Args:
            num_games: Number of games to play
            verbose: Whether to print progress

        Returns:
            List of training examples (state, policy, reward)
        """
        all_examples = []

        for game_idx in range(num_games):
            if verbose:
                print(f"Playing game {game_idx + 1}/{num_games}")

            game = self.play_game(verbose=False)
            examples = game.get_training_examples()
            all_examples.extend(examples)

            if verbose and (game_idx + 1) % 10 == 0:
                print(
                    f"Completed {game_idx + 1} games, {len(all_examples)} examples collected"
                )

        if verbose:
            print(
                f"Generated {len(all_examples)} training examples from {num_games} games"
            )

        return all_examples

    def evaluate_position(self, fen: str) -> Tuple[float, np.ndarray]:
        """
        Evaluate a chess position.

        Args:
            fen: FEN string of position to evaluate

        Returns:
            Value estimate and policy probabilities
        """
        env = ChessEnvironment()
        env.from_fen(fen)

        state = env.get_state()
        policy, value = self.neural_net.predict(state)

        # Mask illegal moves
        legal_moves_mask = env.get_legal_moves_mask()
        policy = policy * legal_moves_mask

        # Renormalize
        if policy.sum() > 0:
            policy = policy / policy.sum()

        return value, policy
