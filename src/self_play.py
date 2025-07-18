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
        # Track whose perspective each state represents
        is_white_turn = True  # First state is from white's perspective

        for i in range(len(self.states)):
            # From white's perspective: +1 for white win, -1 for black win
            # From black's perspective: -1 for white win, +1 for black win
            reward = final_result if is_white_turn else -final_result
            self.rewards.append(reward)
            # Toggle turn for next state
            is_white_turn = not is_white_turn

    def get_training_examples(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Get all training examples from this game."""
        return list(zip(self.states, self.policies, self.rewards))


class SelfPlay:
    """Self-play data generation for AlphaZero."""

    def __init__(
        self,
        neural_net: AlphaZeroNet,
        mcts_simulations: int,
        c_puct: float,
        dirichlet_alpha: float,
        dirichlet_epsilon: float,
        mcts_batch_size: int,
        resign_threshold: int,
        initial_temperature: float,
        min_temperature: float,
        move_limit: int,
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
        self.resign_threshold = resign_threshold
        self.initial_temperature = initial_temperature
        self.min_temperature = min_temperature
        self.move_limit = move_limit

    def _clear_line(self):
        print("\r" + " " * 120 + "\r", end="", flush=True)

    def play_game(self, verbose: bool = True) -> SelfPlayGame:
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
            temperature = max(
                self.min_temperature,
                self.initial_temperature - (move_count * 0.01),
            )

            # Run MCTS to get improved policy (with noise for exploration)
            policy, root = self.mcts.search(
                env, temperature=temperature, add_noise=True
            )

            # Check for resignation based on value estimate from root node
            if move_count > self.resign_threshold:  # Don't resign too early
                position_value = root.mean_value
                # Position value is from current player's perspective
                # Need to check if position is bad for the current player
                resign_threshold = -0.80

                if position_value < resign_threshold:
                    if verbose:
                        self._clear_line()
                        print(
                            f"Resignation at move {move_count}, position value: {position_value:.3f}"
                        )
                    # Set result: if current player resigns, opponent wins
                    # White (player 0) resigning means black wins (-1)
                    # Black (player 1) resigning means white wins (+1)
                    final_result = -1.0 if env.board.turn else 1.0
                    game.set_rewards(final_result)
                    return game

            # Add training example
            game.add_example(state, policy)

            # Select move based on policy
            legal_moves = env.get_legal_moves()
            if len(legal_moves) == 0:
                break

            # Sample move from policy based on temperature
            move_probs = []
            moves = []
            for move in legal_moves:
                move_idx = env.move_to_index(move)
                move_probs.append(policy[move_idx])
                moves.append(move)

            # Normalize probabilities
            move_probs = np.array(move_probs)
            if move_probs.sum() > 0:
                # Temperature affects move selection randomness
                if temperature < 0.01:  # Nearly deterministic
                    chosen_idx = np.argmax(move_probs)
                else:
                    # Apply temperature to the probabilities
                    move_probs = move_probs ** (1 / temperature)
                    move_probs = move_probs / move_probs.sum()  # Renormalize
                    chosen_idx = np.random.choice(len(moves), p=move_probs)
            else:
                chosen_idx = np.random.choice(len(moves))

            chosen_move = moves[chosen_idx]

            # Make the move
            env.make_move(chosen_move)
            move_count += 1

            if verbose and move_count % 10 == 0:
                print(f"Move {move_count}, FEN: {env.get_fen()}", end="\r")

            # Reduced move limit to prevent endless games
            if move_count > self.move_limit:
                if verbose:
                    self._clear_line()
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
                else "Black wins" if final_result < 0 else "Draw"
            )
            self._clear_line()
            print(f"Game finished: {result_str}, Moves: {move_count}")

        return game

    def generate_training_data(
        self, num_games: int, verbose: bool = False
    ) -> tuple[List[Tuple[np.ndarray, np.ndarray, float]], List[float]]:
        """
        Generate training data by playing multiple self-play games.

        Args:
            num_games: Number of games to play
            verbose: Whether to print progress

        Returns:
            Tuple of (training_examples, game_results)
            - training_examples: List of training examples (state, policy, reward)
            - game_results: List of game outcomes (+1 for white win, -1 for black win, 0 for draw)
        """
        all_examples = []
        game_results = []  # Track actual game results

        for game_idx in range(num_games):
            if verbose:
                print(f"Playing game {game_idx + 1}/{num_games}", end="\r")
                print()

            game = self.play_game(verbose=verbose)
            examples = game.get_training_examples()
            all_examples.extend(examples)

            # Track the actual game result (from the last reward in the game)
            final_result = game.rewards[-1] if game.rewards else 0.0
            game_results.append(final_result)

            if verbose and (game_idx + 1) % 10 == 0:
                print(
                    f"Completed {game_idx + 1} games, {len(all_examples)} examples collected"
                )

        if verbose:
            print(
                f"Generated {len(all_examples)} training examples from {num_games} games"
            )
            # Print game result summary
            wins = sum(1 for r in game_results if r > 0)
            draws = sum(1 for r in game_results if r == 0)
            losses = sum(1 for r in game_results if r < 0)
            print(f"Game results: {wins} wins, {draws} draws, {losses} losses")

        return all_examples, game_results

    def evaluate_position(
        self, fen: str, use_mcts: bool = False
    ) -> Tuple[float, np.ndarray]:
        """
        Evaluate a chess position.

        Args:
            fen: FEN string of position to evaluate
            use_mcts: Whether to use MCTS for evaluation (more accurate but slower)

        Returns:
            Value estimate and policy probabilities
        """
        env = ChessEnvironment()
        env.from_fen(fen)

        if use_mcts:
            # Use MCTS for more accurate evaluation (like in actual play)
            policy, root = self.mcts.search(env, temperature=0.0, add_noise=False)
            value = root.mean_value
        else:
            # Use direct neural network evaluation (faster)
            state = env.get_state()
            policy, value = self.neural_net.predict(state)

            # Mask illegal moves
            legal_moves_mask = env.get_legal_moves_mask()
            policy = policy * legal_moves_mask

            # Renormalize
            if policy.sum() > 0:
                policy = policy / policy.sum()

        return value, policy
