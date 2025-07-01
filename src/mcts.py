import math
from typing import Dict, List, Optional, Tuple

import chess
import numpy as np

from .environment import ChessEnvironment


class MCTSNode:
    """Node in the MCTS tree."""

    def __init__(
        self,
        env: ChessEnvironment,
        parent: Optional["MCTSNode"] = None,
        move: Optional[chess.Move] = None,
        prior_prob: float = 0.0,
    ):
        self.env = env
        self.parent = parent
        self.move = move
        self.prior_prob = prior_prob

        # MCTS statistics
        self.visit_count = 0
        self.total_value = 0.0
        self.mean_value = 0.0

        # Children nodes
        self.children: Dict[chess.Move, MCTSNode] = {}
        self.is_expanded = False

    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0

    def expand(self, policy: np.ndarray):
        """
        Expand the node by adding child nodes for all legal moves.

        Args:
            policy: Policy probabilities from neural network
        """
        if self.is_expanded:
            return

        legal_moves = self.env.get_legal_moves()

        for move in legal_moves:
            # Get prior probability for this move
            move_idx = self.env.move_to_index(move)
            prior_prob = policy[move_idx]

            # Create child node
            child_env = self.env.copy()
            child_env.make_move(move)

            child_node = MCTSNode(
                child_env, parent=self, move=move, prior_prob=prior_prob
            )
            self.children[move] = child_node

        self.is_expanded = True

    def select_child(self, c_puct: float = 1.0) -> "MCTSNode":
        """
        Select child node using UCB formula.

        Args:
            c_puct: Exploration constant

        Returns:
            Selected child node
        """
        best_score = -float("inf")
        best_child = None

        for child in self.children.values():
            # UCB score: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            exploitation = child.mean_value
            exploration = (
                c_puct
                * child.prior_prob
                * math.sqrt(self.visit_count)
                / (1 + child.visit_count)
            )
            score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def backup(self, value: float):
        """
        Backup value through the tree.

        Args:
            value: Value to backup
        """
        self.visit_count += 1
        self.total_value += value
        self.mean_value = self.total_value / self.visit_count

        if self.parent:
            # Flip value for parent (opponent's perspective)
            self.parent.backup(-value)

    def get_action_probs(self, temperature: float = 1.0) -> np.ndarray:
        """
        Get action probabilities based on visit counts.

        Args:
            temperature: Temperature parameter for exploration

        Returns:
            Action probabilities array
        """
        probs = np.zeros(4096)

        if temperature == 0:
            # Deterministic: choose most visited move
            best_move = max(
                self.children.keys(), key=lambda m: self.children[m].visit_count
            )
            move_idx = self.env.move_to_index(best_move)
            probs[move_idx] = 1.0
        else:
            # Stochastic: probabilities proportional to visit counts
            visit_counts = np.array(
                [self.children[move].visit_count for move in self.children.keys()]
            )
            if temperature != 1.0:
                visit_counts = visit_counts ** (1 / temperature)

            # Normalize
            if visit_counts.sum() > 0:
                visit_counts = visit_counts / visit_counts.sum()

                for i, move in enumerate(self.children.keys()):
                    move_idx = self.env.move_to_index(move)
                    probs[move_idx] = visit_counts[i]

        return probs


class MCTS:
    """Monte Carlo Tree Search for AlphaZero."""

    def __init__(
        self,
        neural_net,
        c_puct: float = 1.0,
        num_simulations: int = 100,
        dirichlet_alpha: float = 0.25,
        dirichlet_epsilon: float = 0.25,
    ):
        self.neural_net = neural_net
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        # Get device from neural network
        self.device = next(neural_net.parameters()).device

    def add_dirichlet_noise(
        self, policy: np.ndarray, legal_moves: List[chess.Move], env: ChessEnvironment
    ) -> np.ndarray:
        """
        Add Dirichlet noise to policy for exploration.

        Args:
            policy: Original policy probabilities
            legal_moves: List of legal moves
            env: Chess environment for move indexing

        Returns:
            Policy with Dirichlet noise applied
        """
        if len(legal_moves) == 0:
            return policy

        # Generate Dirichlet noise
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))

        # Create noise array matching policy shape
        noise_policy = np.zeros_like(policy)
        for i, move in enumerate(legal_moves):
            move_idx = env.move_to_index(move)
            noise_policy[move_idx] = noise[i]

        # Mix original policy with noise
        noisy_policy = (
            1 - self.dirichlet_epsilon
        ) * policy + self.dirichlet_epsilon * noise_policy

        return noisy_policy

    def search(
        self, env: ChessEnvironment, temperature: float = 1.0, add_noise: bool = False
    ) -> Tuple[np.ndarray, MCTSNode]:
        """
        Perform MCTS search and return improved policy.

        Args:
            env: Chess environment
            temperature: Temperature for action selection
            add_noise: Whether to add Dirichlet noise for exploration

        Returns:
            Improved policy probabilities and root node
        """
        root = MCTSNode(env.copy())

        # Initial expansion of root node with optional noise
        if root.is_leaf():
            policy, value = self.neural_net.predict(root.env.get_state())

            # Mask illegal moves
            legal_moves_mask = root.env.get_legal_moves_mask()
            policy = policy * legal_moves_mask

            # Renormalize policy
            if policy.sum() > 0:
                policy = policy / policy.sum()

            # Add Dirichlet noise if requested (for training)
            if add_noise:
                legal_moves = root.env.get_legal_moves()
                policy = self.add_dirichlet_noise(policy, legal_moves, root.env)

            # Expand root node
            root.expand(policy)

            # Backup initial value
            root.backup(value)

        # Run simulations
        for _ in range(self.num_simulations):
            self._simulate(root)

        # Get improved policy
        policy = root.get_action_probs(temperature)

        return policy, root

    def _simulate(self, node: MCTSNode):
        """
        Run a single MCTS simulation.

        Args:
            node: Current node to simulate from
        """
        # Check if game is over
        if node.env.is_game_over():
            result = node.env.get_result()
            if result is not None:
                node.backup(result)
            return

        # If leaf node, expand and evaluate
        if node.is_leaf():
            policy, value = self.neural_net.predict(node.env.get_state())

            # Mask illegal moves
            legal_moves_mask = node.env.get_legal_moves_mask()
            policy = policy * legal_moves_mask

            # Renormalize policy
            if policy.sum() > 0:
                policy = policy / policy.sum()

            # Expand node
            node.expand(policy)

            # Backup value
            node.backup(value)
            return

        # Select child and continue simulation
        child = node.select_child(self.c_puct)
        self._simulate(child)

    def get_best_move(
        self, env: ChessEnvironment, temperature: float = 0.0
    ) -> chess.Move:
        """
        Get the best move using MCTS.

        Args:
            env: Chess environment
            temperature: Temperature for move selection

        Returns:
            Best move
        """
        policy, root = self.search(env, temperature)

        # Find move with highest probability
        legal_moves = env.get_legal_moves()
        best_prob = -1
        best_move = None

        for move in legal_moves:
            move_idx = env.move_to_index(move)
            if policy[move_idx] > best_prob:
                best_prob = policy[move_idx]
                best_move = move

        return best_move if best_move else legal_moves[0]
