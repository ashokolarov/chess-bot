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

        # Batch inference support
        self.is_evaluating = False  # Flag to track if node is waiting for evaluation

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

    def select_child(self, c_puct: float) -> "MCTSNode":
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
    """Monte Carlo Tree Search for AlphaZero with batch inference."""

    def __init__(
        self,
        neural_net,
        c_puct: float,
        num_simulations: int,
        dirichlet_alpha: float,
        dirichlet_epsilon: float,
        batch_size: int,
    ):
        self.neural_net = neural_net
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.batch_size = batch_size
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
        self, env: ChessEnvironment, temperature, add_noise: bool
    ) -> Tuple[np.ndarray, MCTSNode]:
        """
        Perform MCTS search with batch inference and return improved policy.

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

        # Run simulations with batch inference
        self._run_batched_simulations(root)

        # Get improved policy
        policy = root.get_action_probs(temperature)

        return policy, root

    def _run_batched_simulations(self, root: MCTSNode):
        """
        Run MCTS simulations with batch neural network inference.

        Args:
            root: Root node of the search tree
        """
        # Track paths through the tree for each simulation
        simulation_paths = []

        for sim_idx in range(self.num_simulations):
            # Traverse tree to find leaf node
            path = self._traverse_to_leaf(root)
            simulation_paths.append(path)

            # Collect nodes that need evaluation in batches
            if (
                len(simulation_paths) >= self.batch_size
                or sim_idx == self.num_simulations - 1
            ):
                self._evaluate_batch_and_backup(simulation_paths)
                simulation_paths = []

    def _traverse_to_leaf(self, root: MCTSNode) -> List[MCTSNode]:
        """
        Traverse from root to a leaf node, returning the path.

        Args:
            root: Root node to start traversal

        Returns:
            Path from root to leaf node
        """
        path = []
        current = root

        while True:
            path.append(current)

            # Check if game is over
            if current.env.is_game_over():
                break

            # If leaf node, we'll expand it later
            if current.is_leaf():
                break

            # Select child and continue
            current = current.select_child(self.c_puct)

        return path

    def _evaluate_batch_and_backup(self, simulation_paths: List[List[MCTSNode]]):
        """
        Evaluate leaf nodes in batch and backup values.

        Args:
            simulation_paths: List of paths, each ending at a leaf node
        """
        # Collect unique leaf nodes that need evaluation
        leaf_nodes = []
        path_to_leaf_map = {}  # Maps path index to leaf node

        for path_idx, path in enumerate(simulation_paths):
            leaf_node = path[-1]

            # Skip if game is over
            if leaf_node.env.is_game_over():
                result = leaf_node.env.get_result()
                if result is not None:
                    self._backup_path(path, result)
                continue

            # Skip if already expanded (shouldn't happen but safety check)
            if not leaf_node.is_leaf():
                continue

            # Add to evaluation batch
            if leaf_node not in [n for n, _ in leaf_nodes]:
                leaf_nodes.append((leaf_node, path_idx))
            path_to_leaf_map[path_idx] = leaf_node

        if not leaf_nodes:
            return

        # Prepare batch inference
        states = [node.env.get_state() for node, _ in leaf_nodes]

        # Batch neural network inference
        policies, values = self.neural_net.predict_batch(states)

        # Process results and expand nodes
        leaf_to_policy_value = {}
        for (node, _), policy, value in zip(leaf_nodes, policies, values):
            # Mask illegal moves
            legal_moves_mask = node.env.get_legal_moves_mask()
            policy = policy * legal_moves_mask

            # Renormalize policy
            if policy.sum() > 0:
                policy = policy / policy.sum()

            # Expand node
            node.expand(policy)
            leaf_to_policy_value[node] = value

        # Backup values for all simulation paths
        for path_idx, path in enumerate(simulation_paths):
            leaf_node = path[-1]

            # Skip paths that ended in terminal states
            if leaf_node.env.is_game_over():
                continue

            # Get value for this leaf node
            if leaf_node in leaf_to_policy_value:
                value = leaf_to_policy_value[leaf_node]
                self._backup_path(path, value)

    def _backup_path(self, path: List[MCTSNode], value: float):
        """
        Backup value through a path in the tree.

        Args:
            path: Path of nodes from root to leaf
            value: Value to backup
        """
        # Backup from leaf to root, alternating value sign
        for i, node in enumerate(reversed(path)):
            # Alternate sign for each level (opponent perspective)
            backup_value = value if i % 2 == 0 else -value

            # Only update statistics, don't recurse to parent
            node.visit_count += 1
            node.total_value += backup_value
            node.mean_value = node.total_value / node.visit_count

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
