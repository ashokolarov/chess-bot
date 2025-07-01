import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from .network import AlphaZeroNet


class ChessDataset(Dataset):
    """Dataset for chess training examples."""

    def __init__(
        self, examples: List[Tuple[np.ndarray, np.ndarray, float]], device: str = "cpu"
    ):
        self.examples = examples
        self.device = device

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        state, policy, value = self.examples[idx]

        # Convert to tensors and move to device
        state_tensor = (
            torch.FloatTensor(state).permute(2, 0, 1).to(self.device)
        )  # (12, 8, 8)
        policy_tensor = torch.FloatTensor(policy).to(self.device)
        value_tensor = torch.FloatTensor([value]).to(self.device)

        return state_tensor, policy_tensor, value_tensor


class AlphaZeroTrainer:
    """Trainer for AlphaZero neural network."""

    def __init__(
        self,
        neural_net: AlphaZeroNet,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        device: str = None,
    ):
        self.neural_net = neural_net
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.neural_net.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            neural_net.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Loss functions
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()

        # Training history
        self.training_history = {"policy_loss": [], "value_loss": [], "total_loss": []}

    def train_step(
        self,
        states: torch.Tensor,
        target_policies: torch.Tensor,
        target_values: torch.Tensor,
    ) -> Tuple[float, float, float]:
        """
        Perform a single training step.

        Args:
            states: Batch of board states
            target_policies: Target policy distributions
            target_values: Target position values

        Returns:
            Policy loss, value loss, total loss
        """
        self.neural_net.train()
        self.optimizer.zero_grad()

        # Forward pass
        policy_logits, predicted_values = self.neural_net(states)

        # Calculate losses
        policy_loss = self.calculate_policy_loss(policy_logits, target_policies)
        value_loss = self.value_loss_fn(
            predicted_values.squeeze(), target_values.squeeze()
        )
        total_loss = policy_loss + value_loss

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        return policy_loss.item(), value_loss.item(), total_loss.item()

    def calculate_policy_loss(
        self, policy_logits: torch.Tensor, target_policies: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate policy loss using KL divergence.

        Args:
            policy_logits: Predicted policy logits
            target_policies: Target policy probabilities

        Returns:
            Policy loss
        """
        # Convert logits to log probabilities
        log_probs = torch.log_softmax(policy_logits, dim=1)

        # KL divergence loss: sum(target * log(target / predicted))
        # Equivalent to: sum(target * (log(target) - log(predicted)))
        # Since target * log(target) is constant, we minimize: -sum(target * log(predicted))
        policy_loss = -torch.sum(target_policies * log_probs, dim=1).mean()

        return policy_loss

    def train_epoch(
        self,
        training_examples: List[Tuple[np.ndarray, np.ndarray, float]],
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> Tuple[float, float, float]:
        """
        Train for one epoch on the given examples.

        Args:
            training_examples: List of (state, policy, value) tuples
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data

        Returns:
            Average policy loss, value loss, total loss for the epoch
        """
        # Create dataset and dataloader
        dataset = ChessDataset(training_examples, device=self.device)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0
        num_batches = 0

        for states, policies, values in dataloader:
            # Tensors are already on the correct device from dataset
            pass

            # Training step
            policy_loss, value_loss, batch_loss = self.train_step(
                states, policies, values
            )

            total_policy_loss += policy_loss
            total_value_loss += value_loss
            total_loss += batch_loss
            num_batches += 1

        # Calculate averages
        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        avg_total_loss = total_loss / num_batches

        # Update history
        self.training_history["policy_loss"].append(avg_policy_loss)
        self.training_history["value_loss"].append(avg_value_loss)
        self.training_history["total_loss"].append(avg_total_loss)

        return avg_policy_loss, avg_value_loss, avg_total_loss

    def train(
        self,
        training_examples: List[Tuple[np.ndarray, np.ndarray, float]],
        epochs: int = 10,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> dict:
        """
        Train the neural network for multiple epochs.

        Args:
            training_examples: Training data
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Whether to print progress

        Returns:
            Training history dictionary
        """
        if verbose:
            print(
                f"Training on {len(training_examples)} examples for {epochs} epochs..."
            )

        for epoch in range(epochs):
            # Shuffle examples for each epoch
            shuffled_examples = training_examples.copy()
            random.shuffle(shuffled_examples)

            # Train for one epoch
            policy_loss, value_loss, total_loss = self.train_epoch(
                shuffled_examples, batch_size=batch_size, shuffle=True
            )

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{epochs}: "
                    f"Policy Loss: {policy_loss:.4f}, "
                    f"Value Loss: {value_loss:.4f}, "
                    f"Total Loss: {total_loss:.4f}"
                )

        return self.training_history

    def evaluate(
        self,
        test_examples: List[Tuple[np.ndarray, np.ndarray, float]],
        batch_size: int = 32,
    ) -> Tuple[float, float, float]:
        """
        Evaluate the model on test data.

        Args:
            test_examples: Test data
            batch_size: Batch size for evaluation

        Returns:
            Average policy loss, value loss, total loss
        """
        self.neural_net.eval()

        dataset = ChessDataset(test_examples, device=self.device)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for states, policies, values in dataloader:
                # Forward pass
                policy_logits, predicted_values = self.neural_net(states)

                # Calculate losses
                policy_loss = self.calculate_policy_loss(policy_logits, policies)
                value_loss = self.value_loss_fn(
                    predicted_values.squeeze(), values.squeeze()
                )
                batch_loss = policy_loss + value_loss

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_loss += batch_loss.item()
                num_batches += 1

        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        avg_total_loss = total_loss / num_batches

        return avg_policy_loss, avg_value_loss, avg_total_loss

    def save_checkpoint(
        self, filepath: str, epoch: int = None, additional_info: dict = None
    ):
        """Save training checkpoint."""
        checkpoint = {
            "model_state_dict": self.neural_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_history": self.training_history,
            "epoch": epoch,
        }

        if additional_info:
            checkpoint.update(additional_info)

        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str, load_optimizer: bool = True):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.neural_net.load_state_dict(checkpoint["model_state_dict"])

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "training_history" in checkpoint:
            self.training_history = checkpoint["training_history"]

        return checkpoint.get("epoch", 0)
