from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, StepLR
from torch.utils.data import DataLoader, Dataset

from .network import AlphaZeroNet


class ChessDataset(Dataset):
    """Dataset for chess training examples."""

    def __init__(
        self, examples: List[Tuple[np.ndarray, np.ndarray, float]], device: str = "cpu"
    ):
        self.examples = examples
        self.device = device

        # Precompute tensors for better performance
        self.states = []
        self.policies = []
        self.values = []

        for state, policy, value in examples:
            # Convert to tensors
            state_tensor = torch.FloatTensor(state).permute(2, 0, 1)  # (12, 8, 8)
            policy_tensor = torch.FloatTensor(policy)
            value_tensor = torch.FloatTensor([value])

            self.states.append(state_tensor)
            self.policies.append(policy_tensor)
            self.values.append(value_tensor)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Return precomputed tensors and move to device only once when accessed
        return (
            self.states[idx].to(self.device),
            self.policies[idx].to(self.device),
            self.values[idx].to(self.device),
        )


class AlphaZeroTrainer:
    """Trainer for AlphaZero neural network."""

    def __init__(
        self,
        neural_net: AlphaZeroNet,
        learning_rate: float,
        weight_decay: float,
        device: str,
        scheduler_type: str = "step",
        scheduler_params: dict = None,
    ):
        self.neural_net = neural_net
        self.device = device
        self.neural_net.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            neural_net.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.lr = learning_rate

        # Learning rate scheduler config
        self.scheduler_type = scheduler_type
        self.scheduler_params = scheduler_params

        # Loss functions
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()

        # Training history
        self.training_history = {
            "policy_loss": [],
            "value_loss": [],
            "total_loss": [],
            "learning_rate": [],
        }

    def set_learning_rate(self, new_lr: float):
        """
        Set a new base learning rate.

        Args:
            new_lr: New learning rate value
        """
        self.lr = new_lr

    def _create_scheduler(
        self,
        scheduler_type: str,
        scheduler_params: dict,
        total_num_steps: int = None,
    ):
        """
        Create a learning rate scheduler.

        Args:
            scheduler_type: Type of scheduler ('step')
            scheduler_params: Parameters for the scheduler
            num_training_samples: Number of training samples
        Returns:
            Learning rate scheduler
        """
        if scheduler_type == "step":
            # StepLR: Reduce LR every step_size epochs by gamma
            step_size = scheduler_params.get("step_size", 5)
            gamma = scheduler_params.get("gamma", 0.8)
            return StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == "onecycle":
            # OneCycleLR: Gradually increase LR from min_lr to max_lr
            max_lr_factor = scheduler_params.get("max_lr_factor")
            max_lr = self.lr * max_lr_factor
            pct_start = scheduler_params.get("pct_start")
            return OneCycleLR(
                self.optimizer,
                max_lr=max_lr,
                total_steps=total_num_steps,
                pct_start=pct_start,
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

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
        # Explicitly clear gradients for better memory management
        self.optimizer.zero_grad(set_to_none=True)

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
        # Clip gradients to prevent exploding gradients
        # torch.nn.utils.clip_grad_norm_(self.neural_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.scheduler_type == "onecycle":
            self.scheduler.step()

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
        dataloader: DataLoader,
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

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0
        num_batches = len(dataloader)

        for states, policies, values in dataloader:
            # Training step
            policy_loss, value_loss, batch_loss = self.train_step(
                states, policies, values
            )

            total_policy_loss += policy_loss
            total_value_loss += value_loss
            total_loss += batch_loss

        # Calculate averages
        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        avg_total_loss = total_loss / num_batches

        # Update history
        self.training_history["policy_loss"].append(avg_policy_loss)
        self.training_history["value_loss"].append(avg_value_loss)
        self.training_history["total_loss"].append(avg_total_loss)
        self.training_history["learning_rate"].append(self.get_current_lr())

        return avg_policy_loss, avg_value_loss, avg_total_loss

    def train(
        self,
        training_examples: List[Tuple[np.ndarray, np.ndarray, float]],
        epochs: int,
        batch_size: int,
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

        # Validate batch size is not larger than dataset
        valid_batch_size = min(batch_size, len(training_examples))
        if valid_batch_size < batch_size:
            print(
                f"Warning: Reduced batch size from {batch_size} to {valid_batch_size} to match dataset size"
            )
            batch_size = valid_batch_size

        # Create dataset and dataloader
        dataset = ChessDataset(training_examples, device=self.device)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Create learning rate scheduler
        self.scheduler = self._create_scheduler(
            self.scheduler_type, self.scheduler_params, len(dataloader) * epochs
        )

        for epoch in range(epochs):
            # Train for one epoch
            policy_loss, value_loss, total_loss = self.train_epoch(dataloader)

            if verbose:
                current_lr = self.get_current_lr()
                print(
                    f"Epoch {epoch + 1}/{epochs}: "
                    f"Policy Loss: {policy_loss:.4f}, "
                    f"Value Loss: {value_loss:.4f}, "
                    f"Total Loss: {total_loss:.4f}, "
                    f"LR: {current_lr:.6f}"
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
