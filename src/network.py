from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block for the neural network."""

    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class AlphaZeroNet(nn.Module):
    """
    AlphaZero neural network with policy and value heads.
    Takes 8x8x12 board representation as input.
    """

    def __init__(self, num_res_blocks: int = 4, num_channels: int = 64):
        super(AlphaZeroNet, self).__init__()

        # Store architecture parameters for parallel workers
        self.num_res_blocks = num_res_blocks
        self.num_channels = num_channels

        # Initial convolution
        self.conv_input = nn.Conv2d(
            12, num_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn_input = nn.BatchNorm2d(num_channels)

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, 4096)  # 64*64 possible moves

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Board state tensor of shape (batch_size, 12, 8, 8)

        Returns:
            policy_logits: Move probabilities of shape (batch_size, 4096)
            value: Position evaluation of shape (batch_size, 1)
        """
        # Initial convolution
        x = F.relu(self.bn_input(self.conv_input(x)))

        # Residual blocks
        x = self.res_blocks(x)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy_logits = self.policy_fc(policy)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy_logits, value

    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Make prediction for a single state.

        Args:
            state: Board state of shape (8, 8, 12)

        Returns:
            policy: Move probabilities of shape (4096,)
            value: Position evaluation scalar
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensor and add batch dimension, move to same device as model
            device = next(self.parameters()).device
            state_tensor = (
                torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(device)
            )

            policy_logits, value = self.forward(state_tensor)

            # Convert to probabilities
            policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            value = value.cpu().item()

        return policy, value

    def save_checkpoint(self, filepath: str, optimizer_state=None, epoch=None):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "epoch": epoch,
        }
        if optimizer_state:
            checkpoint["optimizer_state_dict"] = optimizer_state
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str, map_location=None):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=map_location)
        self.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint.get("epoch", 0), checkpoint.get("optimizer_state_dict", None)
