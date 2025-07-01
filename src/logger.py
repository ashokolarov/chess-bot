import json
import os
import time
from datetime import datetime
from typing import Any, Dict

import matplotlib.pyplot as plt


class AlphaZeroLogger:
    """Logger for AlphaZero training progress and metrics."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.create_log_directory()

        # Training metrics
        self.metrics = {
            "iteration": [],
            "policy_loss": [],
            "value_loss": [],
            "total_loss": [],
            "games_played": [],
            "training_examples": [],
            "time_per_iteration": [],
            "avg_game_length": [],
            "win_rate_white": [],
            "draw_rate": [],
            "loss_rate": [],
            "learning_rate": [],
        }

        # For timing
        self.iteration_start_time = None

    def create_log_directory(self):
        """Create logging directory if it doesn't exist."""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def start_iteration(self):
        """Mark the start of a training iteration."""
        self.iteration_start_time = time.time()

    def log_iteration(
        self,
        iteration: int,
        policy_loss: float,
        value_loss: float,
        total_loss: float,
        games_played: int,
        training_examples: int,
        avg_game_length: float = None,
        win_rate_white: float = None,
        draw_rate: float = None,
        loss_rate: float = None,
        learning_rate: float = None,
    ):
        """
        Log metrics for a training iteration.

        Args:
            iteration: Current iteration number
            policy_loss: Average policy loss
            value_loss: Average value loss
            total_loss: Average total loss
            games_played: Number of games played this iteration
            training_examples: Number of training examples generated
            avg_game_length: Average length of games
            win_rate_white: Win rate for white pieces
            draw_rate: Draw rate
            loss_rate: Loss rate
            learning_rate: Learning rate
        """
        # Calculate time per iteration
        time_per_iteration = 0
        if self.iteration_start_time:
            time_per_iteration = time.time() - self.iteration_start_time

        # Update metrics
        self.metrics["iteration"].append(iteration)
        self.metrics["policy_loss"].append(policy_loss)
        self.metrics["value_loss"].append(value_loss)
        self.metrics["total_loss"].append(total_loss)
        self.metrics["games_played"].append(games_played)
        self.metrics["training_examples"].append(training_examples)
        self.metrics["time_per_iteration"].append(time_per_iteration)
        self.metrics["avg_game_length"].append(avg_game_length or 0)
        self.metrics["win_rate_white"].append(win_rate_white or 0)
        self.metrics["draw_rate"].append(draw_rate or 0)
        self.metrics["loss_rate"].append(loss_rate or 0)
        self.metrics["learning_rate"].append(learning_rate or 0)

        # Print progress
        self.print_iteration_summary(
            iteration,
            policy_loss,
            value_loss,
            total_loss,
            games_played,
            training_examples,
            time_per_iteration,
        )

    def print_iteration_summary(
        self,
        iteration: int,
        policy_loss: float,
        value_loss: float,
        total_loss: float,
        games_played: int,
        training_examples: int,
        time_per_iteration: float,
    ):
        """Print summary of current iteration."""
        print(f"\n{'=' * 60}")
        print(f"ITERATION {iteration}")
        print(f"{'=' * 60}")
        print(f"Games Played: {games_played}")
        print(f"Training Examples: {training_examples}")
        print(f"Policy Loss: {policy_loss:.4f}")
        print(f"Value Loss: {value_loss:.4f}")
        print(f"Total Loss: {total_loss:.4f}")
        print(f"Time: {time_per_iteration:.1f}s")
        print(f"{'=' * 60}")

    def plot_training_progress(self, save_plot: bool = True, show_plot: bool = False):
        """
        Plot training progress metrics.

        Args:
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
        """
        if len(self.metrics["iteration"]) == 0:
            print("No data to plot yet.")
            return

        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        fig.suptitle("AlphaZero Training Progress", fontsize=16)

        iterations = self.metrics["iteration"]

        # Loss curves
        axes[0, 0].plot(
            iterations, self.metrics["policy_loss"], label="Policy Loss", color="blue"
        )
        axes[0, 0].plot(
            iterations, self.metrics["value_loss"], label="Value Loss", color="red"
        )
        axes[0, 0].plot(
            iterations, self.metrics["total_loss"], label="Total Loss", color="green"
        )
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training Losses")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Learning rate
        axes[0, 1].plot(
            iterations, self.metrics["learning_rate"], marker="o", color="orange"
        )
        axes[0, 1].set_xlabel("Iteration")
        axes[0, 1].set_ylabel("Learning Rate")
        axes[0, 1].set_title("Learning Rate Schedule")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale("log")  # Log scale for better visualization

        # Games and examples
        ax_twin = axes[0, 2].twinx()
        axes[0, 2].bar(
            iterations,
            self.metrics["games_played"],
            alpha=0.7,
            color="skyblue",
            label="Games Played",
        )
        ax_twin.plot(
            iterations,
            self.metrics["training_examples"],
            color="orange",
            marker="o",
            label="Training Examples",
        )
        axes[0, 2].set_xlabel("Iteration")
        axes[0, 2].set_ylabel("Games Played", color="skyblue")
        ax_twin.set_ylabel("Training Examples", color="orange")
        axes[0, 2].set_title("Games and Training Examples")
        axes[0, 2].grid(True, alpha=0.3)

        # Game statistics
        if any(x > 0 for x in self.metrics["avg_game_length"]):
            axes[1, 0].plot(
                iterations, self.metrics["avg_game_length"], marker="o", color="purple"
            )
            axes[1, 0].set_xlabel("Iteration")
            axes[1, 0].set_ylabel("Average Game Length")
            axes[1, 0].set_title("Average Game Length")
            axes[1, 0].grid(True, alpha=0.3)

        # Win rates
        axes[1, 1].plot(
            iterations,
            self.metrics["win_rate_white"],
            label="White Win Rate",
            color="gray",
            marker="o",
        )
        axes[1, 1].plot(
            iterations,
            self.metrics["draw_rate"],
            label="Draw Rate",
            color="brown",
            marker="s",
        )
        axes[1, 1].plot(
            iterations,
            self.metrics["loss_rate"],
            label="Black Win Rate",
            color="black",
            marker="^",
        )
        axes[1, 1].axhline(y=0.5, color="black", linestyle="--", alpha=0.5, label="50%")
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_ylabel("Rate")
        axes[1, 1].set_title("Game Outcomes")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 1)

        # Time per iteration
        axes[1, 2].plot(
            iterations, self.metrics["time_per_iteration"], marker="o", color="green"
        )
        axes[1, 2].set_xlabel("Iteration")
        axes[1, 2].set_ylabel("Time (seconds)")
        axes[1, 2].set_title("Time per Iteration")
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plot:
            plot_path = os.path.join(self.log_dir, f"training_progress.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"Training progress plot saved to: {plot_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def save_metrics(self, filename: str = None):
        """
        Save metrics to JSON file.

        Args:
            filename: Custom filename (optional)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_metrics_{timestamp}.json"

        filepath = os.path.join(self.log_dir, filename)

        # Add metadata
        data = {
            "metrics": self.metrics,
            "metadata": {
                "total_iterations": len(self.metrics["iteration"]),
                "total_games": sum(self.metrics["games_played"]),
                "total_examples": sum(self.metrics["training_examples"]),
                "total_time": sum(self.metrics["time_per_iteration"]),
                "timestamp": datetime.now().isoformat(),
            },
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Metrics saved to: {filepath}")

    def load_metrics(self, filepath: str):
        """
        Load metrics from JSON file.

        Args:
            filepath: Path to metrics file
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        self.metrics = data["metrics"]
        print(f"Metrics loaded from: {filepath}")

    def print_summary(self):
        """Print overall training summary."""
        if len(self.metrics["iteration"]) == 0:
            print("No training data available.")
            return

        total_games = sum(self.metrics["games_played"])
        total_examples = sum(self.metrics["training_examples"])
        total_time = sum(self.metrics["time_per_iteration"])
        avg_time_per_iteration = total_time / len(self.metrics["iteration"])

        final_policy_loss = self.metrics["policy_loss"][-1]
        final_value_loss = self.metrics["value_loss"][-1]
        final_total_loss = self.metrics["total_loss"][-1]

        print(f"\n{'=' * 60}")
        print("TRAINING SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total Iterations: {len(self.metrics['iteration'])}")
        print(f"Total Games Played: {total_games}")
        print(f"Total Training Examples: {total_examples}")
        print(f"Total Training Time: {total_time / 3600:.2f} hours")
        print(f"Average Time per Iteration: {avg_time_per_iteration:.1f}s")
        print(f"Final Policy Loss: {final_policy_loss:.4f}")
        print(f"Final Value Loss: {final_value_loss:.4f}")
        print(f"Final Total Loss: {final_total_loss:.4f}")
        print(f"{'=' * 60}")

    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get the latest metrics."""
        if len(self.metrics["iteration"]) == 0:
            return {}

        return {
            "iteration": self.metrics["iteration"][-1],
            "policy_loss": self.metrics["policy_loss"][-1],
            "value_loss": self.metrics["value_loss"][-1],
            "total_loss": self.metrics["total_loss"][-1],
            "games_played": self.metrics["games_played"][-1],
            "training_examples": self.metrics["training_examples"][-1],
        }
