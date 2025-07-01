#!/usr/bin/env python3
"""
AlphaZero Chess Bot Training Script

This script trains a chess bot using the AlphaZero algorithm with the following components:
- Neural network with policy and value heads
- Monte Carlo Tree Search for move selection
- Self-play for training data generation
- Supervised learning from self-play games
"""

import argparse
import os
import signal
import sys
from typing import List, Tuple

import numpy as np
import torch

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.environment import ChessEnvironment
from src.logger import AlphaZeroLogger
from src.mcts import MCTS
from src.network import AlphaZeroNet
from src.self_play import SelfPlay
from src.trainer import AlphaZeroTrainer


class AlphaZeroTrainingConfig:
    """Configuration for AlphaZero training."""

    def __init__(self):
        # Network architecture
        self.num_res_blocks = 10
        self.num_channels = 256

        # Training parameters
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.batch_size = 32
        self.epochs_per_iteration = 10

        # Learning rate scheduler parameters
        self.scheduler_type = "step"  # Options: "step", "plateau", "cosine"
        self.scheduler_params = {
            "step_size": 5,  # For StepLR: reduce LR every N iterations
            "gamma": 0.8,  # For StepLR: multiply LR by this factor
            "factor": 0.5,  # For ReduceLROnPlateau: factor to reduce LR
            "patience": 3,  # For ReduceLROnPlateau: iterations to wait before reducing
            "T_max": 25,  # For CosineAnnealingLR: maximum iterations
        }

        # Self-play parameters
        self.games_per_iteration = 20
        self.mcts_simulations = 200
        self.c_puct = np.sqrt(2)
        self.mcts_batch_size = 32  # Batch size for neural network inference
        self.resign_threshold = 40

        # Dirichlet noise parameters
        self.dirichlet_alpha = 0.3
        self.dirichlet_epsilon = 0.4

        # Training loop
        self.num_iterations = 25
        self.checkpoint_interval = 5
        self.max_training_examples = 100000  # Replay buffer size

        # Directories
        self.models_dir = "models"
        self.logs_dir = "logs"

        # Device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"


def analyze_game_data(
    training_examples: List[Tuple[np.ndarray, np.ndarray, float]], num_games: int
) -> dict:
    """Analyze training data to extract statistics."""
    if not training_examples or num_games == 0:
        return {}

    rewards = [example[2] for example in training_examples]
    total = len(rewards)
    wins = sum(1 for r in rewards if r > 0)
    draws = sum(1 for r in rewards if r == 0)

    return {
        "avg_game_length": len(training_examples) / num_games if num_games > 0 else 0,
        "win_rate_white": wins / total if total > 0 else 0.5,
        "draw_rate": draws / total if total > 0 else 0,
    }


def train_alphazero(config: AlphaZeroTrainingConfig, resume_from: str = None):
    """
    Main AlphaZero training loop.

    Args:
        config: Training configuration
        resume_from: Path to checkpoint to resume from (optional)
    """

    # Global variables for signal handler
    trainer = None
    current_iteration = 0

    def signal_handler(signum, frame):
        """Handle interruption signals by saving current model state."""
        print(f"\n\nTraining interrupted by signal {signum}")
        if trainer is not None:
            emergency_save_path = os.path.join(
                config.models_dir, f"emergency_save_iter_{current_iteration}.pth"
            )
            trainer.save_checkpoint(emergency_save_path, current_iteration)
            print(f"Emergency checkpoint saved: {emergency_save_path}")
        print("Exiting gracefully...")
        sys.exit(0)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    print("Starting AlphaZero Chess Training")
    print(f"Device: {config.device}")
    print(f"Games per iteration: {config.games_per_iteration}")
    print(f"MCTS simulations: {config.mcts_simulations}")
    print(f"MCTS batch size: {config.mcts_batch_size}")
    print(f"Training iterations: {config.num_iterations}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Scheduler: {config.scheduler_type}")
    if config.scheduler_type == "step":
        print(
            f"  Step size: {config.scheduler_params['step_size']}, Gamma: {config.scheduler_params['gamma']}"
        )
    elif config.scheduler_type == "plateau":
        print(
            f"  Patience: {config.scheduler_params['patience']}, Factor: {config.scheduler_params['factor']}"
        )
    elif config.scheduler_type == "cosine":
        print(f"  T_max: {config.scheduler_params['T_max']}")
    print(f"Dirichlet noise: α={config.dirichlet_alpha}, ε={config.dirichlet_epsilon}")

    # Create directories
    os.makedirs(config.models_dir, exist_ok=True)
    os.makedirs(config.logs_dir, exist_ok=True)

    # Initialize components
    neural_net = AlphaZeroNet(config.num_res_blocks, config.num_channels)
    trainer = AlphaZeroTrainer(
        neural_net,
        config.learning_rate,
        config.weight_decay,
        config.device,
        config.scheduler_type,
        config.scheduler_params,
    )
    logger = AlphaZeroLogger(config.logs_dir)

    # Resume from checkpoint if specified
    start_iteration = 0
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming training from: {resume_from}")
        start_iteration = trainer.load_checkpoint(resume_from)
        print(f"Resumed from iteration {start_iteration}")

    # Training loop
    all_training_examples = []

    try:
        for iteration in range(start_iteration, config.num_iterations):
            current_iteration = iteration + 1  # Update for signal handler
            print(f"\n{'=' * 60}")
            print(f"STARTING ITERATION {current_iteration}/{config.num_iterations}")
            print(f"{'=' * 60}")

            logger.start_iteration()

            # Self-play phase
            print("Phase 1: Self-play data generation")
            self_play = SelfPlay(
                neural_net,
                mcts_simulations=config.mcts_simulations,
                c_puct=config.c_puct,
                dirichlet_alpha=config.dirichlet_alpha,
                dirichlet_epsilon=config.dirichlet_epsilon,
                mcts_batch_size=config.mcts_batch_size,
                resign_threshold=config.resign_threshold,
            )

            # Generate training data
            new_examples = self_play.generate_training_data(
                config.games_per_iteration, verbose=True
            )

            # Add new examples to training data
            all_training_examples.extend(new_examples)

            # Keep only recent examples (memory management)
            if len(all_training_examples) > config.max_training_examples:
                all_training_examples = all_training_examples[
                    -config.max_training_examples :
                ]

            print(f"Total training examples: {len(all_training_examples)}")

            # Training phase
            print("Phase 2: Neural network training")
            history = trainer.train(
                all_training_examples,
                epochs=config.epochs_per_iteration,
                batch_size=config.batch_size,
                verbose=True,
            )

            # Get final losses from training
            final_policy_loss = history["policy_loss"][-1]
            final_value_loss = history["value_loss"][-1]
            final_total_loss = history["total_loss"][-1]

            # Step the learning rate scheduler
            if config.scheduler_type == "plateau":
                trainer.step_scheduler(final_total_loss)
            else:
                trainer.step_scheduler()

            current_lr = trainer.get_current_lr()
            print(f"Learning rate after iteration {iteration + 1}: {current_lr:.6f}")

            # Analyze game data
            game_stats = analyze_game_data(new_examples, config.games_per_iteration)

            # Log iteration results
            logger.log_iteration(
                iteration + 1,
                final_policy_loss,
                final_value_loss,
                final_total_loss,
                config.games_per_iteration,
                len(new_examples),
                game_stats.get("avg_game_length"),
                game_stats.get("win_rate_white"),
                game_stats.get("draw_rate"),
                current_lr,
            )

            # Save checkpoint
            if (iteration + 1) % config.checkpoint_interval == 0:
                checkpoint_path = os.path.join(
                    config.models_dir, f"checkpoint_iter_{iteration + 1}.pth"
                )
                trainer.save_checkpoint(checkpoint_path, iteration + 1)
                print(f"Checkpoint saved: {checkpoint_path}")

            # Save progress visualization
            logger.plot_training_progress(save_plot=True, show_plot=False)

            # Clean up memory
            del new_examples
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print(f"\n\nTraining interrupted by user at iteration {current_iteration}")
        emergency_save_path = os.path.join(
            config.models_dir, f"emergency_save_iter_{current_iteration}.pth"
        )
        trainer.save_checkpoint(emergency_save_path, current_iteration)
        print(f"Emergency checkpoint saved: {emergency_save_path}")
        print("Training stopped gracefully.")
        return

    # Final model save
    final_model_path = os.path.join(config.models_dir, "final_model.pth")
    trainer.save_checkpoint(final_model_path, config.num_iterations)
    print(f"\nFinal model saved: {final_model_path}")

    # Final summary
    logger.print_summary()
    logger.save_metrics()
    logger.plot_training_progress(save_plot=True, show_plot=False)

    print("\nTraining completed!")


def test_model(model_path: str, num_games: int = 5):
    """
    Test a trained model by playing sample games.

    Args:
        model_path: Path to trained model
        num_games: Number of test games to play
    """
    print(f"Testing model: {model_path}")

    # Load model
    neural_net = AlphaZeroNet()
    neural_net.load_checkpoint(model_path)
    neural_net.eval()

    # Create MCTS player
    mcts_player = MCTS(
        neural_net,
        c_puct=1.0,
        num_simulations=100,
        dirichlet_alpha=0.25,
        dirichlet_epsilon=0.0,  # No noise for testing
        batch_size=1,
    )

    # Play test games
    for game_idx in range(num_games):
        print(f"\nGame {game_idx + 1}/{num_games}")
        env = ChessEnvironment()
        move_count = 0

        while not env.is_game_over() and move_count < 200:
            # Get best move
            best_move = mcts_player.get_best_move(env, temperature=0.0)

            # Make move
            env.make_move(best_move)
            move_count += 1

            if move_count % 10 == 0:
                print(f"Move {move_count}: {best_move}")

        # Print result
        result = env.get_result()
        if result == 1.0:
            print("White wins!")
        elif result == -1.0:
            print("Black wins!")
        else:
            print("Draw!")

        print(f"Game length: {move_count} moves")
        print(f"Final position: {env.get_fen()}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AlphaZero Chess Training")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument(
        "--test", type=str, help="Test a trained model (provide model path)"
    )
    parser.add_argument("--resume", type=str, help="Resume training from checkpoint")
    parser.add_argument(
        "--iterations", type=int, default=None, help="Number of training iterations"
    )
    parser.add_argument("--games", type=int, default=None, help="Games per iteration")
    parser.add_argument(
        "--simulations", type=int, default=None, help="MCTS simulations per move"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Epochs per iteration")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Training batch size"
    )
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=None,
        help="Dirichlet noise alpha parameter",
    )
    parser.add_argument(
        "--dirichlet-epsilon",
        type=float,
        default=None,
        help="Dirichlet noise epsilon parameter",
    )
    parser.add_argument(
        "--batch-size-mcts",
        type=int,
        default=None,
        help="Batch size for MCTS neural network inference",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["step", "plateau", "cosine"],
        default=None,
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--scheduler-step-size",
        type=int,
        default=None,
        help="Step size for StepLR scheduler",
    )
    parser.add_argument(
        "--scheduler-gamma",
        type=float,
        default=None,
        help="Gamma factor for StepLR scheduler",
    )
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=None,
        help="Patience for ReduceLROnPlateau scheduler",
    )

    args = parser.parse_args()

    if args.train:
        config = AlphaZeroTrainingConfig()
        if args.iterations is not None:
            config.num_iterations = args.iterations
        if args.games is not None:
            config.games_per_iteration = args.games
        if args.simulations is not None:
            config.mcts_simulations = args.simulations
        if args.epochs is not None:
            config.epochs_per_iteration = args.epochs
        if args.batch_size is not None:
            config.batch_size = args.batch_size
        if args.dirichlet_alpha is not None:
            config.dirichlet_alpha = args.dirichlet_alpha
        if args.dirichlet_epsilon is not None:
            config.dirichlet_epsilon = args.dirichlet_epsilon
        if args.batch_size_mcts is not None:
            config.mcts_batch_size = args.batch_size_mcts
        if args.scheduler is not None:
            config.scheduler_type = args.scheduler
        if args.scheduler_step_size is not None:
            config.scheduler_params["step_size"] = args.scheduler_step_size
        if args.scheduler_gamma is not None:
            config.scheduler_params["gamma"] = args.scheduler_gamma
        if args.scheduler_patience is not None:
            config.scheduler_params["patience"] = args.scheduler_patience

        train_alphazero(config, resume_from=args.resume)

    elif args.test:
        if not os.path.exists(args.test):
            print(f"Model file not found: {args.test}")
            return
        test_model(args.test, num_games=5)

    else:
        print("Please specify --train or --test")
        print("Use --help for more options")


if __name__ == "__main__":
    main()
