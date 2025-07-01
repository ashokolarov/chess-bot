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
from typing import List

import torch

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.config import Config
from src.environment import ChessEnvironment
from src.logger import AlphaZeroLogger
from src.mcts import MCTS
from src.network import AlphaZeroNet
from src.self_play import SelfPlay
from src.trainer import AlphaZeroTrainer


def analyze_game_data(game_results: List[float], num_games: int) -> dict:
    """Analyze game results to extract statistics."""
    if not game_results or num_games == 0:
        return {}

    # Analyze actual game results
    wins = sum(1 for r in game_results if r > 0)
    draws = sum(1 for r in game_results if r == 0)
    losses = sum(1 for r in game_results if r < 0)

    total_games = len(game_results)

    # Calculate rates
    win_rate = wins / total_games if total_games > 0 else 0
    draw_rate = draws / total_games if total_games > 0 else 0
    loss_rate = losses / total_games if total_games > 0 else 0

    return {
        "win_rate_white": win_rate,
        "draw_rate": draw_rate,
        "loss_rate": loss_rate,
        "total_games": total_games,
    }


def train_alphazero(config: Config):
    """
    Main AlphaZero training loop.

    Args:
        config: Configuration object
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
                config.get("directories.models_dir"),
                f"emergency_save_iter_{current_iteration}.pth",
            )
            trainer.save_checkpoint(emergency_save_path, current_iteration)
            print(f"Emergency checkpoint saved: {emergency_save_path}")
        print("Exiting gracefully...")
        sys.exit(0)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    print("Starting AlphaZero Chess Training")
    config.print_config()

    # Create directories
    models_dir = config.get("directories.models_dir")
    logs_dir = config.get("directories.logs_dir")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Initialize components
    neural_net = AlphaZeroNet(
        config.get("network.num_res_blocks"), config.get("network.num_channels")
    )
    trainer = AlphaZeroTrainer(
        neural_net,
        config.get("training.learning_rate"),
        config.get("training.weight_decay"),
        config.get_device(),
        config.get("scheduler.type"),
        config.get_scheduler_config(),
    )
    logger = AlphaZeroLogger(logs_dir)

    # Resume from checkpoint if specified
    start_iteration = 0
    if config.get("training.resume_from") and os.path.exists(
        config.get("training.resume_from")
    ):
        print(f"Resuming training from: {config.get('training.resume_from')}")
        start_iteration = trainer.load_checkpoint(config.get("training.resume_from"))
        print(f"Resumed from iteration {start_iteration}")

    # Training loop
    all_training_examples = []

    try:
        for iteration in range(start_iteration, config.get("training.num_iterations")):
            current_iteration = iteration + 1  # Update for signal handler
            print(f"{'=' * 60}")
            print(
                f"STARTING ITERATION {current_iteration}/{config.get('training.num_iterations')}"
            )
            print(f"{'=' * 60}")

            logger.start_iteration()

            # Create self-play instance
            self_play = SelfPlay(
                neural_net,
                mcts_simulations=config.get("self_play.mcts_simulations"),
                c_puct=config.get("self_play.c_puct"),
                dirichlet_alpha=config.get("dirichlet.alpha"),
                dirichlet_epsilon=config.get("dirichlet.epsilon"),
                mcts_batch_size=config.get("self_play.mcts_batch_size"),
                resign_threshold=config.get("self_play.resign_threshold"),
                initial_temperature=config.get("self_play.initial_temperature"),
                min_temperature=config.get("self_play.min_temperature"),
                move_limit=config.get("self_play.move_limit"),
            )

            # Generate training data
            new_examples, game_results = self_play.generate_training_data(
                config.get("self_play.games_per_iteration"),
                verbose=config.get("verbose", False),
            )

            # Add new examples to the training set
            all_training_examples.extend(new_examples)

            # Keep only recent examples (memory management)
            if len(all_training_examples) > config.get(
                "training.max_training_examples"
            ):
                all_training_examples = all_training_examples[
                    -config.get("training.max_training_examples") :
                ]

            # Train the network
            print(f"\nTraining network on {len(all_training_examples)} examples...")
            history = trainer.train(
                all_training_examples,
                epochs=config.get("training.epochs_per_iteration"),
                batch_size=config.get("training.batch_size"),
                verbose=True,
            )

            # Get final losses from training
            final_policy_loss = history["policy_loss"][-1]
            final_value_loss = history["value_loss"][-1]
            final_total_loss = history["total_loss"][-1]

            current_lr = trainer.get_current_lr()

            # Analyze game data using actual game results
            game_stats = analyze_game_data(
                game_results, config.get("self_play.games_per_iteration")
            )

            # Calculate average game length from training examples
            avg_game_length = (
                len(new_examples) / len(game_results) if game_results else 0
            )

            # Log iteration results
            logger.log_iteration(
                current_iteration,
                final_policy_loss,
                final_value_loss,
                final_total_loss,
                config.get("self_play.games_per_iteration"),
                len(new_examples),
                avg_game_length,
                game_stats.get("win_rate_white"),
                game_stats.get("draw_rate"),
                game_stats.get("loss_rate"),
                current_lr,
            )

            # Save checkpoint
            if (current_iteration) % config.get("training.checkpoint_interval") == 0:
                checkpoint_path = os.path.join(
                    models_dir, f"checkpoint_iter_{current_iteration}.pth"
                )
                trainer.save_checkpoint(checkpoint_path, current_iteration)
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
            models_dir, f"emergency_save_iter_{current_iteration}.pth"
        )
        trainer.save_checkpoint(emergency_save_path, current_iteration)
        print(f"Emergency checkpoint saved: {emergency_save_path}")
        print("Training stopped gracefully.")
        return

    # Final model save
    final_model_path = os.path.join(models_dir, "final_model.pth")
    trainer.save_checkpoint(final_model_path, config.get("training.num_iterations"))
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
        print(f"\nGame {game_idx + 1}/{num_games}\n")
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

    args = parser.parse_args()

    if args.train:
        config = Config("train_config.yaml")
        if args.iterations is not None:
            config.set("training.num_iterations", args.iterations)
        if args.games is not None:
            config.set("self_play.games_per_iteration", args.games)
        if args.simulations is not None:
            config.set("self_play.mcts_simulations", args.simulations)
        if args.epochs is not None:
            config.set("training.epochs_per_iteration", args.epochs)
        if args.batch_size is not None:
            config.set("training.batch_size", args.batch_size)
        if args.batch_size_mcts is not None:
            config.set("self_play.mcts_batch_size", args.batch_size_mcts)
        if args.scheduler is not None:
            config.set("scheduler.type", args.scheduler)
        if args.resume is not None:
            config.set("training.resume_from", args.resume)

        train_alphazero(config)

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
