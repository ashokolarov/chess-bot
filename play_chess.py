#!/usr/bin/env python3
"""
Interactive Chess Interface for playing against AlphaZero bot.
"""

import os
import sys

import chess

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.environment import ChessEnvironment
from src.mcts import MCTS
from src.network import AlphaZeroNet


def print_board(board):
    """Print the chess board in a readable format."""
    print("\n" + "=" * 50)
    print("  a b c d e f g h")
    print("  ---------------")

    for rank in range(7, -1, -1):
        row = f"{rank + 1} "
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece is None:
                row += ". "
            else:
                if piece.color == chess.WHITE:
                    row += piece.symbol().upper() + " "
                else:
                    row += piece.symbol().lower() + " "
        row += f" {rank + 1}"
        print(row)

    print("  ---------------")
    print("  a b c d e f g h")
    print("=" * 50)


def get_user_move(board):
    """Get a valid move from the user."""
    legal_moves = list(board.legal_moves)

    if len(legal_moves) == 0:
        return None

    print(f"\nLegal moves: {', '.join([str(move) for move in legal_moves[:10]])}")
    if len(legal_moves) > 10:
        print(f"... and {len(legal_moves) - 10} more")

    while True:
        try:
            move_str = input(
                "\nEnter your move (e.g., 'e2e4' or 'O-O' for castling): "
            ).strip()

            if move_str.lower() in ["quit", "exit", "q"]:
                return None

            # Try to parse the move
            move = chess.Move.from_uci(move_str)

            if move in legal_moves:
                return move
            else:
                print(f"❌ Invalid move: {move_str}")
                print("Please enter a legal move from the list above.")

        except ValueError:
            print(f"❌ Invalid move format: {move_str}")
            print("Please use format like 'e2e4', 'g1f3', etc.")


def play_against_alphazero(
    model_path: str, user_plays_white: bool = True, mcts_simulations: int = 100
):
    """
    Play a game against the AlphaZero bot.

    Args:
        model_path: Path to trained model
        user_plays_white: Whether user plays white pieces
        mcts_simulations: Number of MCTS simulations for bot moves
    """
    print("🎮 AlphaZero Chess Interface")
    print("=" * 50)

    # Load the trained model
    print(f"Loading model from: {model_path}")
    neural_net = AlphaZeroNet(
        num_res_blocks=8, num_channels=64
    )  # Match training config
    neural_net.load_checkpoint(model_path)
    neural_net.eval()

    # Create MCTS player
    mcts_player = MCTS(neural_net, num_simulations=mcts_simulations)

    # Initialize game
    env = ChessEnvironment()
    board = env.board

    print(f"\nYou are playing {'White' if user_plays_white else 'Black'}")
    print("AlphaZero is playing " + ("Black" if user_plays_white else "White"))
    print("\nCommands: 'quit' to exit, 'resign' to resign")

    move_count = 0

    while not board.is_game_over():
        print_board(board)
        print(f"\nMove {move_count + 1}")
        print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")

        if (board.turn == chess.WHITE) == user_plays_white:
            # User's turn
            print("\n🤔 Your turn...")
            move = get_user_move(board)

            if move is None:
                print("👋 Game ended by user.")
                return

            print(f"✅ You played: {move}")

        else:
            # AlphaZero's turn
            print("\n🤖 AlphaZero is thinking...")
            move = mcts_player.get_best_move(env, temperature=0.0)
            print(f"🤖 AlphaZero plays: {move}")

        # Make the move
        env.make_move(move)
        move_count += 1

        # Check for game end conditions
        if board.is_checkmate():
            winner = "White" if not board.turn else "Black"
            print_board(board)
            print(f"\n🏆 Checkmate! {winner} wins!")
            break
        elif board.is_stalemate():
            print_board(board)
            print(f"\n🤝 Stalemate! It's a draw!")
            break
        elif board.is_insufficient_material():
            print_board(board)
            print(f"\n🤝 Insufficient material! It's a draw!")
            break
        elif board.is_fifty_moves():
            print_board(board)
            print(f"\n🤝 Fifty-move rule! It's a draw!")
            break
        elif board.is_repetition(3):
            print_board(board)
            print(f"\n🤝 Threefold repetition! It's a draw!")
            break
        elif board.is_check():
            print("⚡ Check!")

    print(f"\nGame finished after {move_count} moves.")
    print(f"Final position: {board.fen()}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Play Chess against AlphaZero")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--color",
        type=str,
        choices=["white", "black"],
        default="white",
        help="Color to play (default: white)",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=100,
        help="MCTS simulations per move (default: 100)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        print("Please train a model first with: python main.py --train")
        return

    user_plays_white = args.color.lower() == "white"

    try:
        play_against_alphazero(args.model, user_plays_white, args.simulations)
    except KeyboardInterrupt:
        print("\n\n👋 Game interrupted. Thanks for playing!")
    except Exception as e:
        print(f"\n❌ Error during game: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
