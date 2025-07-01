from typing import List, Optional, Tuple

import chess
import chess.engine
import numpy as np


class ChessEnvironment:
    """
    Chess environment wrapper using python-chess library.
    Handles board representation, legal moves, and game state.
    """

    def __init__(self):
        self.board = chess.Board()
        self.history = []

    def reset(self):
        """Reset the game to starting position."""
        self.board = chess.Board()
        self.history = []
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """
        Convert board position to neural network input format.
        Returns 8x8x12 array (6 piece types x 2 colors).
        """
        state = np.zeros((8, 8, 12), dtype=np.float32)

        piece_map = {
            chess.PAWN: 0,
            chess.ROOK: 1,
            chess.KNIGHT: 2,
            chess.BISHOP: 3,
            chess.QUEEN: 4,
            chess.KING: 5,
        }

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row, col = divmod(square, 8)
                piece_type = piece_map[piece.piece_type]
                color_offset = 0 if piece.color == chess.WHITE else 6
                state[row, col, piece_type + color_offset] = 1.0

        return state

    def get_legal_moves(self) -> List[chess.Move]:
        """Get list of legal moves in current position."""
        return list(self.board.legal_moves)

    def get_legal_moves_mask(self) -> np.ndarray:
        """
        Get binary mask for legal moves.
        Returns array of size 4096 (64*64 possible from-to combinations).
        """
        mask = np.zeros(4096, dtype=np.float32)
        for move in self.board.legal_moves:
            move_idx = move.from_square * 64 + move.to_square
            mask[move_idx] = 1.0
        return mask

    def move_to_index(self, move: chess.Move) -> int:
        """Convert chess move to array index."""
        return move.from_square * 64 + move.to_square

    def index_to_move(self, idx: int) -> chess.Move:
        """Convert array index to chess move."""
        from_square = idx // 64
        to_square = idx % 64
        return chess.Move(from_square, to_square)

    def make_move(self, move: chess.Move) -> Tuple[np.ndarray, float, bool]:
        """
        Make a move and return new state, reward, and done flag.
        """
        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {move}")

        self.history.append(self.board.copy())
        self.board.push(move)

        # Calculate reward
        reward = 0.0
        done = False

        if self.board.is_checkmate():
            # Winner gets +1, loser gets -1
            reward = 1.0 if not self.board.turn else -1.0
            done = True
        elif (
            self.board.is_stalemate()
            or self.board.is_insufficient_material()
            or self.board.is_repetition(3)
            or self.board.is_fifty_moves()
        ):
            reward = 0.0  # Draw
            done = True

        return self.get_state(), reward, done

    def is_game_over(self) -> bool:
        """Check if game is over."""
        return self.board.is_game_over()

    def get_result(self) -> Optional[float]:
        """
        Get game result from current player's perspective.
        Returns: 1.0 for win, -1.0 for loss, 0.0 for draw, None if game not over.
        """
        if not self.board.is_game_over():
            return None

        result = self.board.result()
        if result == "1-0":  # White wins
            return 1.0 if self.board.turn == chess.BLACK else -1.0
        elif result == "0-1":  # Black wins
            return 1.0 if self.board.turn == chess.WHITE else -1.0
        else:  # Draw
            return 0.0

    def copy(self):
        """Create a copy of the current environment."""
        env_copy = ChessEnvironment()
        env_copy.board = self.board.copy()
        env_copy.history = self.history.copy()
        return env_copy

    def get_fen(self) -> str:
        """Get FEN string of current position."""
        return self.board.fen()

    def from_fen(self, fen: str):
        """Set position from FEN string."""
        self.board = chess.Board(fen)
        self.history = []
