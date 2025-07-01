# AlphaZero Chess Bot

A minimal implementation of the AlphaZero algorithm for training a chess bot to play at amateur level.

## Features

- **Complete AlphaZero Implementation**: Neural network with policy and value heads, MCTS, self-play data generation
- **Chess Integration**: Uses `python-chess` library for game logic and move validation
- **Training Infrastructure**: Systematic training loop with checkpointing and progress tracking
- **Tracking and Visualization**: Built-in loss tracking and matplotlib-based progress plots
- **Configurable**: Easy to adjust hyperparameters for different training scenarios


## Usage

### Training a Model

Start training with default settings (set in the config class):
```bash
python main.py --train
```

Customize training parameters:
```bash
python main.py --train --iterations 100 --games 20 --simulations 100 --max-examples 200000 --dirichlet-alpha 0.3 --dirichlet-epsilon 0.25
```

Resume training from a checkpoint:
```bash
python main.py --train --resume models/checkpoint_iter_25.pth
```

### Testing a Trained Model

Test a trained model by watching it play sample games:
```bash
python main.py --test models/final_model.pth
```

### Playing Against the Bot

Play an interactive game against a trained model:
```bash
# Play as White (you move first)
python play_chess.py --model models/final_model.pth --color white

# Play as Black (bot moves first)
python play_chess.py --model models/final_model.pth --color black

# Adjust bot strength (more simulations = stronger but slower)
python play_chess.py --model models/final_model.pth --simulations 200
```

### Configuration

Key training parameters can be adjusted in `main.py`:

- `num_iterations`: Number of training iterations 
- `games_per_iteration`: Self-play games per iteration 
- `mcts_simulations`: MCTS simulations per move 
- `max_training_examples`: Maximum training examples to keep in memory
- `dirichlet_alpha`: Dirichlet noise concentration parameter 
- `dirichlet_epsilon`: Mixing ratio between network policy and noise 
- `epochs_per_iteration`: Neural network training epochs per iteration 
- `batch_size`: Training batch size 
- `mcts_batch_size`  # Batch size for MCTS to perform batch inference instead of feeding them to network separately

## Training Process

Each training iteration consists of:

1. **Self-Play Phase**: The current model plays games against itself, generating training examples
   - **Dirichlet Noise**: Random noise is added to encourage exploration of different moves
2. **Training Phase**: The neural network is trained on collected self-play data
3. **Logging**: Progress metrics are recorded and visualized
4. **Checkpointing**: Model state is saved for resuming training


## Output Files

Training produces several outputs:

- `models/`: Directory containing model checkpoints and final trained model
- `logs/`: Directory containing training metrics and progress plots
- `logs/training_progress.png`: Visual progress charts updated after each iteration
- `logs/training_metrics_*.json`: Detailed training metrics in JSON format

## Architecture

The implementation consists of:

- `src/environment.py`: Chess game wrapper using python-chess
- `src/network.py`: Neural network with ResNet-style architecture
- `src/mcts.py`: Monte Carlo Tree Search implementation
- `src/self_play.py`: Self-play game generation
- `src/trainer.py`: Neural network training loop
- `src/logger.py`: Progress tracking and visualization
- `main.py`: Training orchestration script
- `play_chess.py`: Interactive chess interface for playing against the bot

## Tips for Better Performance

1. **Increase MCTS simulations** for stronger play (but slower training)
2. **More games per iteration** for better data diversity
