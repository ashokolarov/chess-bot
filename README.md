# AlphaZero Chess Bot

A minimal implementation of the AlphaZero algorithm for training a chess bot to play at amateur level.

## Features

- **Complete AlphaZero Implementation**: Neural network with policy and value heads, MCTS, self-play data generation
- **Chess Integration**: Uses `python-chess` library for game logic and move validation
- **Training Infrastructure**: Systematic training loop with checkpointing and progress tracking
- **Local Visualization**: Built-in loss tracking and matplotlib-based progress plots
- **Memory Efficient**: Manages training data to prevent memory overflow
- **Configurable**: Easy to adjust hyperparameters for different training scenarios

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Verify installation by running a quick test:
```bash
python -c "import torch, chess, numpy as np, matplotlib.pyplot as plt; print('All dependencies installed successfully!')"
```

## Usage

### Training a Model

Start training with default settings (50 iterations, 10 games per iteration):
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

- `num_iterations`: Number of training iterations (default: 50)
- `games_per_iteration`: Self-play games per iteration (default: 10)
- `mcts_simulations`: MCTS simulations per move (default: 50)
- `max_training_examples`: Maximum training examples to keep in memory (default: 100,000)
- `dirichlet_alpha`: Dirichlet noise concentration parameter (default: 0.25)
- `dirichlet_epsilon`: Mixing ratio between network policy and noise (default: 0.25)
- `epochs_per_iteration`: Neural network training epochs per iteration (default: 5)
- `batch_size`: Training batch size (default: 32)

## Training Process

Each training iteration consists of:

1. **Self-Play Phase**: The current model plays games against itself, generating training examples
   - **Dirichlet Noise**: Random noise is added to encourage exploration of different moves
2. **Training Phase**: The neural network is trained on collected self-play data
3. **Logging**: Progress metrics are recorded and visualized
4. **Checkpointing**: Model state is saved for resuming training

### Dirichlet Noise

The implementation includes Dirichlet noise (a key component of AlphaZero) that:
- **Improves Exploration**: Prevents the bot from getting stuck in local optima early in training
- **Increases Diversity**: Generates more varied training games and positions
- **Better Convergence**: Helps achieve stronger final play by exploring more move sequences

The noise is only applied during training self-play, not during evaluation or human games.

## Output Files

Training produces several outputs:

- `models/`: Directory containing model checkpoints and final trained model
- `logs/`: Directory containing training metrics and progress plots
- `logs/training_progress.png`: Visual progress charts updated after each iteration
- `logs/training_metrics_*.json`: Detailed training metrics in JSON format

## Expected Performance

With default settings:
- **Training Time**: ~2-4 hours on CPU, ~30 minutes on GPU
- **Skill Level**: After 50 iterations, the bot should play at beginner level (~800-1000 ELO)
- **Memory Usage**: ~1-2GB RAM during training
- **Convergence**: Loss should steadily decrease over first 20-30 iterations

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
3. **Longer training** (100+ iterations) for amateur-level strength
4. **GPU acceleration** significantly speeds up training
5. **Adjust learning rate** if loss plateaus or oscillates

## Troubleshooting

**Out of Memory**: Reduce `batch_size`, `games_per_iteration`, or `max_training_examples`
**Slow Training**: Use GPU if available, reduce `mcts_simulations`
**Poor Convergence**: Try adjusting learning rate or increase training data

## Example Training Output

```
Starting AlphaZero Chess Training
Device: cpu
Games per iteration: 10
MCTS simulations: 50
Training iterations: 50

============================================================
STARTING ITERATION 1/50
============================================================
Phase 1: Self-play data generation
Playing game 1/10
...
Generated 1247 training examples from 10 games
Phase 2: Neural network training
Training on 1247 examples for 5 epochs...
Epoch 1/5: Policy Loss: 2.3456, Value Loss: 0.8901, Total Loss: 3.2357
...
```

The bot will gradually improve over iterations, and you can monitor progress through the generated plots and logs. 