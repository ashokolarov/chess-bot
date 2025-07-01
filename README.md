# AlphaZero Chess Bot

A minimal implementation of the AlphaZero algorithm for training a chess bot to play at amateur level.

## Features

- **Complete AlphaZero Implementation**: Neural network with policy and value heads, MCTS, self-play data generation
- **Chess Integration**: Uses `python-chess` library for game logic and move validation
- **Training Infrastructure**: Systematic training loop with checkpointing and progress tracking
- **Tracking and Visualization**: Built-in loss tracking and matplotlib-based progress plots
- **Configurable**: Easy to adjust hyperparameters for different training scenarios


## Usage

### Training

Start training with default configuration:

```bash
python main.py --train
```

Override configuration parameters:

```bash
python main.py --train --iterations 10 --games 50 --simulations 400
```

Resume training from a checkpoint:

```bash
python main.py --train --resume models/checkpoint_iter_5.pth
```

### Playing Against the Bot

Play against a trained model:

```bash
python play_chess.py --model models/final_model.pth
```

Play as black:

```bash
python play_chess.py --model models/final_model.pth --color black
```

Adjust MCTS simulations:

```bash
python play_chess.py --model models/final_model.pth --simulations 500
```

### Configuration

Modify `config.yaml` to adjust training parameters:

```yaml
# Increase training intensity
training_loop:
  num_iterations: 50
  games_per_iteration: 100

# Adjust network size
network:
  num_res_blocks: 20
  num_channels: 512

# Change exploration
self_play:
  mcts_simulations: 400
  c_puct: 2.0
```

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

