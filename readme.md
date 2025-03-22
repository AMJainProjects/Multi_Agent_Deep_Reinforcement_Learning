# MADDQN: Multi-Agent Reinforcement Learning for Financial Trading

This repository implements the Multi-Agent Double Deep Q-Network (MADDQN) framework described in the paper "A multi-agent reinforcement learning framework for optimizing financial trading strategies based on TimesNet" (Huang et al., 2024).

## Overview

MADDQN is a novel multi-agent reinforcement learning approach for financial trading that balances the pursuit of maximum revenue with risk avoidance. The framework uses two specialized agents with different investment preferences:

1. **Risk Agent**: Uses the Sharpe ratio to balance risk and return
2. **Return Agent**: Focuses on maximizing mid-term returns
3. **Final Agent**: Combines the "advice" from both agents to make optimal trading decisions

The implementation employs state-of-the-art time series models:
- TimesNet for Risk and Return agents (time series feature extraction)
- Multi-Scale CNN for the Final Agent (combines inputs from sub-agents)

## Key Features

- **Multi-agent framework** that balances risk and return
- **TimesNet** implementation for financial time series processing
- **Hierarchical decision-making** structure
- **Double DQN algorithm** for stable learning
- **Comparative evaluation** with traditional trading strategies
- **Mixed dataset support** for generalized model training

## Requirements

```
numpy
pandas
torch
matplotlib
seaborn
yfinance
tqdm
```

## Project Structure

- `environment.py`: Trading environment simulation
- `timesnet.py`: TimesNet model for time series processing
- `multiscale_cnn.py`: Multi-scale CNN implementation for the final agent
- `maddqn.py`: Core MADDQN implementation 
- `data_loader.py`: Financial data loading and preprocessing
- `train.py`: Script for training the MADDQN model
- `evaluate.py`: Script for evaluating the trained model
- `compare_methods.py`: Comparative evaluation with baseline methods
- `demo.py`: Quick demo with synthetic data

## Usage

### Quick Demo

To run a quick demo with synthetic data:

```bash
python demo.py
```

### Training

To train the MADDQN model on a specific ticker:

```bash
python train_basic.py --ticker SPY --start-date 2010-01-01 --end-date 2021-12-31 --num-episodes 100
```

For training on a mixed dataset (as in the paper):

```bash
python train_basic.py --mixed-dataset --tickers DIA,SPY,QQQ --start-date 2010-01-01 --end-date 2021-12-31 --num-episodes 100
```

### Evaluation

To evaluate a trained model:

```bash
python evaluate.py --ticker SPY --start-date 2020-01-01 --end-date 2021-12-31 --model-path results/model_final.pt
```

### Comparing with Baselines

To compare MADDQN with baseline methods:

```bash
python compare_methods.py --tickers SPY,AAPL,MSFT --start-date 2020-01-01 --end-date 2021-12-31 --model-path results/model_final.pt
```

## Model Details

### Risk Agent

The Risk Agent uses the Sharpe ratio as a reward function to balance risk and return:

```
Risk_Agent_Reward = position * (mean(returns) / std(returns))
```

### Return Agent

The Return Agent focuses on mid-term returns:

```
Return_Agent_Reward = position * ((price_t+n - price_t) / price_t * 100)
```

### Final Agent

The Final Agent uses short-term profit as its reward function:

```
Final_Agent_Reward = position * ((price_t+1 - price_t) / price_t * 100)
```

It takes both the state and the Q-values from the sub-agents to make trading decisions.

## Results

The MADDQN framework demonstrates superior performance compared to traditional trading strategies across various financial assets, showing:

1. Higher cumulative returns
2. Lower maximum drawdowns
3. Better risk-adjusted performance (Sharpe ratio)
4. Improved generalization across different assets

## Paper Reference

```
@article{huang2024multi,
  title={A multi-agent reinforcement learning framework for optimizing financial trading strategies based on TimesNet},
  author={Huang, Yuling and Zhou, Chujin and Cui, Kai and Lu, Xiaoping},
  journal={Expert Systems with Applications},
  volume={237},
  pages={121502},
  year={2024},
  publisher={Elsevier}
}
```

## License

This project is for research purposes only. It is not financial advice.

## Acknowledgements

This implementation is based on the paper by Huang et al. The TimesNet implementation is adapted from the original paper "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis" (Wu et al., 2023).
