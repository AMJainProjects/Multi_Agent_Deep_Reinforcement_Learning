"""
Main entry point for MADDQN training with configurable TimesNet implementation.

This script provides a unified interface for training with either the basic
or enhanced TimesNet implementation.
"""

import argparse
import logging
import os
from unified_train import train_maddqn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train MADDQN on financial data')

    # Data parameters
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol for single asset training')
    parser.add_argument('--tickers', type=str, default='DIA,SPY,QQQ', help='Comma-separated list of tickers for mixed dataset')
    parser.add_argument('--start-date', type=str, default='2010-01-01', help='Start date for data')
    parser.add_argument('--end-date', type=str, default='2021-12-31', help='End date for data')
    parser.add_argument('--data-cache', type=str, default='data_cache', help='Directory to cache downloaded data')
    parser.add_argument('--mixed-dataset', action='store_true', help='Use mixed dataset for training')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Ratio of data to use for training')

    # Environment parameters
    parser.add_argument('--window-size', type=int, default=10, help='Window size for observations')
    parser.add_argument('--initial-balance', type=float, default=10000, help='Initial balance for trading')
    parser.add_argument('--transaction-fee', type=float, default=0.001, help='Transaction fee as percentage')

    # MADDQN parameters
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension of the networks')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon-start', type=float, default=0.9, help='Initial exploration rate')
    parser.add_argument('--epsilon-end', type=float, default=0.05, help='Final exploration rate')
    parser.add_argument('--epsilon-decay', type=int, default=200, help='Exploration rate decay steps')
    parser.add_argument('--target-update', type=int, default=10, help='Steps between target network updates')
    parser.add_argument('--buffer-size', type=int, default=1000, help='Replay buffer size')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning-starts', type=int, default=100, help='Steps before learning starts')

    # Training parameters
    parser.add_argument('--num-episodes', type=int, default=100, help='Number of episodes to train')
    parser.add_argument('--log-interval', type=int, default=10, help='Episodes between logging')
    parser.add_argument('--checkpoint-interval', type=int, default=25, help='Episodes between model checkpoints')
    parser.add_argument('--output-dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--use-cuda', action='store_true', help='Use CUDA if available')

    # TimesNet implementation selection
    parser.add_argument('--timesnet-type', type=str, choices=['basic', 'enhanced'], default='basic',
                        help='Type of TimesNet to use (basic or enhanced)')

    # Parse arguments
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Log which TimesNet implementation is being used
    logger.info(f"Using {args.timesnet_type} TimesNet implementation")

    # Train MADDQN
    model, experiment_dir = train_maddqn(args)

    logger.info(f"Training complete. Results saved to {experiment_dir}")

if __name__ == "__main__":
    main()