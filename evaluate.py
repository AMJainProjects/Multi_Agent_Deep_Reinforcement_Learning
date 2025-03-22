import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import logging
import json
from collections import defaultdict

from data_loader import FinancialDataLoader
from environment import TradingEnvironment
from structure.timesnet_factory import create_maddqn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def calculate_metrics(env):
    """
    Calculate performance metrics from environment history.

    Args:
        env: Trading environment with completed trading history

    Returns:
        Dictionary of performance metrics
    """
    # Convert trade history to DataFrame
    trade_df = pd.DataFrame(env.trade_history)

    # Skip if no trades were made
    if len(trade_df) == 0:
        return {
            'cumulative_return': 0.0,
            'annualized_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'total_trades': 0
        }

    # Calculate total return
    initial_value = env.initial_balance
    final_value = env.total_value
    cumulative_return = (final_value - initial_value) / initial_value * 100

    # Calculate annualized return
    days = len(trade_df)
    annual_factor = 252 / days  # Assuming 252 trading days per year
    annualized_return = ((1 + cumulative_return / 100) ** annual_factor - 1) * 100

    # Calculate Sharpe ratio (using returns)
    if 'return' in trade_df.columns:
        returns = trade_df['return'].values
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return != 0:
            sharpe_ratio = mean_return / std_return * np.sqrt(252)  # Annualized Sharpe
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0

    # Calculate maximum drawdown
    if 'total_value' in trade_df.columns:
        max_values = trade_df['total_value'].cummax()
        drawdowns = (trade_df['total_value'] - max_values) / max_values * 100
        max_drawdown = abs(drawdowns.min())
    else:
        max_drawdown = 0.0

    # Calculate win rate
    if 'return' in trade_df.columns:
        positive_trades = (trade_df['return'] > 0).sum()
        total_trades = len(trade_df)
        win_rate = positive_trades / total_trades * 100 if total_trades > 0 else 0.0
    else:
        win_rate = 0.0
        total_trades = 0

    return {
        'cumulative_return': cumulative_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': total_trades
    }


def load_model_config(model_path, device='cpu'):
    """
    Load model configuration from checkpoint.

    Args:
        model_path: Path to model checkpoint
        device: Device to load the model on

    Returns:
        Model configuration
    """
    checkpoint = torch.load(model_path, map_location=device)
    return checkpoint.get('args', {})


def evaluate_model(args):
    """
    Evaluate a trained MADDQN model on test data.

    Args:
        args: Command-line arguments

    Returns:
        Dictionary of evaluation metrics
    """
    # Set device
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model configuration to determine TimesNet type
    model_config = load_model_config(args.model_path, device)
    timesnet_type = model_config.get('timesnet_type', 'basic')

    logger.info(f"Model uses {timesnet_type} TimesNet implementation")

    # Load and preprocess data
    logger.info("Loading and preprocessing data")
    data_loader = FinancialDataLoader(cache_dir=args.data_cache)

    results = {}

    # Process tickers
    if args.tickers:
        tickers = args.tickers.split(',')
    else:
        tickers = [args.ticker]

    for ticker in tickers:
        logger.info(f"Evaluating on ticker: {ticker}")

        # Load data
        data = data_loader.load_single_asset(ticker, args.start_date, args.end_date)
        processed_data = data_loader.preprocess_data(data)

        # Split into training and testing sets
        train_data, test_data = data_loader.split_train_test(processed_data, args.train_ratio)

        # Create trading environment
        env = TradingEnvironment(
            test_data,  # Use test data for evaluation
            window_size=args.window_size,
            initial_balance=args.initial_balance,
            transaction_fee=args.transaction_fee
        )

        # Initialize MADDQN
        state_dim = (args.window_size, env.normalized_data.shape[1])
        action_dim = 3  # sell, hold, buy

        # Load pretrained model using factory
        logger.info(f"Loading model from {args.model_path}")

        # Load checkpoint
        checkpoint = torch.load(args.model_path, map_location=device)

        # Create MADDQN with correct TimesNet type
        maddqn = create_maddqn(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=model_config.get('hidden_dim', 128),
            learning_rate=model_config.get('learning_rate', 0.001),
            gamma=model_config.get('gamma', 0.99),
            epsilon_start=model_config.get('epsilon_start', 0.9),
            epsilon_end=model_config.get('epsilon_end', 0.05),
            epsilon_decay=model_config.get('epsilon_decay', 200),
            target_update=model_config.get('target_update', 10),
            buffer_size=model_config.get('buffer_size', 1000),
            batch_size=model_config.get('batch_size', 64),
            device=device,
            timesnet_type=timesnet_type
        )

        # Load model weights
        maddqn.risk_agent.policy_net.load_state_dict(checkpoint['risk_agent_state_dict'])
        maddqn.return_agent.policy_net.load_state_dict(checkpoint['return_agent_state_dict'])
        maddqn.final_agent.policy_net.load_state_dict(checkpoint['final_agent_state_dict'])

        # Set to evaluation mode
        maddqn.risk_agent.policy_net.eval()
        maddqn.return_agent.policy_net.eval()
        maddqn.final_agent.policy_net.eval()

        # Set epsilon to 0 for deterministic actions
        for agent_attr in ['risk_agent', 'return_agent', 'final_agent']:
            agent = getattr(maddqn, agent_attr)
            agent.epsilon_start = 0
            agent.epsilon_end = 0

        # Run evaluation episode
        state = env.reset()
        done = False

        # Record actions for analysis
        actions_history = []
        risk_actions = []
        return_actions = []
        final_actions = []
        prices = []

        logger.info("Starting evaluation episode")

        with torch.no_grad():
            while not done:
                # Select action
                action, risk_q_values, return_q_values = maddqn.select_action(state, epsilon=0.0)  # No exploration

                # Get individual agent actions (for analysis)
                risk_action = np.argmax(risk_q_values)
                return_action = np.argmax(return_q_values)

                # Take action in environment
                next_state, reward, done, info = env.step(action)

                # Record data for analysis
                actions_history.append({
                    'step': env.current_step,
                    'price': info['price'],
                    'risk_action': risk_action,
                    'return_action': return_action,
                    'final_action': action,
                    'position': info['position'],
                    'reward': reward,
                    'total_value': info['total_value']
                })

                risk_actions.append(risk_action)
                return_actions.append(return_action)
                final_actions.append(action)
                prices.append(info['price'])

                # Update state
                state = next_state

        # Calculate performance metrics
        metrics = calculate_metrics(env)
        results[ticker] = metrics

        logger.info(f"Results for {ticker}:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.2f}")

        # Save actions history for this ticker
        actions_df = pd.DataFrame(actions_history)
        actions_df.to_csv(os.path.join(args.output_dir, f"{ticker}_actions.csv"), index=False)

        # Plot results
        plt.figure(figsize=(15, 10))

        # Plot price with actions
        plt.subplot(2, 1, 1)
        plt.plot(prices, label='Price')
        plt.title(f"{ticker} Price and Actions")

        # Mark buy actions
        buy_indices = [i for i, a in enumerate(final_actions) if a == 2]
        buy_prices = [prices[i] for i in buy_indices]
        if buy_indices:
            plt.scatter(buy_indices, buy_prices, color='green', marker='^', label='Buy')

        # Mark sell actions
        sell_indices = [i for i, a in enumerate(final_actions) if a == 0]
        sell_prices = [prices[i] for i in sell_indices]
        if sell_indices:
            plt.scatter(sell_indices, sell_prices, color='red', marker='v', label='Sell')

        plt.legend()

        # Plot portfolio value
        plt.subplot(2, 1, 2)
        total_values = [item['total_value'] for item in actions_history]
        plt.plot(total_values, label='Portfolio Value')
        plt.title(f"{ticker} Portfolio Value")
        plt.xlabel('Step')
        plt.ylabel('Value')

        # Add metrics text
        plt.figtext(0.01, 0.01, f"Cumulative Return: {metrics['cumulative_return']:.2f}%\n"
                                f"Annualized Return: {metrics['annualized_return']:.2f}%\n"
                                f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
                                f"Max Drawdown: {metrics['max_drawdown']:.2f}%\n"
                                f"Win Rate: {metrics['win_rate']:.2f}%\n"
                                f"Total Trades: {metrics['total_trades']}\n",
                    fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"{ticker}_evaluation.png"))

        # Analyze agent agreement
        agreement_count = sum(1 for ra, rta, fa in zip(risk_actions, return_actions, final_actions)
                              if ra == rta and ra == fa)
        agreement_rate = agreement_count / len(final_actions) * 100

        risk_final_agreement = sum(1 for ra, fa in zip(risk_actions, final_actions) if ra == fa)
        risk_final_rate = risk_final_agreement / len(final_actions) * 100

        return_final_agreement = sum(1 for rta, fa in zip(return_actions, final_actions) if rta == fa)
        return_final_rate = return_final_agreement / len(final_actions) * 100

        logger.info(f"Agent Agreement Analysis for {ticker}:")
        logger.info(f"  All agents agree: {agreement_rate:.2f}%")
        logger.info(f"  Risk-Final agree: {risk_final_rate:.2f}%")
        logger.info(f"  Return-Final agree: {return_final_rate:.2f}%")

    # Calculate average metrics across all tickers
    if len(tickers) > 1:
        avg_metrics = defaultdict(float)
        for ticker, metrics in results.items():
            for metric, value in metrics.items():
                avg_metrics[metric] += value

        for metric in avg_metrics:
            avg_metrics[metric] /= len(tickers)

        logger.info("Average metrics across all tickers:")
        for metric, value in avg_metrics.items():
            logger.info(f"  {metric}: {value:.2f}")

        results['average'] = dict(avg_metrics)

    # Save all results to JSON
    with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate MADDQN on financial data')

    # Data parameters
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol for evaluation')
    parser.add_argument('--tickers', type=str, default=None,
                        help='Comma-separated list of tickers for batch evaluation')
    parser.add_argument('--start-date', type=str, default='2010-01-01', help='Start date for data')
    parser.add_argument('--end-date', type=str, default='2021-12-31', help='End date for data')
    parser.add_argument('--data-cache', type=str, default='data_cache', help='Directory to cache downloaded data')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Ratio of data to use for training')

    # Environment parameters
    parser.add_argument('--window-size', type=int, default=10, help='Window size for observations')
    parser.add_argument('--initial-balance', type=float, default=10000, help='Initial balance for trading')
    parser.add_argument('--transaction-fee', type=float, default=0.001, help='Transaction fee as percentage')

    # Model parameters
    parser.add_argument('--model-path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results', help='Directory to save results')
    parser.add_argument('--use-cuda', action='store_true', help='Use CUDA if available')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Evaluate MADDQN
    results = evaluate_model(args)

    logger.info(f"Evaluation complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()