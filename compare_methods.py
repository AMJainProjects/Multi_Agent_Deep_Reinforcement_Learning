import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import logging
from collections import defaultdict
import seaborn as sns

from data_loader import FinancialDataLoader
from environment import TradingEnvironment
from structure.timesnet_factory import create_maddqn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("comparison.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BuyAndHoldStrategy:
    """Simple buy and hold strategy."""

    def __init__(self, action_dim=3):
        self.action_dim = action_dim

    def select_action(self, state):
        """Always buy (action = 2)"""
        return 2


class SellAndHoldStrategy:
    """Simple sell and hold strategy."""

    def __init__(self, action_dim=3):
        self.action_dim = action_dim

    def select_action(self, state):
        """Always sell (action = 0)"""
        return 0


class MovingAverageTrendFollowing:
    """Moving average trend following strategy."""

    def __init__(self, action_dim=3, window_size=10, ma_short=5, ma_long=20):
        self.action_dim = action_dim
        self.window_size = window_size
        self.ma_short = ma_short
        self.ma_long = ma_long

    def select_action(self, state):
        """
        Buy if short MA crosses above long MA, sell if it crosses below.

        Args:
            state: Window of price data [window_size, features]

        Returns:
            action: 0 (sell), 1 (hold), or 2 (buy)
        """
        # Extract close prices from state
        # Assuming close price is the 4th feature (index 3)
        close_prices = state[:, 3]

        # Calculate MAs (if window is smaller than MA period, use whole window)
        ma_short_period = min(self.ma_short, len(close_prices))
        ma_long_period = min(self.ma_long, len(close_prices))

        ma_short_value = np.mean(close_prices[-ma_short_period:])
        ma_long_value = np.mean(close_prices[-ma_long_period:])

        # Check prior values to detect crossover
        if len(close_prices) > 1:
            prev_close_prices = close_prices[:-1]
            prev_ma_short_value = np.mean(prev_close_prices[-ma_short_period:])
            prev_ma_long_value = np.mean(prev_close_prices[-ma_long_period:])

            # Buy if short MA crosses above long MA
            if ma_short_value > ma_long_value and prev_ma_short_value <= prev_ma_long_value:
                return 2  # Buy

            # Sell if short MA crosses below long MA
            elif ma_short_value < ma_long_value and prev_ma_short_value >= prev_ma_long_value:
                return 0  # Sell

        # Otherwise hold
        return 1  # Hold


class MovingAverageMeanReversion:
    """Moving average mean reversion strategy."""

    def __init__(self, action_dim=3, window_size=10, ma_period=20, threshold=0.05):
        self.action_dim = action_dim
        self.window_size = window_size
        self.ma_period = ma_period
        self.threshold = threshold

    def select_action(self, state):
        """
        Buy if price drops significantly below MA, sell if it rises above.

        Args:
            state: Window of price data [window_size, features]

        Returns:
            action: 0 (sell), 1 (hold), or 2 (buy)
        """
        # Extract close prices from state
        # Assuming close price is the 4th feature (index 3)
        close_prices = state[:, 3]

        # Calculate MA (if window is smaller than MA period, use whole window)
        ma_period = min(self.ma_period, len(close_prices))
        ma_value = np.mean(close_prices[-ma_period:])

        # Calculate current price
        current_price = close_prices[-1]

        # Calculate deviation from MA
        deviation = (current_price - ma_value) / ma_value

        # Buy if price is below MA by threshold
        if deviation < -self.threshold:
            return 2  # Buy

        # Sell if price is above MA by threshold
        elif deviation > self.threshold:
            return 0  # Sell

        # Otherwise hold
        return 1  # Hold


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


def run_strategy(env, strategy):
    """
    Run a trading strategy in the environment.

    Args:
        env: Trading environment
        strategy: Trading strategy object with select_action method

    Returns:
        Dictionary of performance metrics
    """
    state = env.reset()
    done = False

    while not done:
        # Select action using strategy
        action = strategy.select_action(state)

        # Take action in environment
        next_state, reward, done, info = env.step(action)

        # Update state
        state = next_state

    # Calculate metrics
    return calculate_metrics(env)


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


def evaluate_strategies(args):
    """
    Evaluate multiple trading strategies on test data.

    Args:
        args: Command-line arguments

    Returns:
        DataFrame with comparison results
    """
    # Set device
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and preprocess data
    logger.info("Loading and preprocessing data")
    data_loader = FinancialDataLoader(cache_dir=args.data_cache)

    # Process tickers
    if args.tickers:
        tickers = args.tickers.split(',')
    else:
        tickers = [args.ticker]

    # Define strategies to compare
    strategies = {
        'Buy and Hold': BuyAndHoldStrategy(),
        'Sell and Hold': SellAndHoldStrategy(),
        'MA Trend Following': MovingAverageTrendFollowing(window_size=args.window_size),
        'MA Mean Reversion': MovingAverageMeanReversion(window_size=args.window_size),
    }

    # Add MADDQN if model path provided
    timesnet_type = 'basic'  # Default value
    if args.model_path:
        # Load model configuration to determine TimesNet type
        model_config = load_model_config(args.model_path, device)
        timesnet_type = model_config.get('timesnet_type', 'basic')
        logger.info(f"Model uses {timesnet_type} TimesNet implementation")
        strategies['MADDQN'] = 'model'

    # Results storage
    all_results = {}

    # Initialize counters for risk and return agents (if model provided)
    risk_agent_best_count = 0
    return_agent_best_count = 0

    for ticker in tickers:
        logger.info(f"Evaluating strategies on ticker: {ticker}")

        # Load data
        data = data_loader.load_single_asset(ticker, args.start_date, args.end_date)
        processed_data = data_loader.preprocess_data(data)

        # Split into training and testing sets
        train_data, test_data = data_loader.split_train_test(processed_data, args.train_ratio)

        # Store results for this ticker
        ticker_results = {}

        # Run each strategy
        for strategy_name, strategy in strategies.items():
            logger.info(f"Running {strategy_name} strategy...")

            # Create fresh environment for each strategy
            env = TradingEnvironment(
                test_data,  # Use test data for evaluation
                window_size=args.window_size,
                initial_balance=args.initial_balance,
                transaction_fee=args.transaction_fee
            )

            # Special handling for MADDQN
            if strategy_name == 'MADDQN':
                # Initialize MADDQN
                state_dim = (args.window_size, env.normalized_data.shape[1])
                action_dim = 3  # sell, hold, buy

                # Load pretrained model
                logger.info(f"Loading model from {args.model_path}")

                # Load checkpoint
                checkpoint = torch.load(args.model_path, map_location=device)

                # Extract model configuration
                cfg = checkpoint.get('args', {})

                # Create MADDQN with correct TimesNet type
                maddqn = create_maddqn(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    hidden_dim=cfg.get('hidden_dim', 128),
                    learning_rate=cfg.get('learning_rate', 0.001),
                    gamma=cfg.get('gamma', 0.99),
                    epsilon_start=cfg.get('epsilon_start', 0.9),
                    epsilon_end=cfg.get('epsilon_end', 0.05),
                    epsilon_decay=cfg.get('epsilon_decay', 200),
                    target_update=cfg.get('target_update', 10),
                    buffer_size=cfg.get('buffer_size', 1000),
                    batch_size=cfg.get('batch_size', 64),
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

                with torch.no_grad():
                    while not done:
                        # Select action with MADDQN
                        action, risk_q_values, return_q_values = maddqn.select_action(state, epsilon=0.0)

                        # Get individual agent actions (for analysis)
                        risk_action = np.argmax(risk_q_values)
                        return_action = np.argmax(return_q_values)

                        # Record which sub-agent's action was chosen by final agent
                        if action == risk_action and action != return_action:
                            risk_agent_best_count += 1
                        elif action == return_action and action != risk_action:
                            return_agent_best_count += 1

                        # Record actions
                        actions_history.append({
                            'risk_action': risk_action,
                            'return_action': return_action,
                            'final_action': action
                        })

                        # Take action in environment
                        next_state, reward, done, info = env.step(action)

                        # Update state
                        state = next_state
            else:
                # Run other strategies
                run_strategy(env, strategy)

            # Calculate metrics
            metrics = calculate_metrics(env)
            ticker_results[strategy_name] = metrics

            # Display metrics
            logger.info(f"Results for {strategy_name} on {ticker}:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.2f}")

        # Save results for this ticker
        all_results[ticker] = ticker_results

    # Calculate average metrics across all tickers
    if len(tickers) > 1:
        avg_results = defaultdict(lambda: defaultdict(float))

        for ticker, ticker_results in all_results.items():
            for strategy_name, metrics in ticker_results.items():
                for metric, value in metrics.items():
                    avg_results[strategy_name][metric] += value

        # Divide by number of tickers
        for strategy_name in avg_results:
            for metric in avg_results[strategy_name]:
                avg_results[strategy_name][metric] /= len(tickers)

        # Add average results
        all_results['average'] = dict(avg_results)

    # Convert results to DataFrame for easier comparison
    results_df = pd.DataFrame(columns=['ticker', 'strategy', 'cumulative_return', 'annualized_return',
                                       'sharpe_ratio', 'max_drawdown', 'win_rate', 'total_trades'])

    row_idx = 0
    for ticker, ticker_results in all_results.items():
        for strategy_name, metrics in ticker_results.items():
            results_df.loc[row_idx] = [
                ticker,
                strategy_name,
                metrics['cumulative_return'],
                metrics['annualized_return'],
                metrics['sharpe_ratio'],
                metrics['max_drawdown'],
                metrics['win_rate'],
                metrics['total_trades']
            ]
            row_idx += 1

    # Save results to CSV
    results_csv_path = os.path.join(args.output_dir, 'strategy_comparison.csv')
    results_df.to_csv(results_csv_path, index=False)
    logger.info(f"Saved comparison results to {results_csv_path}")

    # Create comparison charts
    # 1. Cumulative returns by ticker and strategy
    plt.figure(figsize=(12, 8))
    sns.barplot(x='ticker', y='cumulative_return', hue='strategy', data=results_df)
    plt.title('Cumulative Returns by Strategy and Ticker')
    plt.xlabel('Ticker')
    plt.ylabel('Cumulative Return (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'cumulative_returns_comparison.png'))

    # 2. Sharpe ratio by ticker and strategy
    plt.figure(figsize=(12, 8))
    sns.barplot(x='ticker', y='sharpe_ratio', hue='strategy', data=results_df)
    plt.title('Sharpe Ratio by Strategy and Ticker')
    plt.xlabel('Ticker')
    plt.ylabel('Sharpe Ratio')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'sharpe_ratio_comparison.png'))

    # 3. Max drawdown by ticker and strategy
    plt.figure(figsize=(12, 8))
    sns.barplot(x='ticker', y='max_drawdown', hue='strategy', data=results_df)
    plt.title('Maximum Drawdown by Strategy and Ticker')
    plt.xlabel('Ticker')
    plt.ylabel('Maximum Drawdown (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'max_drawdown_comparison.png'))

    # If MADDQN was evaluated, show agent decision analysis
    if args.model_path:
        logger.info("MADDQN Agent Decision Analysis:")
        logger.info(f"Risk Agent chosen as best: {risk_agent_best_count} times")
        logger.info(f"Return Agent chosen as best: {return_agent_best_count} times")

        # Create pie chart of agent decisions
        plt.figure(figsize=(8, 8))
        plt.pie(
            [risk_agent_best_count, return_agent_best_count],
            labels=['Risk Agent', 'Return Agent'],
            autopct='%1.1f%%',
            startangle=90
        )
        plt.title(f'MADDQN ({timesnet_type.capitalize()}) Final Agent Decision Sources')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'maddqn_decision_sources.png'))

    return results_df


def main():
    parser = argparse.ArgumentParser(description='Compare trading strategies on financial data')

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
    parser.add_argument('--model-path', type=str, default=None, help='Path to the MADDQN model checkpoint (optional)')
    parser.add_argument('--output-dir', type=str, default='./comparison_results', help='Directory to save results')
    parser.add_argument('--use-cuda', action='store_true', help='Use CUDA if available')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Evaluate strategies
    results = evaluate_strategies(args)

    logger.info(f"Comparison complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()