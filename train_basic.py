import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import datetime
import json

from data_loader import FinancialDataLoader
from environment import TradingEnvironment
from maddqn_basic import MADDQN

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

def train_maddqn(args):
    """
    Train the MADDQN framework on financial data.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Trained MADDQN model
    """
    # Set random seed for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create experiment directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(args.output_dir, f"maddqn_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data")
    data_loader = FinancialDataLoader(cache_dir=args.data_cache)
    
    if args.mixed_dataset:
        # Load mixed dataset for generalization experiments
        tickers = args.tickers.split(',')
        datasets = data_loader.load_mixed_dataset(tickers, args.start_date, args.end_date)
        
        # Preprocess each dataset
        processed_datasets = {}
        for ticker, data in datasets.items():
            processed_datasets[ticker] = data_loader.preprocess_data(data)
        
        # Split into train/test sets
        train_datasets = {}
        test_datasets = {}
        for ticker, data in processed_datasets.items():
            train_data, test_data = data_loader.split_train_test(data, args.train_ratio)
            train_datasets[ticker] = train_data
            test_datasets[ticker] = test_data
        
        # Create environment using the first ticker's training data
        # (we'll use the model for other assets later in testing)
        primary_ticker = tickers[0]
        env = TradingEnvironment(
            train_datasets[primary_ticker], 
            window_size=args.window_size,
            initial_balance=args.initial_balance,
            transaction_fee=args.transaction_fee
        )
        
        # For mixed dataset training, create random samples from all training datasets
        all_train_samples = []
        for ticker, data in train_datasets.items():
            samples = data_loader.create_window_samples(data, args.window_size)
            all_train_samples.append(samples)
        
        combined_train_samples = np.concatenate(all_train_samples, axis=0)
        np.random.shuffle(combined_train_samples)
        
        # Save the processed datasets for later use
        for ticker, data in processed_datasets.items():
            data.to_csv(os.path.join(experiment_dir, f"{ticker}_processed.csv"))
    
    else:
        # Load single asset dataset
        data = data_loader.load_single_asset(args.ticker, args.start_date, args.end_date)
        processed_data = data_loader.preprocess_data(data)
        
        # Split into training and testing sets
        train_data, test_data = data_loader.split_train_test(processed_data, args.train_ratio)
        
        # Create trading environment
        env = TradingEnvironment(
            train_data, 
            window_size=args.window_size,
            initial_balance=args.initial_balance,
            transaction_fee=args.transaction_fee
        )
        
        # Save the processed data for later use
        processed_data.to_csv(os.path.join(experiment_dir, f"{args.ticker}_processed.csv"))
    
    # Initialize MADDQN framework
    state_dim = (args.window_size, env.normalized_data.shape[1])
    action_dim = 3  # sell, hold, buy
    
    logger.info(f"Initializing MADDQN with state_dim={state_dim}, action_dim={action_dim}")
    maddqn = MADDQN(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        target_update=args.target_update,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        device=device
    )
    
    # Training variables
    episode_rewards = []
    risk_loss_history = []
    return_loss_history = []
    final_loss_history = []
    total_values = []
    
    # Training loop
    logger.info(f"Starting training for {args.num_episodes} episodes")
    
    for episode in tqdm(range(args.num_episodes)):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action using the MADDQN framework
            action, risk_q_values, return_q_values = maddqn.select_action(state)
            
            # Take action in the environment
            next_state, reward, done, info = env.step(action)
            
            # Calculate specific rewards for each agent
            risk_reward = env.calculate_risk_reward()
            return_reward = env.calculate_return_reward()
            final_reward = reward  # Profit-based reward
            
            # Store transitions in the replay buffers
            maddqn.store_transition(
                state, risk_q_values, return_q_values, action,
                risk_reward, return_reward, final_reward,
                next_state, done
            )
            
            # Update network parameters
            if episode >= args.learning_starts:
                losses = maddqn.update()
                
                if losses['risk_loss'] is not None:
                    risk_loss_history.append(losses['risk_loss'])
                
                if losses['return_loss'] is not None:
                    return_loss_history.append(losses['return_loss'])
                
                if losses['final_loss'] is not None:
                    final_loss_history.append(losses['final_loss'])
            
            # Update state and accumulate reward
            state = next_state
            episode_reward += reward
        
        # Record episode results
        episode_rewards.append(episode_reward)
        total_values.append(env.total_value)
        
        # Log progress
        if (episode + 1) % args.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-args.log_interval:])
            avg_total_value = np.mean(total_values[-args.log_interval:])
            
            logger.info(f"Episode {episode+1}/{args.num_episodes} - "
                       f"Avg Reward: {avg_reward:.2f}, Avg Total Value: {avg_total_value:.2f}")
        
        # Save model checkpoint
        if (episode + 1) % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(experiment_dir, f"model_ep{episode+1}.pt")
            torch.save({
                'risk_agent_state_dict': maddqn.risk_agent.policy_net.state_dict(),
                'return_agent_state_dict': maddqn.return_agent.policy_net.state_dict(),
                'final_agent_state_dict': maddqn.final_agent.policy_net.state_dict(),
                'episode': episode,
                'episode_rewards': episode_rewards,
                'total_values': total_values,
                'args': vars(args)
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Plot training results
    plt.figure(figsize=(15, 10))
    
    # Episode rewards
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Portfolio value
    plt.subplot(2, 2, 2)
    plt.plot(total_values)
    plt.title('Portfolio Value')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    
    # Loss histories
    plt.subplot(2, 2, 3)
    if risk_loss_history:
        plt.plot(risk_loss_history, label='Risk Agent')
    if return_loss_history:
        plt.plot(return_loss_history, label='Return Agent')
    if final_loss_history:
        plt.plot(final_loss_history, label='Final Agent')
    plt.title('Loss History')
    plt.xlabel('Update Step')
    plt.ylabel('Loss')
    plt.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'training_results.png'))
    
    # Save final model
    final_model_path = os.path.join(experiment_dir, 'model_final.pt')
    torch.save({
        'risk_agent_state_dict': maddqn.risk_agent.policy_net.state_dict(),
        'return_agent_state_dict': maddqn.return_agent.policy_net.state_dict(),
        'final_agent_state_dict': maddqn.final_agent.policy_net.state_dict(),
        'episode': args.num_episodes,
        'episode_rewards': episode_rewards,
        'total_values': total_values,
        'args': vars(args)
    }, final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    return maddqn, experiment_dir

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
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train MADDQN
    maddqn, experiment_dir = train_maddqn(args)
    
    logger.info(f"Training complete. Results saved to {experiment_dir}")

if __name__ == "__main__":
    main()
