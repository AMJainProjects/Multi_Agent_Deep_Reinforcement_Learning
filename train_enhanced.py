import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

from maddqn_enhanced import MADDQN
from environment import TradingEnvironment
from data_loader import FinancialDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_enhanced_maddqn(ticker="SPY", start_date="2010-01-01", end_date="2021-12-31"):
    """
    Train the enhanced MADDQN framework with TimesNet on financial data.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Training data start date
        end_date: Training data end date
        
    Returns:
        Trained MADDQN model and performance metrics
    """
    # Load and preprocess data
    logger.info(f"Loading and preprocessing data for {ticker}")
    data_loader = FinancialDataLoader(cache_dir="data_cache")
    
    # Load data
    data = data_loader.load_single_asset(ticker, start_date, end_date)
    processed_data = data_loader.preprocess_data(data)
    
    # Split into training and testing sets
    train_data, test_data = data_loader.split_train_test(processed_data, train_ratio=0.8)
    
    # Parameters
    window_size = 10
    initial_balance = 10000
    transaction_fee = 0.001
    hidden_dim = 128
    learning_rate = 0.001
    gamma = 0.99
    epsilon_start = 0.9
    epsilon_end = 0.05
    epsilon_decay = 200
    target_update = 10
    buffer_size = 1000
    batch_size = 64
    num_episodes = 100
    
    # Create environment
    env = TradingEnvironment(
        train_data,
        window_size=window_size,
        initial_balance=initial_balance,
        transaction_fee=transaction_fee
    )
    
    # Initialize MADDQN
    state_dim = (window_size, env.normalized_data.shape[1])
    action_dim = 3  # sell, hold, buy
    
    logger.info(f"Initializing Enhanced MADDQN with state_dim={state_dim}, action_dim={action_dim}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    maddqn = MADDQN(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        target_update=target_update,
        buffer_size=buffer_size,
        batch_size=batch_size,
        device=device
    )
    
    # Training variables
    episode_rewards = []
    total_values = []
    risk_loss_history = []
    return_loss_history = []
    final_loss_history = []
    
    # Training loop
    logger.info(f"Starting training for {num_episodes} episodes")
    
    for episode in tqdm(range(num_episodes)):
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
            if episode >= 5:  # Start learning after a few episodes
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
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_total_value = np.mean(total_values[-10:])
            
            logger.info(f"Episode {episode+1}/{num_episodes} - "
                       f"Avg Reward: {avg_reward:.2f}, Avg Total Value: {avg_total_value:.2f}")
    
    # Test the trained model
    logger.info("Testing the trained model on test data...")
    
    # Create test environment
    test_env = TradingEnvironment(
        test_data,
        window_size=window_size,
        initial_balance=initial_balance,
        transaction_fee=transaction_fee
    )
    
    # Run test episode
    state = test_env.reset()
    done = False
    
    test_rewards = []
    test_values = []
    risk_actions = []
    return_actions = []
    final_actions = []
    prices = []
    
    while not done:
        # Select action
        action, risk_q_values, return_q_values = maddqn.select_action(state, epsilon=0.0)  # No exploration
        
        # Get individual agent actions (for analysis)
        risk_action = np.argmax(risk_q_values)
        return_action = np.argmax(return_q_values)
        
        # Take action in environment
        next_state, reward, done, info = test_env.step(action)
        
        # Record data
        test_rewards.append(reward)
        test_values.append(info['total_value'])
        risk_actions.append(risk_action)
        return_actions.append(return_action)
        final_actions.append(action)
        prices.append(info['price'])
        
        # Update state
        state = next_state
    
    # Calculate test metrics
    cumulative_return = (test_env.total_value - initial_balance) / initial_balance * 100
    logger.info(f"Test Cumulative Return: {cumulative_return:.2f}%")
    
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
    plt.title('Portfolio Value During Training')
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
    
    # Test results
    plt.subplot(2, 2, 4)
    plt.plot(prices, label='Price', color='blue', alpha=0.5)
    plt.plot(test_values, label='Portfolio Value', color='green')
    plt.title('Test Performance')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('enhanced_maddqn_results.png')
    
    return maddqn, {
        'cumulative_return': cumulative_return,
        'episode_rewards': episode_rewards,
        'total_values': total_values,
        'test_values': test_values,
        'prices': prices,
        'risk_actions': risk_actions,
        'return_actions': return_actions,
        'final_actions': final_actions
    }

if __name__ == "__main__":
    # Train model on S&P 500 ETF
    model, metrics = train_enhanced_maddqn(ticker="SPY")
    
    # Save results
    results = {
        'cumulative_return': metrics['cumulative_return'],
        'final_portfolio_value': metrics['test_values'][-1] if metrics['test_values'] else None,
        'training_episodes': len(metrics['episode_rewards']),
        'final_training_value': metrics['total_values'][-1] if metrics['total_values'] else None
    }
    
    # Display summary
    print("\nTraining and Testing Results Summary:")
    print(f"  Test Cumulative Return: {results['cumulative_return']:.2f}%")
    print(f"  Final Test Portfolio Value: ${results['final_portfolio_value']:.2f}")
    print(f"  Training Episodes: {results['training_episodes']}")
    print(f"  Final Training Portfolio Value: ${results['final_training_value']:.2f}")
    
    # Analyze agent agreement
    agreement_count = sum(1 for ra, rta, fa in zip(metrics['risk_actions'], metrics['return_actions'], metrics['final_actions']) 
                          if ra == rta and ra == fa)
    agreement_rate = agreement_count / len(metrics['final_actions']) * 100
    
    risk_final_agreement = sum(1 for ra, fa in zip(metrics['risk_actions'], metrics['final_actions']) if ra == fa)
    risk_final_rate = risk_final_agreement / len(metrics['final_actions']) * 100
    
    return_final_agreement = sum(1 for rta, fa in zip(metrics['return_actions'], metrics['final_actions']) if rta == fa)
    return_final_rate = return_final_agreement / len(metrics['final_actions']) * 100
    
    print("\nAgent Agreement Analysis:")
    print(f"  All agents agree: {agreement_rate:.2f}%")
    print(f"  Risk-Final agree: {risk_final_rate:.2f}%")
    print(f"  Return-Final agree: {return_final_rate:.2f}%")
