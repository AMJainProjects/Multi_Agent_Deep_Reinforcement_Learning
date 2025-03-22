import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# Import the components
from environment import TradingEnvironment
from maddqn_basic import MADDQN
from data_loader import FinancialDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_synthetic_data(num_days=1000, volatility=0.01, trend=0.0005, seed=42):
    """
    Create synthetic stock data for testing.
    
    Args:
        num_days: Number of trading days
        volatility: Daily volatility
        trend: Daily trend (positive for uptrend, negative for downtrend)
        seed: Random seed
        
    Returns:
        DataFrame with synthetic OHLCV data
    """
    np.random.seed(seed)
    
    # Generate price data
    price = 100  # Starting price
    prices = [price]
    
    for _ in range(num_days - 1):
        # Random component
        change = np.random.normal(trend, volatility)
        price *= (1 + change)
        prices.append(price)
    
    prices = np.array(prices)
    
    # Create OHLCV data
    dates = pd.date_range(start='2020-01-01', periods=num_days, freq='B')
    data = pd.DataFrame(index=dates)
    
    # Generate OHLC from close prices
    data['close'] = prices
    data['open'] = prices * (1 + np.random.normal(0, volatility/3, num_days))
    data['high'] = np.maximum(data['open'], data['close']) * (1 + np.abs(np.random.normal(0, volatility/2, num_days)))
    data['low'] = np.minimum(data['open'], data['close']) * (1 - np.abs(np.random.normal(0, volatility/2, num_days)))
    
    # Generate trading volume
    data['volume'] = np.random.gamma(shape=2.0, scale=1000000, size=num_days)
    data['volume'] *= 1 + 0.5 * np.abs(np.log(data['close'] / data['close'].shift(1)).fillna(0))
    
    return data

def demo():
    """Run a quick demo of MADDQN on synthetic data."""
    # Parameters
    window_size = 10
    initial_balance = 10000
    transaction_fee = 0.001
    hidden_dim = 64
    learning_rate = 0.001
    gamma = 0.99
    epsilon_start = 0.9
    epsilon_end = 0.1
    epsilon_decay = 200
    target_update = 10
    buffer_size = 1000
    batch_size = 32
    num_episodes = 50
    
    # Create synthetic data
    logger.info("Creating synthetic data...")
    data = create_synthetic_data(num_days=500, volatility=0.02, trend=0.0002)
    
    # Add basic technical indicators
    data['ma5'] = data['close'].rolling(window=5).mean()
    data['ma10'] = data['close'].rolling(window=10).mean()
    data['daily_return'] = data['close'].pct_change()
    
    # Fill NaN values
    data = data.fillna(method='bfill')
    
    # Split into train and test sets
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
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
    
    logger.info(f"Initializing MADDQN with state_dim={state_dim}, action_dim={action_dim}")
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
                maddqn.update()
            
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
    
    # Plot training results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(2, 1, 2)
    plt.plot(total_values)
    plt.title('Portfolio Value')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    
    plt.tight_layout()
    plt.savefig('demo_training.png')
    
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
    
    # Plot test results
    plt.figure(figsize=(12, 10))
    
    # Plot price with actions
    plt.subplot(2, 1, 1)
    plt.plot(prices, label='Price')
    plt.title('Price and Actions')
    
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
    plt.plot(test_values, label='Portfolio Value')
    plt.title('Portfolio Value')
    plt.xlabel('Step')
    plt.ylabel('Value')
    
    plt.tight_layout()
    plt.savefig('demo_testing.png')
    
    # Analyze agent agreement
    agreement_count = sum(1 for ra, rta, fa in zip(risk_actions, return_actions, final_actions) 
                         if ra == rta and ra == fa)
    agreement_rate = agreement_count / len(final_actions) * 100
    
    risk_final_agreement = sum(1 for ra, fa in zip(risk_actions, final_actions) if ra == fa)
    risk_final_rate = risk_final_agreement / len(final_actions) * 100
    
    return_final_agreement = sum(1 for rta, fa in zip(return_actions, final_actions) if rta == fa)
    return_final_rate = return_final_agreement / len(final_actions) * 100
    
    logger.info(f"Agent Agreement Analysis:")
    logger.info(f"  All agents agree: {agreement_rate:.2f}%")
    logger.info(f"  Risk-Final agree: {risk_final_rate:.2f}%")
    logger.info(f"  Return-Final agree: {return_final_rate:.2f}%")
    
    # Create pie chart of agent decisions
    plt.figure(figsize=(8, 8))
    
    # Count decisions where final agent agreed with only one sub-agent
    risk_only = sum(1 for ra, rta, fa in zip(risk_actions, return_actions, final_actions) 
                   if fa == ra and fa != rta)
    return_only = sum(1 for ra, rta, fa in zip(risk_actions, return_actions, final_actions) 
                     if fa == rta and fa != ra)
    both_agree = sum(1 for ra, rta, fa in zip(risk_actions, return_actions, final_actions) 
                    if fa == ra and fa == rta)
    none_agree = sum(1 for ra, rta, fa in zip(risk_actions, return_actions, final_actions) 
                    if fa != ra and fa != rta)
    
    plt.pie(
        [risk_only, return_only, both_agree, none_agree], 
        labels=['Risk Agent', 'Return Agent', 'Both Agents', 'Independent Decision'],
        autopct='%1.1f%%',
        startangle=90
    )
    plt.title('MADDQN Final Agent Decision Sources')
    plt.tight_layout()
    plt.savefig('demo_decision_sources.png')
    
    logger.info("Demo completed successfully!")

if __name__ == "__main__":
    demo()
