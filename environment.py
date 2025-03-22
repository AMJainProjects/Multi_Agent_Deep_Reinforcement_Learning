import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class TradingEnvironment:
    """
    A financial trading environment that simulates market interactions.
    
    This environment allows agents to:
    - Observe the market state (prices, volumes, technical indicators)
    - Take actions (buy, hold, sell)
    - Receive rewards based on their trading performance
    """
    
    def __init__(self, data, window_size=10, initial_balance=10000, transaction_fee=0.001):
        """
        Initialize the trading environment.
        
        Args:
            data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            window_size: Size of the observation window
            initial_balance: Initial balance for trading
            transaction_fee: Fee for each transaction as a percentage
        """
        self.data = data
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        
        # Normalize price data
        self.scaler = StandardScaler()
        self.normalized_data = self.normalize_data()
        
        # Reset environment to initial state
        self.reset()
        
    def normalize_data(self):
        """Normalize the price data using StandardScaler."""
        price_data = self.data[['open', 'high', 'low', 'close', 'volume']].values
        normalized_data = self.scaler.fit_transform(price_data)
        return normalized_data
        
    def reset(self):
        """Reset the environment to initial state."""
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.asset_value = 0
        self.total_value = self.balance
        self.position = 0  # -1 (short), 0 (neutral), 1 (long)
        self.trade_history = []
        
        return self._get_observation()
    
    def _get_observation(self):
        """Get the current observation (state)."""
        # Get window of normalized data
        observation = self.normalized_data[self.current_step - self.window_size:self.current_step]
        return observation
    
    def _get_price(self):
        """Get the current price."""
        return self.data.iloc[self.current_step]['close']
    
    def step(self, action):
        """
        Take an action in the environment.
        
        Args:
            action: 0 (sell), 1 (hold), 2 (buy)
            
        Returns:
            next_observation, reward, done, info
        """
        # Convert action from 0,1,2 to -1,0,1 for easier processing
        action_mapping = action - 1  # Convert to -1 (sell), 0 (hold), 1 (buy)
        
        # Get current price
        current_price = self._get_price()
        
        # Update position based on action
        self._update_position(action_mapping, current_price)
        
        # Calculate profit reward (simplest reward function)
        reward = self._calculate_profit_reward(current_price)
        
        # Move to the next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Get the next observation
        next_observation = self._get_observation() if not done else None
        
        # Calculate info
        info = {
            'balance': self.balance,
            'shares_held': self.shares_held,
            'asset_value': self.asset_value,
            'total_value': self.total_value,
            'position': self.position,
            'price': current_price
        }
        
        return next_observation, reward, done, info
    
    def _update_position(self, action, current_price):
        """Update position based on the action."""
        # Previous position
        prev_position = self.position
        prev_total_value = self.total_value
        
        # Update position based on action
        if action == -1:  # Sell
            if prev_position == 1:  # If currently long, sell all shares
                self.balance += self.shares_held * current_price * (1 - self.transaction_fee)
                self.shares_held = 0
                self.position = 0
            elif prev_position == 0:  # If neutral, go short
                shares_to_short = self.balance / current_price
                self.balance -= shares_to_short * current_price * self.transaction_fee
                self.shares_held = -shares_to_short
                self.position = -1
                
        elif action == 1:  # Buy
            if prev_position == -1:  # If currently short, buy to cover
                self.balance += self.shares_held * current_price * (1 - self.transaction_fee)
                self.shares_held = 0
                self.position = 0
            elif prev_position == 0:  # If neutral, go long
                shares_to_buy = self.balance / current_price
                self.balance -= shares_to_buy * current_price * (1 + self.transaction_fee)
                self.shares_held = shares_to_buy
                self.position = 1
        
        # Update asset value and total value
        self.asset_value = self.shares_held * current_price
        self.total_value = self.balance + self.asset_value
        
        # Record the trade
        self.trade_history.append({
            'step': self.current_step,
            'price': current_price,
            'action': action,
            'position': self.position,
            'total_value': self.total_value,
            'return': ((self.total_value - prev_total_value) / prev_total_value) * 100 if prev_total_value > 0 else 0
        })
    
    def _calculate_profit_reward(self, current_price):
        """Calculate simple profit-based reward."""
        # This is the Final Agent reward function from the paper
        if self.position == 0:
            return 0
        
        # Calculate short-term return
        next_step = min(self.current_step + 1, len(self.data) - 1)
        next_price = self.data.iloc[next_step]['close']
        
        short_term_return = (next_price - current_price) / current_price * 100
        reward = self.position * short_term_return
        
        return reward

    def calculate_risk_reward(self):
        """Calculates reward based on Sharpe ratio (Risk Agent reward)."""
        if self.position == 0:
            return 0
        
        # Get historical prices for return calculation
        historical_returns = []
        for i in range(min(10, len(self.trade_history))):
            if len(self.trade_history) > i:
                historical_returns.append(self.trade_history[-(i+1)]['return'])
        
        # Calculate Sharpe ratio
        if len(historical_returns) > 1:
            mean_return = np.mean(historical_returns)
            std_return = np.std(historical_returns)
            
            if std_return == 0:
                sharpe_ratio = 0
            else:
                sharpe_ratio = mean_return / std_return
        else:
            sharpe_ratio = 0
        
        # Reward is position * Sharpe ratio
        reward = self.position * sharpe_ratio
        
        return reward

    def calculate_return_reward(self, lookahead=5):
        """Calculates reward based on mid-term return (Return Agent reward)."""
        if self.position == 0:
            return 0
        
        current_price = self._get_price()
        
        # Get future price (if available)
        future_step = min(self.current_step + lookahead, len(self.data) - 1)
        future_price = self.data.iloc[future_step]['close']
        
        # Calculate mid-term return
        mid_term_return = (future_price - current_price) / current_price * 100
        
        # Reward is position * mid-term return
        reward = self.position * mid_term_return
        
        return reward
    
    def render(self):
        """Display the current state of the environment."""
        print(f"Step: {self.current_step}")
        print(f"Price: ${self._get_price():.2f}")
        print(f"Position: {['Short', 'Neutral', 'Long'][self.position+1]}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Shares held: {self.shares_held:.2f}")
        print(f"Asset value: ${self.asset_value:.2f}")
        print(f"Total value: ${self.total_value:.2f}")
        print("--------------------")
