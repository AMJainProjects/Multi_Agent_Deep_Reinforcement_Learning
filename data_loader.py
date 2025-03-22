import pandas as pd
import numpy as np
import yfinance as yf
import os
from datetime import datetime, timedelta


class FinancialDataLoader:
    """
    Data loader for financial time series data.
    
    This class handles the loading, preprocessing, and splitting of financial data,
    supporting both single-asset and multi-asset datasets as described in the paper.
    """
    
    def __init__(self, cache_dir='data_cache'):
        """
        Initialize the data loader.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def load_single_asset(self, ticker, start_date, end_date, cache=True):
        """
        Load data for a single financial asset.
        
        Args:
            ticker: Asset ticker symbol
            start_date: Start date for data
            end_date: End date for data
            cache: Whether to use cached data
            
        Returns:
            DataFrame with OHLCV data
        """
        cache_file = os.path.join(self.cache_dir, f"{ticker}_{start_date}_{end_date}.csv")
        
        # If cache is enabled and file exists, load from cache
        if cache and os.path.exists(cache_file):
            print(f"Loading {ticker} data from cache...")
            data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return data
        
        # Download data
        print(f"Downloading {ticker} data...")
        data = yf.download(ticker, start=start_date, end=end_date)
        
        # Ensure we have the expected columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Downloaded data missing required column: {col}")
        
        # Rename columns to lowercase
        data.columns = [col.lower() for col in data.columns]
        
        # Handle missing values
        data.fillna(method='ffill', inplace=True)  # Forward fill
        data.fillna(method='bfill', inplace=True)  # Backward fill
        
        # Cache data if enabled
        if cache:
            data.to_csv(cache_file)
        
        return data
    
    def load_mixed_dataset(self, tickers, start_date, end_date, cache=True):
        """
        Load a mixed dataset containing multiple assets.
        
        Args:
            tickers: List of asset ticker symbols
            start_date: Start date for data
            end_date: End date for data
            cache: Whether to use cached data
            
        Returns:
            Dictionary of DataFrames with OHLCV data for each asset
        """
        datasets = {}
        
        for ticker in tickers:
            data = self.load_single_asset(ticker, start_date, end_date, cache)
            datasets[ticker] = data
        
        return datasets
    
    def preprocess_data(self, data, add_technical_indicators=True):
        """
        Preprocess financial data.
        
        Args:
            data: DataFrame with OHLCV data
            add_technical_indicators: Whether to add technical indicators
            
        Returns:
            Preprocessed DataFrame
        """
        df = data.copy()
        
        # Ensure we have the required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Data missing required column: {col}")
        
        if add_technical_indicators:
            # Add simple technical indicators
            
            # 1. Moving Averages
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma10'] = df['close'].rolling(window=10).mean()
            df['ma20'] = df['close'].rolling(window=20).mean()
            
            # 2. Relative price changes
            df['daily_return'] = df['close'].pct_change()
            df['weekly_return'] = df['close'].pct_change(periods=5)
            
            # 3. Volatility (rolling standard deviation)
            df['volatility_5d'] = df['daily_return'].rolling(window=5).std()
            df['volatility_10d'] = df['daily_return'].rolling(window=10).std()
            
            # 4. Relative Strength Index (RSI)
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # 5. MACD (Moving Average Convergence Divergence)
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            
            # 6. Price Relative to Moving Averages
            df['price_to_ma5'] = df['close'] / df['ma5'] - 1
            df['price_to_ma10'] = df['close'] / df['ma10'] - 1
            df['price_to_ma20'] = df['close'] / df['ma20'] - 1
        
        # Handle NaN values from technical indicators
        df.fillna(method='bfill', inplace=True)
        
        return df
    
    def split_train_test(self, data, train_ratio=0.8):
        """
        Split data into training and testing sets.
        
        Args:
            data: DataFrame with financial data
            train_ratio: Ratio of data to use for training
            
        Returns:
            train_data, test_data
        """
        total_days = len(data)
        train_days = int(total_days * train_ratio)
        
        train_data = data.iloc[:train_days]
        test_data = data.iloc[train_days:]
        
        return train_data, test_data
    
    def create_window_samples(self, data, window_size):
        """
        Create windowed samples from time series data.
        
        Args:
            data: DataFrame with financial data
            window_size: Size of the window
            
        Returns:
            List of windowed samples
        """
        samples = []
        
        for i in range(len(data) - window_size):
            window = data.iloc[i:i+window_size]
            samples.append(window.values)
        
        return np.array(samples)
    
    def create_mixed_samples(self, datasets, window_size):
        """
        Create mixed samples from multiple assets.
        
        This is used for the generalization experiment in the paper,
        where they combine data from multiple US stock indices.
        
        Args:
            datasets: Dictionary of asset DataFrames
            window_size: Size of the window
            
        Returns:
            Array of mixed samples
        """
        all_samples = []
        
        for ticker, data in datasets.items():
            samples = self.create_window_samples(data, window_size)
            all_samples.append(samples)
        
        # Combine samples from all assets
        combined_samples = np.concatenate(all_samples, axis=0)
        
        # Shuffle samples
        np.random.shuffle(combined_samples)
        
        return combined_samples

# Example usage:
if __name__ == "__main__":
    data_loader = FinancialDataLoader()
    
    # Load single asset
    spy_data = data_loader.load_single_asset("SPY", "2019-01-01", "2021-12-31")
    print("SPY data shape:", spy_data.shape)
    
    # Preprocess data
    processed_spy = data_loader.preprocess_data(spy_data)
    print("Processed SPY data shape:", processed_spy.shape)
    print("Columns:", processed_spy.columns.tolist())
    
    # Split into train/test
    train_spy, test_spy = data_loader.split_train_test(processed_spy)
    print("Train shape:", train_spy.shape)
    print("Test shape:", test_spy.shape)
    
    # Create windowed samples
    window_size = 10
    train_samples = data_loader.create_window_samples(train_spy, window_size)
    print("Train samples shape:", train_samples.shape)
    
    # Load mixed dataset
    tickers = ["DIA", "SPY", "QQQ"]  # Dow Jones, S&P 500, Nasdaq
    mixed_data = data_loader.load_mixed_dataset(tickers, "2019-01-01", "2021-12-31")
    print("Mixed dataset loaded for tickers:", list(mixed_data.keys()))
