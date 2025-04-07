"""
Script to create a comprehensive interactive dashboard comparing 
all trading strategies using Plotly.

This script can be run after evaluating multiple strategies to generate
a master dashboard with all results.
"""

import os
import argparse
import pandas as pd
import json
from plotly_utils.plotly_utils import create_dashboard, plot_multi_metric_comparison

def load_results(comparison_dir):
    """
    Load comparison results from CSV and action files.
    
    Args:
        comparison_dir: Directory containing comparison results
        
    Returns:
        results_df: DataFrame with comparison results
        all_data: Dictionary with detailed data for each strategy
    """
    # Load comparison results CSV
    results_csv = os.path.join(comparison_dir, 'strategy_comparison.csv')
    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"Comparison results CSV not found: {results_csv}")
    
    results_df = pd.read_csv(results_csv)
    
    # Load detailed data for each strategy
    all_data = {}
    
    # Get unique tickers and strategies
    tickers = results_df['ticker'].unique()
    strategies = results_df['strategy'].unique()
    
    for ticker in tickers:
        ticker_data = {}
        
        # Check for MADDQN actions file
        maddqn_file = os.path.join(comparison_dir, f"{ticker}_actions.csv")
        if os.path.exists(maddqn_file):
            maddqn_df = pd.read_csv(maddqn_file)
            ticker_data['MADDQN'] = {
                'prices': maddqn_df['price'].tolist(),
                'actions': maddqn_df['final_action'].tolist(),
                'portfolio_values': maddqn_df['total_value'].tolist(),
                'risk_actions': maddqn_df['risk_action'].tolist() if 'risk_action' in maddqn_df.columns else None,
                'return_actions': maddqn_df['return_action'].tolist() if 'return_action' in maddqn_df.columns else None
            }
        
        # Add data to all_data dictionary
        if ticker_data:
            all_data[ticker] = ticker_data
    
    return results_df, all_data

def create_strategy_dashboards(results_df, all_data, output_dir):
    """
    Create interactive dashboards for strategy comparison.
    
    Args:
        results_df: DataFrame with comparison results
        all_data: Dictionary with detailed data for each strategy
        output_dir: Directory to save results
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create multi-metric dashboard for all tickers and strategies
    dashboard_path = os.path.join(output_dir, 'full_strategy_dashboard.html')
    
    plot_multi_metric_comparison(
        results_df=results_df,
        save_path=dashboard_path,
        show=False
    )
    
    print(f"Full strategy dashboard saved to {dashboard_path}")
    
    # Create detailed dashboards for each ticker
    for ticker, ticker_data in all_data.items():
        # Filter results for this ticker
        ticker_results = results_df[results_df['ticker'] == ticker]
        
        # Prepare data for dashboard
        strategies = list(ticker_data.keys())
        results_dict = {}
        prices_dict = {}
        actions_dict = {}
        portfolio_values_dict = {}
        
        for strategy in strategies:
            # Get metrics from results DataFrame
            strategy_metrics = ticker_results[ticker_results['strategy'] == strategy]
            if not strategy_metrics.empty:
                metrics = {
                    'cumulative_return': strategy_metrics['cumulative_return'].values[0],
                    'annualized_return': strategy_metrics['annualized_return'].values[0],
                    'sharpe_ratio': strategy_metrics['sharpe_ratio'].values[0],
                    'max_drawdown': strategy_metrics['max_drawdown'].values[0],
                    'win_rate': strategy_metrics['win_rate'].values[0],
                    'total_trades': strategy_metrics['total_trades'].values[0]
                }
            else:
                metrics = {}
            
            # Get detailed data
            strategy_data = ticker_data.get(strategy, {})
            prices = strategy_data.get('prices', [])
            actions = strategy_data.get('actions', [])
            portfolio_values = strategy_data.get('portfolio_values', [])
            
            # Add to dictionaries
            results_dict[strategy] = metrics
            prices_dict[strategy] = prices
            actions_dict[strategy] = actions
            portfolio_values_dict[strategy] = portfolio_values
        
        # Create dashboard
        if results_dict:
            dashboard_path = os.path.join(output_dir, f'{ticker}_dashboard.html')
            
            create_dashboard(
                results_dict=results_dict,
                prices_dict=prices_dict,
                actions_dict=actions_dict,
                portfolio_values_dict=portfolio_values_dict,
                save_path=dashboard_path,
                show=False
            )
            
            print(f"Dashboard for {ticker} saved to {dashboard_path}")

def main():
    parser = argparse.ArgumentParser(description='Create comprehensive strategy dashboards')
    
    parser.add_argument('--comparison-dir', type=str, required=True, 
                        help='Directory containing comparison results')
    parser.add_argument('--output-dir', type=str, default='./dashboards', 
                        help='Directory to save dashboards')
    
    args = parser.parse_args()
    
    # Load results
    results_df, all_data = load_results(args.comparison_dir)
    
    # Create dashboards
    create_strategy_dashboards(results_df, all_data, args.output_dir)
    
    print("All dashboards created successfully!")

if __name__ == "__main__":
    main()
