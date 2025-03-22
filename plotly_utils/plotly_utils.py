"""
Plotting utilities for MADDQN trading strategies using Plotly.

This module provides interactive visualization functions for:
1. Training results
2. Backtesting results
3. Strategy comparisons
4. Agent decision analysis
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_training_results(episode_rewards, total_values, risk_loss_history=None, 
                         return_loss_history=None, final_loss_history=None, 
                         save_path=None, show=True, title_prefix=""):
    """
    Plot training results with interactive Plotly charts.
    
    Args:
        episode_rewards: List of episode rewards
        total_values: List of portfolio values
        risk_loss_history: List of risk agent loss values (optional)
        return_loss_history: List of return agent loss values (optional)
        final_loss_history: List of final agent loss values (optional)
        save_path: Path to save the HTML file (optional)
        show: Whether to show the plot (default: True)
        title_prefix: Prefix for chart titles (e.g., "Basic " or "Enhanced ")
        
    Returns:
        Plotly figure object
    """
    # Determine number of rows based on available data
    num_rows = 2
    if risk_loss_history is not None or return_loss_history is not None or final_loss_history is not None:
        num_rows = 3
    
    # Create figure with subplots
    fig = make_subplots(
        rows=num_rows, 
        cols=1,
        subplot_titles=(
            f'{title_prefix}Episode Rewards', 
            f'{title_prefix}Portfolio Value',
            f'{title_prefix}Loss History' if num_rows > 2 else None
        ),
        vertical_spacing=0.1
    )
    
    # Episode rewards
    fig.add_trace(
        go.Scatter(
            x=list(range(len(episode_rewards))),
            y=episode_rewards,
            mode='lines',
            name='Episode Rewards',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Portfolio value
    fig.add_trace(
        go.Scatter(
            x=list(range(len(total_values))),
            y=total_values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='green', width=2)
        ),
        row=2, col=1
    )
    
    # Loss history
    if num_rows > 2:
        if risk_loss_history:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(risk_loss_history))),
                    y=risk_loss_history,
                    mode='lines',
                    name='Risk Agent Loss',
                    line=dict(color='red', width=1.5)
                ),
                row=3, col=1
            )
        
        if return_loss_history:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(return_loss_history))),
                    y=return_loss_history,
                    mode='lines',
                    name='Return Agent Loss',
                    line=dict(color='orange', width=1.5)
                ),
                row=3, col=1
            )
        
        if final_loss_history:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(final_loss_history))),
                    y=final_loss_history,
                    mode='lines',
                    name='Final Agent Loss',
                    line=dict(color='purple', width=1.5)
                ),
                row=3, col=1
            )
    
    # Update layout
    fig.update_layout(
        title=f'{title_prefix}MADDQN Training Results',
        height=300 * num_rows,
        width=1000,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(title_text="Episode", row=num_rows, col=1)
    
    if risk_loss_history is not None or return_loss_history is not None or final_loss_history is not None:
        fig.update_xaxes(title_text="Update Step", row=3, col=1)
    
    # Save if path provided
    if save_path:
        fig.write_html(save_path)
    
    # Show if requested
    if show:
        fig.show()
    
    return fig


def plot_backtest_results(prices, actions=None, total_values=None, 
                        risk_actions=None, return_actions=None, final_actions=None, 
                        metrics=None, save_path=None, show=True, ticker=""):
    """
    Plot backtest results with interactive Plotly charts.
    
    Args:
        prices: List of prices
        actions: List of action values (0, 1, 2) for sell, hold, buy
        total_values: List of portfolio values
        risk_actions: List of risk agent actions
        return_actions: List of return agent actions
        final_actions: List of final agent actions
        metrics: Dictionary of performance metrics
        save_path: Path to save the HTML file (optional)
        show: Whether to show the plot (default: True)
        ticker: Ticker symbol for the plotted asset
        
    Returns:
        Plotly figure object
    """
    # Determine if we need to show agent disagreement
    show_agent_comparison = (risk_actions is not None and return_actions is not None and final_actions is not None)
    
    # Create figure with subplots
    num_rows = 2 if total_values is not None else 1
    subplot_titles = [f"{ticker} Price and Actions"]
    
    if total_values is not None:
        subplot_titles.append(f"{ticker} Portfolio Value")
    
    fig = make_subplots(
        rows=num_rows, 
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.15
    )
    
    # Price chart
    fig.add_trace(
        go.Scatter(
            x=list(range(len(prices))),
            y=prices,
            mode='lines',
            name='Price',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Add buy/sell markers if actions provided
    if actions is not None or final_actions is not None:
        # Use final_actions if available, otherwise use actions
        act_data = final_actions if final_actions is not None else actions
        
        # Buy signals
        buy_indices = [i for i, a in enumerate(act_data) if a == 2]
        if buy_indices:
            buy_prices = [prices[i] for i in buy_indices]
            fig.add_trace(
                go.Scatter(
                    x=buy_indices,
                    y=buy_prices,
                    mode='markers',
                    name='Buy',
                    marker=dict(color='green', size=10, symbol='triangle-up')
                ),
                row=1, col=1
            )
        
        # Sell signals
        sell_indices = [i for i, a in enumerate(act_data) if a == 0]
        if sell_indices:
            sell_prices = [prices[i] for i in sell_indices]
            fig.add_trace(
                go.Scatter(
                    x=sell_indices,
                    y=sell_prices,
                    mode='markers',
                    name='Sell',
                    marker=dict(color='red', size=10, symbol='triangle-down')
                ),
                row=1, col=1
            )
    
    # Add portfolio value chart if provided
    if total_values is not None:
        fig.add_trace(
            go.Scatter(
                x=list(range(len(total_values))),
                y=total_values,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
    
    # Add metrics as annotations if provided
    if metrics is not None:
        metrics_text = f"<b>Performance Metrics:</b><br>"
        metrics_text += f"Cumulative Return: {metrics.get('cumulative_return', 0):.2f}%<br>"
        metrics_text += f"Annualized Return: {metrics.get('annualized_return', 0):.2f}%<br>"
        metrics_text += f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}<br>"
        metrics_text += f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%<br>"
        metrics_text += f"Win Rate: {metrics.get('win_rate', 0):.2f}%<br>"
        metrics_text += f"Total Trades: {metrics.get('total_trades', 0)}"
        
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.01, y=0,
            text=metrics_text,
            showarrow=False,
            font=dict(size=12),
            align="left",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1,
            borderpad=4
        )
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} Backtest Results",
        height=400 * num_rows,
        width=1000,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # X-axis labels
    fig.update_xaxes(title_text="Step", row=num_rows, col=1)
    
    # Save if path provided
    if save_path:
        fig.write_html(save_path)
    
    # Show if requested
    if show:
        fig.show()
    
    # If we have agent disagreement data, create an additional plot
    if show_agent_comparison:
        agent_fig = plot_agent_decisions(
            risk_actions, 
            return_actions, 
            final_actions, 
            prices=prices,
            save_path=save_path.replace('.html', '_agent_analysis.html') if save_path else None,
            show=show, 
            ticker=ticker
        )
    
    return fig


def plot_agent_decisions(risk_actions, return_actions, final_actions, prices=None,
                       save_path=None, show=True, ticker=""):
    """
    Plot analysis of different agent decisions and their agreement/disagreement.
    
    Args:
        risk_actions: List of risk agent actions (0, 1, 2)
        return_actions: List of return agent actions (0, 1, 2)
        final_actions: List of final agent actions (0, 1, 2)
        prices: List of prices (optional)
        save_path: Path to save the HTML file (optional)
        show: Whether to show the plot (default: True)
        ticker: Ticker symbol for the plotted asset
        
    Returns:
        Plotly figure object
    """
    # Calculate agreement statistics
    risk_only = sum(1 for ra, rta, fa in zip(risk_actions, return_actions, final_actions) 
                    if fa == ra and fa != rta)
    return_only = sum(1 for ra, rta, fa in zip(risk_actions, return_actions, final_actions) 
                      if fa == rta and fa != ra)
    both_agree = sum(1 for ra, rta, fa in zip(risk_actions, return_actions, final_actions) 
                     if fa == ra and fa == rta)
    none_agree = sum(1 for ra, rta, fa in zip(risk_actions, return_actions, final_actions) 
                     if fa != ra and fa != rta)
    
    # Calculate disagreement points for timeline
    disagreement_indices = [i for i, (ra, rta) in enumerate(zip(risk_actions, return_actions)) if ra != rta]
    
    # Create figure with subplots
    num_rows = 2 if prices is not None else 1
    fig = make_subplots(
        rows=num_rows, 
        cols=2,
        specs=[[{"type": "xy"}, {"type": "domain"}]] + [[{"colspan": 2}, None]] * (num_rows-1),
        subplot_titles=(
            "Agent Action Timeline", 
            "Decision Source Distribution",
            "Price with Agent Disagreement" if prices is not None else None
        ),
        vertical_spacing=0.15,
        column_widths=[0.7, 0.3]
    )
    
    # Agent action timeline
    agent_names = ['Risk Agent', 'Return Agent', 'Final Agent']
    agent_actions = [risk_actions, return_actions, final_actions]
    colors = ['red', 'blue', 'purple']
    
    for i, (name, actions, color) in enumerate(zip(agent_names, agent_actions, colors)):
        # Add a small offset to each agent line to avoid overlap
        offset = (i - 1) * 0.1
        
        # Convert actions from 0,1,2 to -1,0,1 for better visualization
        plot_actions = [a - 1 + offset for a in actions]
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(actions))),
                y=plot_actions,
                mode='lines',
                name=name,
                line=dict(color=color, width=1.5)
            ),
            row=1, col=1
        )
    
    # Decision source pie chart
    fig.add_trace(
        go.Pie(
            labels=['Risk Agent Only', 'Return Agent Only', 'Both Agents Agree', 'Independent Decision'],
            values=[risk_only, return_only, both_agree, none_agree],
            hole=0.3,
            marker_colors=['red', 'blue', 'purple', 'gray']
        ),
        row=1, col=2
    )
    
    # Add price chart with disagreement points if prices provided
    if prices is not None and len(disagreement_indices) > 0:
        fig.add_trace(
            go.Scatter(
                x=list(range(len(prices))),
                y=prices,
                mode='lines',
                name='Price',
                line=dict(color='black', width=2)
            ),
            row=2, col=1
        )
        
        # Highlight disagreement points
        disagreement_prices = [prices[i] for i in disagreement_indices]
        fig.add_trace(
            go.Scatter(
                x=disagreement_indices,
                y=disagreement_prices,
                mode='markers',
                name='Agent Disagreement',
                marker=dict(color='orange', size=8, symbol='circle')
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} Agent Decision Analysis",
        height=400 * num_rows,
        width=1200,
        showlegend=True
    )
    
    # Update axes
    fig.update_yaxes(
        title_text="Action (-1=Sell, 0=Hold, 1=Buy)", 
        tickvals=[-1, 0, 1],
        ticktext=['Sell', 'Hold', 'Buy'],
        row=1, col=1
    )
    
    fig.update_xaxes(title_text="Step", row=num_rows, col=1)
    
    # Save if path provided
    if save_path:
        fig.write_html(save_path)
    
    # Show if requested
    if show:
        fig.show()
    
    return fig


def plot_strategy_comparison(results_df, metric='cumulative_return', 
                           save_path=None, show=True, chart_type='bar'):
    """
    Plot strategy comparison chart using Plotly.
    
    Args:
        results_df: DataFrame with columns 'ticker', 'strategy', and metric columns
        metric: Metric to plot ('cumulative_return', 'sharpe_ratio', etc.)
        save_path: Path to save the HTML file (optional)
        show: Whether to show the plot (default: True)
        chart_type: Type of chart ('bar' or 'line')
        
    Returns:
        Plotly figure object
    """
    metric_titles = {
        'cumulative_return': 'Cumulative Return (%)',
        'annualized_return': 'Annualized Return (%)',
        'sharpe_ratio': 'Sharpe Ratio',
        'max_drawdown': 'Maximum Drawdown (%)',
        'win_rate': 'Win Rate (%)',
        'total_trades': 'Total Trades'
    }
    
    title = f"{metric_titles.get(metric, metric)} by Strategy and Ticker"
    
    if chart_type == 'bar':
        fig = px.bar(
            results_df, 
            x='ticker', 
            y=metric, 
            color='strategy',
            barmode='group',
            title=title,
            labels={
                'ticker': 'Ticker',
                metric: metric_titles.get(metric, metric)
            }
        )
    else:  # line chart
        fig = px.line(
            results_df, 
            x='ticker', 
            y=metric, 
            color='strategy',
            markers=True,
            title=title,
            labels={
                'ticker': 'Ticker',
                metric: metric_titles.get(metric, metric)
            }
        )
    
    # Update layout
    fig.update_layout(
        height=600,
        width=1000,
        legend_title_text='Strategy'
    )
    
    # Save if path provided
    if save_path:
        fig.write_html(save_path)
    
    # Show if requested
    if show:
        fig.show()
    
    return fig


def plot_multi_metric_comparison(results_df, save_path=None, show=True):
    """
    Plot multiple metrics for strategy comparison.
    
    Args:
        results_df: DataFrame with columns 'ticker', 'strategy', and metric columns
        save_path: Path to save the HTML file (optional)
        show: Whether to show the plot (default: True)
        
    Returns:
        Plotly figure object
    """
    # Define metrics to plot
    metrics = [
        'cumulative_return', 'sharpe_ratio', 'max_drawdown', 'win_rate'
    ]
    
    metric_titles = {
        'cumulative_return': 'Cumulative Return (%)',
        'sharpe_ratio': 'Sharpe Ratio',
        'max_drawdown': 'Maximum Drawdown (%)',
        'win_rate': 'Win Rate (%)'
    }
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, 
        cols=2,
        subplot_titles=[metric_titles[m] for m in metrics],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Get unique tickers and strategies
    tickers = results_df['ticker'].unique()
    strategies = results_df['strategy'].unique()
    
    # Assign colors to strategies
    colors = px.colors.qualitative.Plotly[:len(strategies)]
    strategy_colors = {strategy: color for strategy, color in zip(strategies, colors)}
    
    # Add each metric to the subplots
    for i, metric in enumerate(metrics):
        row = i // 2 + 1
        col = i % 2 + 1
        
        for strategy in strategies:
            strategy_data = results_df[results_df['strategy'] == strategy]
            
            fig.add_trace(
                go.Bar(
                    x=strategy_data['ticker'],
                    y=strategy_data[metric],
                    name=strategy,
                    marker_color=strategy_colors[strategy],
                    showlegend=(i == 0)  # Only show legend for first subplot
                ),
                row=row, col=col
            )
    
    # Update layout
    fig.update_layout(
        title="Multi-Metric Strategy Comparison",
        height=800,
        width=1200,
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Adjust y-axes for each metric
    # For max_drawdown, lower values are better, so invert the axis
    fig.update_yaxes(title_text=metric_titles['max_drawdown'], autorange="reversed", row=2, col=1)
    
    # Save if path provided
    if save_path:
        fig.write_html(save_path)
    
    # Show if requested
    if show:
        fig.show()
    
    return fig


def create_dashboard(results_dict, prices_dict, 
                    actions_dict=None, portfolio_values_dict=None,
                    save_path=None, show=True):
    """
    Create a comprehensive dashboard of results.
    
    Args:
        results_dict: Dictionary mapping strategy names to performance metrics
        prices_dict: Dictionary mapping strategy names to price data
        actions_dict: Dictionary mapping strategy names to action data (optional)
        portfolio_values_dict: Dictionary mapping strategy names to portfolio value data (optional)
        save_path: Path to save the HTML file (optional)
        show: Whether to show the plot (default: True)
        
    Returns:
        Plotly figure object
    """
    # Number of strategies
    num_strategies = len(results_dict)
    
    # Create a very large subplot figure
    fig = make_subplots(
        rows=num_strategies, 
        cols=2,
        subplot_titles=[f"{strategy} - Price & Actions" for strategy in results_dict.keys()] +
                       [f"{strategy} - Portfolio Value" for strategy in results_dict.keys()],
        vertical_spacing=0.05,
        horizontal_spacing=0.05
    )
    
    # Add data for each strategy
    for i, (strategy, metrics) in enumerate(results_dict.items()):
        row = i + 1
        
        # Add price chart with actions
        prices = prices_dict.get(strategy, [])
        fig.add_trace(
            go.Scatter(
                x=list(range(len(prices))),
                y=prices,
                mode='lines',
                name=f'{strategy} Price',
                line=dict(color='blue', width=1.5),
                showlegend=False
            ),
            row=row, col=1
        )
        
        # Add buy/sell markers if available
        if actions_dict and strategy in actions_dict:
            actions = actions_dict[strategy]
            
            # Buy signals
            buy_indices = [i for i, a in enumerate(actions) if a == 2]
            if buy_indices:
                buy_prices = [prices[i] for i in buy_indices]
                fig.add_trace(
                    go.Scatter(
                        x=buy_indices,
                        y=buy_prices,
                        mode='markers',
                        name=f'{strategy} Buy',
                        marker=dict(color='green', size=6, symbol='triangle-up'),
                        showlegend=False
                    ),
                    row=row, col=1
                )
            
            # Sell signals
            sell_indices = [i for i, a in enumerate(actions) if a == 0]
            if sell_indices:
                sell_prices = [prices[i] for i in sell_indices]
                fig.add_trace(
                    go.Scatter(
                        x=sell_indices,
                        y=sell_prices,
                        mode='markers',
                        name=f'{strategy} Sell',
                        marker=dict(color='red', size=6, symbol='triangle-down'),
                        showlegend=False
                    ),
                    row=row, col=1
                )
        
        # Add portfolio value chart if available
        if portfolio_values_dict and strategy in portfolio_values_dict:
            portfolio_values = portfolio_values_dict[strategy]
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(portfolio_values))),
                    y=portfolio_values,
                    mode='lines',
                    name=f'{strategy} Portfolio',
                    line=dict(color='green', width=1.5),
                    showlegend=False
                ),
                row=row, col=2
            )
        
        # Add metrics annotation
        metrics_text = f"<b>{strategy} Metrics:</b><br>"
        metrics_text += f"Return: {metrics.get('cumulative_return', 0):.2f}%<br>"
        metrics_text += f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}<br>"
        metrics_text += f"MaxDD: {metrics.get('max_drawdown', 0):.2f}%<br>"
        metrics_text += f"WinRate: {metrics.get('win_rate', 0):.2f}%"
        
        fig.add_annotation(
            xref=f"x{row*2-1}", yref=f"y{row*2-1}",
            x=0.95, y=0.95,
            text=metrics_text,
            showarrow=False,
            font=dict(size=10),
            align="right",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1,
            borderpad=4
        )
    
    # Update layout
    fig.update_layout(
        title="Trading Strategy Dashboard",
        height=350 * num_strategies,
        width=1200,
        showlegend=False
    )
    
    # Save if path provided
    if save_path:
        fig.write_html(save_path)
    
    # Show if requested
    if show:
        fig.show()
    
    return fig
