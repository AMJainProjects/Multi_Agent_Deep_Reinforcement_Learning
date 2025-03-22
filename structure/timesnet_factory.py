"""
Factory module for TimesNet implementations selection.

This module provides factory functions to dynamically select between
the basic and enhanced TimesNet implementations.
"""
import torch

def create_timesnet_agent(state_dim, action_dim, hidden_dim=128, learning_rate=0.001,
                        gamma=0.99, epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=200,
                        target_update=10, buffer_size=1000, batch_size=64, device='cpu',
                        timesnet_type='basic'):
    """
    Factory function for creating the appropriate TimesNetAgent instance.

    Args:
        state_dim: State dimension
        action_dim: Action dimension
        hidden_dim: Hidden dimension
        learning_rate: Learning rate
        gamma: Discount factor
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay: Exploration rate decay steps
        target_update: Steps between target network updates
        buffer_size: Replay buffer size
        batch_size: Batch size
        device: Device to use
        timesnet_type: Type of TimesNet to use ('basic' or 'enhanced')

    Returns:
        TimesNetAgent instance with the specified implementation
    """
    if timesnet_type == 'basic':
        # Import TimesNetAgent from maddqn_basic
        from maddqn_basic import TimesNetAgent
    elif timesnet_type == 'enhanced':
        # Import TimesNetAgent from maddqn_enhanced
        from maddqn_enhanced import EnhancedTimesNetAgent as TimesNetAgent
    else:
        raise ValueError(f"Unknown TimesNet type: {timesnet_type}. Must be 'basic' or 'enhanced'.")

    # Create TimesNetAgent instance with the selected implementation
    return TimesNetAgent(
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

def create_maddqn(state_dim, action_dim, hidden_dim=128, learning_rate=0.001,
                 gamma=0.99, epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=200,
                 target_update=10, buffer_size=1000, batch_size=64, device='cpu',
                 timesnet_type='basic'):
    """
    Factory function for creating the appropriate MADDQN instance.

    Args:
        state_dim: State dimension
        action_dim: Action dimension
        hidden_dim: Hidden dimension
        learning_rate: Learning rate
        gamma: Discount factor
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay: Exploration rate decay steps
        target_update: Steps between target network updates
        buffer_size: Replay buffer size
        batch_size: Batch size
        device: Device to use
        timesnet_type: Type of TimesNet to use ('basic' or 'enhanced')

    Returns:
        MADDQN instance with the specified TimesNet implementation
    """
    if timesnet_type == 'basic':
        # Import MADDQN from maddqn_basic
        from maddqn_basic import MADDQN
    elif timesnet_type == 'enhanced':
        # Import MADDQN from maddqn_enhanced
        from maddqn_enhanced import MADDQN
    else:
        raise ValueError(f"Unknown TimesNet type: {timesnet_type}. Must be 'basic' or 'enhanced'.")

    # Create MADDQN instance with the selected implementation
    return MADDQN(
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