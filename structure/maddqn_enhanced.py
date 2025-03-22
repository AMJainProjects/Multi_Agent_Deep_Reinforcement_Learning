import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque, namedtuple
import math

# Import the enhanced TimesNet implementation
from enhanced_timesnet import EnhancedTimesNet, TimesNetAgent

# Experience tuple for replay buffer
Experience = namedtuple('Experience', 
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    """
    Replay buffer for storing and sampling experiences.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample random batch of experiences"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """
    Base DQN Agent class that implements Double DQN algorithm.
    """
    def __init__(self, state_dim, action_dim, hidden_dim, learning_rate=0.001,
                gamma=0.99, epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=200,
                target_update=10, buffer_size=1000, batch_size=64, device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.batch_size = batch_size
        self.device = device
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Initialize network (to be overridden by subclasses)
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        
        # Initialize step counter for target network update
        self.steps_done = 0
        
    def select_action(self, state, epsilon=None):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            epsilon: Exploration rate (if None, use schedule)
            
        Returns:
            action: Selected action
        """
        # Set epsilon based on schedule if not specified
        if epsilon is None:
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                      math.exp(-1. * self.steps_done / self.epsilon_decay)
        
        self.steps_done += 1
        
        # With probability epsilon, select random action
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        
        # Otherwise, select action with highest Q-value
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.max(1)[1].item()
    
    def update(self):
        """Update networks using Double DQN algorithm"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        experiences = self.replay_buffer.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(np.array(batch.action)).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(np.array(batch.reward)).unsqueeze(1).to(self.device)
        non_final_mask = torch.BoolTensor(tuple(map(lambda s: s is not None, batch.next_state))).to(self.device)
        
        non_final_next_states = torch.FloatTensor(np.array([s for s in batch.next_state if s is not None])).to(self.device)
        
        # Get Q(s_t, a) for all actions
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.batch_size, 1, device=self.device)
        
        # Double DQN: use policy net to select action, target net to evaluate
        if len(non_final_next_states) > 0:
            with torch.no_grad():
                # Get actions from policy network
                next_actions = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                # Evaluate with target network
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_actions)
        
        # Compute expected Q values
        expected_state_action_values = reward_batch + (self.gamma * next_state_values)
        
        # Compute loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to stabilize training
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # Update target network periodically
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def get_q_values(self, state):
        """Get Q-values for a given state"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.policy_net(state_tensor).cpu().numpy()[0]

class EnhancedTimesNetAgent(DQNAgent):
    """
    DQN Agent using Enhanced TimesNet for time series feature extraction.
    This agent is used for both Risk and Return agents.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128, learning_rate=0.001,
                gamma=0.99, epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=200,
                target_update=10, buffer_size=1000, batch_size=64, device='cpu'):
        super().__init__(state_dim, action_dim, hidden_dim, learning_rate, gamma,
                        epsilon_start, epsilon_end, epsilon_decay, target_update,
                        buffer_size, batch_size, device)
        
        # Initialize TimesNet policy and target networks
        self.policy_net = TimesNetAgent(state_dim, action_dim, hidden_dim).to(device)
        self.target_net = TimesNetAgent(state_dim, action_dim, hidden_dim).to(device)
        
        # Initialize target network with policy network weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

class MultiScaleCNN(nn.Module):
    """
    Multi-Scale Convolutional Neural Network for financial time series analysis.
    
    This network applies convolutions at different scales to capture patterns
    of various time horizons, as described in the MADDQN paper.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_q_values=6):
        super(MultiScaleCNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_q_values = num_q_values
        
        # Input processing
        self.input_conv = nn.Conv1d(input_dim, hidden_dim // 2, kernel_size=1)
        
        # Multi-scale feature extraction
        # Single-scale module (1x3 filter)
        self.single_scale = nn.Conv1d(hidden_dim // 2, hidden_dim // 6, kernel_size=3, padding=1)
        
        # Medium-scale module (3x3 filter)
        self.medium_scale = nn.Conv1d(hidden_dim // 2, hidden_dim // 6, kernel_size=5, padding=2)
        
        # Global-scale module (5x5 filter)
        self.global_scale = nn.Conv1d(hidden_dim // 2, hidden_dim // 6, kernel_size=7, padding=3)
        
        # Backbone for feature processing
        self.backbone = nn.Sequential(
            nn.Conv1d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1),
            
            nn.Conv1d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            
            nn.Conv1d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU()
        )
        
        # Process Q-values from sub-agents
        self.q_processor = nn.Sequential(
            nn.Linear(num_q_values, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Output processor
        self.output_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x, q_values):
        """
        Forward pass of MultiScaleCNN.
        
        Args:
            x: Input time series data [Batch, Length, Features]
            q_values: Q-values from sub-agents [Batch, num_q_values]
            
        Returns:
            Output tensor [Batch, output_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Transpose for 1D convolution: [Batch, Features, Length]
        x = x.transpose(1, 2)
        
        # Input processing
        x = self.input_conv(x)
        
        # Multi-scale feature extraction
        x1 = self.single_scale(x)
        x2 = self.medium_scale(x)
        x3 = self.global_scale(x)
        
        # Concatenate multi-scale features
        x_combined = torch.cat([x1, x2, x3], dim=1)
        
        # Apply backbone
        x_features = self.backbone(x_combined)
        
        # Global average pooling
        x_features = F.adaptive_avg_pool1d(x_features, 1).squeeze(-1)
        
        # Process Q-values from sub-agents
        q_features = self.q_processor(q_values)
        
        # Combine features from time series and Q-values
        combined_features = torch.cat([x_features, q_features], dim=1)
        
        # Output processing
        output = self.output_processor(combined_features)
        
        return output

class FinalAgent(DQNAgent):
    """
    Final Agent that uses Multi-Scale CNN and combines Q-values from sub-agents.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128, learning_rate=0.001,
                gamma=0.99, epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=200,
                target_update=10, buffer_size=1000, batch_size=64, device='cpu'):
        super().__init__(state_dim, action_dim, hidden_dim, learning_rate, gamma,
                        epsilon_start, epsilon_end, epsilon_decay, target_update,
                        buffer_size, batch_size, device)
        
        # Number of Q-values from sub-agents (risk + return agents)
        self.num_q_values = action_dim * 2  
        
        # Initialize MultiScaleCNN policy and target networks
        self.policy_net = MultiScaleCNN(state_dim[1], hidden_dim, action_dim, self.num_q_values).to(device)
        self.target_net = MultiScaleCNN(state_dim[1], hidden_dim, action_dim, self.num_q_values).to(device)
        
        # Initialize target network with policy network weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Specialized experience buffer for final agent
        self.final_buffer = []
    
    def select_action(self, state, q_values, epsilon=None):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            q_values: Q-values from sub-agents
            epsilon: Exploration rate (if None, use schedule)
            
        Returns:
            action: Selected action
        """
        # Set epsilon based on schedule if not specified
        if epsilon is None:
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                      math.exp(-1. * self.steps_done / self.epsilon_decay)
        
        self.steps_done += 1
        
        # With probability epsilon, select random action
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        
        # Otherwise, select action with highest Q-value
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values_tensor = torch.FloatTensor(q_values).unsqueeze(0).to(self.device)
            output_q_values = self.policy_net(state_tensor, q_values_tensor)
            return output_q_values.max(1)[1].item()
    
    def get_q_values(self, state, q_values):
        """Get Q-values for a given state and sub-agent Q-values"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values_tensor = torch.FloatTensor(q_values).unsqueeze(0).to(self.device)
            return self.policy_net(state_tensor, q_values_tensor).cpu().numpy()[0]
            
    def store_experience(self, state, q_values, action, reward, next_state, next_q_values, done):
        """Store experience in the final agent's buffer"""
        self.final_buffer.append((state, q_values, action, reward, next_state, next_q_values, done))
        
        # Limit buffer size
        if len(self.final_buffer) > self.replay_buffer.capacity:
            self.final_buffer.pop(0)
    
    def update(self):
        """Update networks using Double DQN algorithm with Q-values from sub-agents"""
        if len(self.final_buffer) < self.batch_size:
            return None
        
        # Sample batch from final buffer
        batch = random.sample(self.final_buffer, min(self.batch_size, len(self.final_buffer)))
        
        # Extract components
        states = [item[0] for item in batch]
        q_values_list = [item[1] for item in batch]
        actions = [item[2] for item in batch]
        rewards = [item[3] for item in batch]
        next_states = [item[4] for item in batch]
        next_q_values_list = [item[5] for item in batch]
        dones = [item[6] for item in batch]
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(states)).to(self.device)
        q_values_batch = torch.FloatTensor(np.array(q_values_list)).to(self.device)
        action_batch = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        
        non_final_mask = torch.BoolTensor(tuple(map(lambda s: s is not None, next_states))).to(self.device)
        
        non_final_next_states = torch.FloatTensor(np.array([s for s in next_states if s is not None])).to(self.device)
        non_final_next_q_values = torch.FloatTensor(np.array([q for i, q in enumerate(next_q_values_list) if next_states[i] is not None])).to(self.device)
        
        # Get Q(s_t, a) for all actions
        state_action_values = self.policy_net(state_batch, q_values_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(len(batch), 1, device=self.device)
        
        # Double DQN: use policy net to select action, target net to evaluate
        if len(non_final_next_states) > 0:
            with torch.no_grad():
                # Get actions from policy network
                next_actions = self.policy_net(non_final_next_states, non_final_next_q_values).max(1)[1].unsqueeze(1)
                # Evaluate with target network
                next_state_values[non_final_mask] = self.target_net(non_final_next_states, non_final_next_q_values).gather(1, next_actions)
        
        # Compute expected Q values
        expected_state_action_values = reward_batch + (self.gamma * next_state_values * (~torch.tensor(dones, device=self.device).unsqueeze(1)))
        
        # Compute loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to stabilize training
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # Update target network periodically
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()

class MADDQN:
    """
    Multi-Agent Double Deep Q-Network framework.
    
    This framework integrates multiple agents with different reward functions:
    - Risk Agent: Focuses on balancing risk and return (Sharpe ratio)
    - Return Agent: Focuses on maximizing mid-term returns
    - Final Agent: Combines the outputs of both agents to make the final decision
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128, learning_rate=0.001,
                gamma=0.99, epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=200,
                target_update=10, buffer_size=1000, batch_size=64, device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.batch_size = batch_size
        
        # Initialize risk agent (Enhanced TimesNet with Sharpe ratio reward)
        self.risk_agent = EnhancedTimesNetAgent(state_dim, action_dim, hidden_dim, learning_rate,
                                       gamma, epsilon_start, epsilon_end, epsilon_decay,
                                       target_update, buffer_size, batch_size, device)
        
        # Initialize return agent (Enhanced TimesNet with mid-term return reward)
        self.return_agent = EnhancedTimesNetAgent(state_dim, action_dim, hidden_dim, learning_rate,
                                         gamma, epsilon_start, epsilon_end, epsilon_decay,
                                         target_update, buffer_size, batch_size, device)
        
        # Initialize final agent (Multi-Scale CNN with short-term reward)
        self.final_agent = FinalAgent(state_dim, action_dim, hidden_dim, learning_rate,
                                     gamma, epsilon_start, epsilon_end, epsilon_decay,
                                     target_update, buffer_size, batch_size, device)
    
    def select_action(self, state, epsilon=None):
        """
        Select trading action using the MADDQN framework
        
        Args:
            state: Current state
            epsilon: Exploration rate (if None, use schedule)
            
        Returns:
            action: The selected action
            risk_q_values: Q-values from risk agent
            return_q_values: Q-values from return agent
        """
        # Get Q-values from risk agent and return agent
        risk_q_values = self.risk_agent.get_q_values(state)
        return_q_values = self.return_agent.get_q_values(state)
        
        # Combine Q-values for final agent
        combined_q_values = np.concatenate([risk_q_values, return_q_values])
        
        # Select action using final agent
        action = self.final_agent.select_action(state, combined_q_values, epsilon)
        
        return action, risk_q_values, return_q_values
    
    def store_transition(self, state, risk_q_values, return_q_values, action, 
                         risk_reward, return_reward, final_reward, 
                         next_state, done):
        """
        Store transitions in all agents' replay buffers
        
        Args:
            state: Current state
            risk_q_values: Q-values from risk agent
            return_q_values: Q-values from return agent
            action: Action taken
            risk_reward: Reward for risk agent
            return_reward: Reward for return agent
            final_reward: Reward for final agent
            next_state: Next state
            done: Whether episode is done
        """
        # Store transition for risk agent
        self.risk_agent.replay_buffer.push(state, action, risk_reward, next_state, done)
        
        # Store transition for return agent
        self.return_agent.replay_buffer.push(state, action, return_reward, next_state, done)
        
        # Get next Q-values for final agent if next state exists
        if next_state is not None:
            next_risk_q_values = self.risk_agent.get_q_values(next_state)
            next_return_q_values = self.return_agent.get_q_values(next_state)
            next_combined_q_values = np.concatenate([next_risk_q_values, next_return_q_values])
        else:
            next_combined_q_values = None
        
        # Store transition for final agent
        combined_q_values = np.concatenate([risk_q_values, return_q_values])
        self.final_agent.store_experience(state, combined_q_values, action, final_reward, 
                                         next_state, next_combined_q_values, done)
    
    def update(self):
        """Update all agents"""
        risk_loss = self.risk_agent.update()
        return_loss = self.return_agent.update()
        final_loss = self.final_agent.update()
        
        return {
            'risk_loss': risk_loss,
            'return_loss': return_loss,
            'final_loss': final_loss
        }