import torch
import torch.nn as nn
import torch.nn.functional as F

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
