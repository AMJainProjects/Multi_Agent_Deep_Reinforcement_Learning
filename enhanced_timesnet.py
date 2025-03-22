import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Inception_Block_V1(nn.Module):
    """
    Inception block for multi-scale feature extraction as described in the TimesNet paper.
    """
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        
        # Create multiple kernels of increasing size for multi-scale feature extraction
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=2*i+1, 
                padding=i
            ))
        self.kernels = nn.ModuleList(kernels)
        
        if init_weight:
            self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        # Stack and average the results from different kernel sizes
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


def FFT_for_Period(x, k=2):
    """
    Find dominant periods in time series data using Fast Fourier Transform.
    
    Args:
        x: Input tensor [Batch, Length, Channel]
        k: Number of top periods to extract
        
    Returns:
        period_list: List of dominant periods
        period_weight: Weights for each period based on FFT amplitude
    """
    # Perform FFT on input
    xf = torch.fft.rfft(x, dim=1)
    
    # Find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0  # Ignore DC component
    
    # Get top-k frequencies
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    
    # Convert to periods
    period = x.shape[1] // top_list
    
    # Return periods and their corresponding weights (amplitudes)
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    """
    Core block of TimesNet that transforms 1D time series into 2D tensors
    based on periodicity, processes them with 2D convolutions, and transforms back.
    """
    def __init__(self, seq_len, pred_len=0, top_k=2, d_model=64, d_ff=256, num_kernels=6):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        
        # Parameter-efficient inception block design
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels)
        )
        
    def forward(self, x):
        B, T, N = x.size()  # Batch, Time length, Channels
        
        # Find dominant periods
        period_list, period_weight = FFT_for_Period(x, self.k)
        
        # Process each period
        res = []
        for i in range(self.k):
            period = period_list[i]
            
            # Padding if needed
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            
            # Reshape to 2D tensor based on period
            # [Batch, Length, Channel] -> [Batch, Channel, Length//period, period]
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            
            # Apply 2D convolution to capture temporal variations
            out = self.conv(out)
            
            # Reshape back to 1D
            # [Batch, Channel, Length//period, period] -> [Batch, Length, Channel]
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        
        # Stack results from different periods
        res = torch.stack(res, dim=-1)
        
        # Adaptive aggregation with softmax weights
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        
        # Residual connection
        res = res + x
        return res


class EnhancedTimesNet(nn.Module):
    """
    Enhanced TimesNet implementation for financial time series analysis in MADDQN.
    
    This model leverages the TimesNet architecture to capture multi-scale temporal
    patterns in financial data for better Q-value estimation.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len=10, num_blocks=2, top_k=2):
        super(EnhancedTimesNet, self).__init__()
        self.input_dim = input_dim  # Number of features in input
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim  # Number of actions
        self.seq_len = seq_len
        self.num_blocks = num_blocks
        self.top_k = top_k
        
        # Input embedding
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, hidden_dim) / np.sqrt(hidden_dim))
        
        # TimesBlocks for temporal feature extraction
        self.time_blocks = nn.ModuleList([
            TimesBlock(
                seq_len=seq_len,
                pred_len=0,
                top_k=top_k,
                d_model=hidden_dim,
                d_ff=hidden_dim*4,
                num_kernels=6
            ) for _ in range(num_blocks)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Output layer to produce Q-values
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        """
        Forward pass of EnhancedTimesNet.
        
        Args:
            x: Input tensor [Batch, Length, Features]
            
        Returns:
            Q-values [Batch, output_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Input embedding
        x_embed = self.embedding(x)
        
        # Add positional encoding
        x_embed = x_embed + self.pos_encoding[:, :seq_len, :]
        
        # Apply TimesBlocks
        for i in range(self.num_blocks):
            x_embed = self.layer_norm(self.time_blocks[i](x_embed))
        
        # Apply dropout
        x_embed = self.dropout(x_embed)
        
        # Global average pooling
        x_pooled = x_embed.mean(dim=1)
        
        # Output layer
        q_values = self.output_layer(x_pooled)
        
        return q_values


# Adapter for integrating with the existing MADDQN framework
class TimesNetAgent(nn.Module):
    """
    TimesNet agent for the MADDQN framework.
    This is a wrapper that adapts the EnhancedTimesNet to the existing interface.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(TimesNetAgent, self).__init__()
        # Extract sequence length and feature dimension from state_dim
        seq_len, feature_dim = state_dim
        
        # Create the enhanced TimesNet model
        self.model = EnhancedTimesNet(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            seq_len=seq_len,
            num_blocks=2,
            top_k=3  # Number of periods to extract
        )
    
    def forward(self, x):
        """
        Forward pass of TimesNetAgent.
        
        Args:
            x: Input tensor [Batch, Length, Features]
            
        Returns:
            Q-values [Batch, action_dim]
        """
        return self.model(x)
