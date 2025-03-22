import torch
import torch.nn as nn
import numpy as np

class TimesBlock(nn.Module):
    """
    The core block of TimesNet for time-series data processing.
    
    TimesBlock transforms 1D time-series data into 2D tensors based on periodicity,
    processes them using 2D convolutional layers, and reshapes back to 1D.
    """
    def __init__(self, input_dim, output_dim, kernel_size=3):
        super(TimesBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        
        # 2D vision backbone
        self.backbone = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(output_dim),
            nn.GELU(),
            
            # Second convolutional layer
            nn.Conv2d(output_dim, output_dim, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(output_dim),
            nn.GELU()
        )
        
    def forward(self, x, period):
        """
        Forward pass of TimesBlock.
        
        Args:
            x: Input tensor [Batch, Length, Channel]
            period: Period for reshaping the time-series
            
        Returns:
            Output tensor [Batch, Length, Channel]
        """
        batch_size, length, channel = x.shape
        
        # Pad if needed to make length divisible by period
        if length % period != 0:
            padding_length = period - (length % period)
            padding = torch.zeros(batch_size, padding_length, channel, device=x.device)
            x = torch.cat([x, padding], dim=1)
            length = length + padding_length
        
        # Reshape to 2D format [Batch, Channel, Length/Period, Period]
        x_reshape = x.reshape(batch_size, length // period, period, channel)
        x_reshape = x_reshape.permute(0, 3, 1, 2)  # Rearrange dimensions
        
        # Apply 2D vision backbone
        y = self.backbone(x_reshape)
        
        # Reshape back to 1D format [Batch, Length, Channel]
        y = y.permute(0, 2, 3, 1)  # [Batch, Length/Period, Period, Channel]
        y = y.reshape(batch_size, length, channel)
        
        # Remove padding if it was added
        if length != batch_size:
            y = y[:, :batch_size, :]
        
        return y

class TimesNet(nn.Module):
    """
    TimesNet: A network designed for time series data modeling.
    
    This implementation follows the paper's description of TimesNet, which
    analyzes time-series from a multi-periodicity perspective and transforms
    1D time-series into 2D tensors for processing.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=3, top_k=2):
        super(TimesNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_blocks = num_blocks
        self.top_k = top_k
        
        # Input embedding
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding parameter
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, hidden_dim) / np.sqrt(hidden_dim))
        
        # TimesBlocks
        self.time_blocks = nn.ModuleList([
            TimesBlock(hidden_dim, hidden_dim) for _ in range(num_blocks)
        ])
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def _fft_period(self, x):
        """
        Find dominant periods using Fast Fourier Transform.
        
        Args:
            x: Input tensor [Batch, Length, Channel]
            
        Returns:
            List of periods for each sequence in the batch
        """
        batch_size, length, _ = x.shape
        
        # Compute FFT on the mean of channels
        xf = torch.fft.rfft(x.mean(dim=-1), dim=1)
        frequency_domain = torch.abs(xf)
        
        # Get frequencies
        freq = torch.fft.rfftfreq(length, d=1.0)
        
        # Find dominant frequencies (exclude DC component at freq=0)
        frequency_domain = frequency_domain[:, 1:]
        freq = freq[1:]
        
        if len(freq) == 0:  # Handle edge case
            return [[2] * self.top_k for _ in range(batch_size)]
        
        # Get top-k frequencies
        _, top_indices = torch.topk(frequency_domain, min(self.top_k, len(freq)), dim=1)
        
        # Convert frequencies to periods
        periods = []
        for i in range(batch_size):
            batch_periods = []
            for j in range(min(self.top_k, len(freq))):
                if top_indices[i, j] < len(freq):
                    period = int(1.0 / freq[top_indices[i, j].item()])
                    # Ensure period is valid (at least 2 and not larger than sequence length)
                    period = max(2, min(period, length // 2))
                    batch_periods.append(period)
                else:
                    # Default period if index is out of bounds
                    batch_periods.append(2)
            
            # Fill with default periods if needed
            while len(batch_periods) < self.top_k:
                batch_periods.append(2)
                
            periods.append(batch_periods)
        
        return periods
        
    def forward(self, x):
        """
        Forward pass of TimesNet.
        
        Args:
            x: Input tensor [Batch, Length, Features]
            
        Returns:
            Output tensor [Batch, output_dim]
        """
        batch_size, length, _ = x.shape
        
        # Input embedding
        x_embed = self.embedding(x)
        
        # Add positional encoding
        x_embed = x_embed + self.pos_encoding[:, :length, :]
        
        # Find periods using FFT
        periods = self._fft_period(x_embed)
        
        # Apply TimesBlocks for each sequence with its periods
        outputs = []
        for i in range(batch_size):
            batch_x = x_embed[i:i+1]  # Keep batch dimension
            output = batch_x
            
            # Process with each period
            for j, period in enumerate(periods[i]):
                # Apply TimesBlock and add residual connection
                output = output + self.time_blocks[j % self.num_blocks](output, period)
            
            outputs.append(output)
        
        # Combine outputs
        x = torch.cat(outputs, dim=0)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Output layer
        x = self.output_layer(x)
        
        return x
