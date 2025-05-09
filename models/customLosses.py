""" This python file holds classes for relative MSE and relative MAE loss functions."""

import torch
import torch.nn as nn
import torch.fft

class RelMSE(nn.Module):
    """
    Relative Mean Squared Error (MSE) loss function.
    Computes the MSE between the predicted and true values, normalized by the true values.
    """
    def __init__(self, eps=1e-8):
        super(RelMSE, self).__init__()
        self.eps = eps  # small constant to avoid division by zero

    def forward(self, y_pred, y_true):
        """ Compute the relative Mean Squared Error (MSE) loss according to the formula:
        (1/n)*sum((y_true - y_pred)^2 / (y_true^2 + eps))
        
        Note: Division is inside the sum, not outside.
        """
        # Calculate squared differences
        squared_diff = (y_true - y_pred) ** 2
        
        # Calculate denominator (y_true^2) with epsilon for numerical stability
        denominator = y_true ** 2 + self.eps
        
        # Calculate element-wise relative squared errors
        relative_squared_errors = squared_diff / denominator
        
        # Calculate mean across all elements to get a scalar
        loss = torch.mean(relative_squared_errors)
        
        return loss
    

class RelMAE(nn.Module):
    """
    Relative Mean Absolute Error (MAE) loss function.
    Computes the MAE between the predicted and true values.
    """
    def __init__(self, eps=1e-8):
        super(RelMAE, self).__init__()
        self.eps = eps  # small constant to avoid division by zero

    def forward(self, y_pred, y_true):
        """ Compute the relative Mean Absolute Error (MAE) loss according to the formula:
        (1/n)*sum(abs(y_true - y_pred) / (abs(y_true) + eps))
        
        Note: Division is inside the sum, not outside.
        """
        # Calculate absolute differences
        abs_diff = torch.abs(y_true - y_pred)
        
        # Calculate denominator (abs(y_true)) with epsilon for numerical stability
        denominator = torch.abs(y_true) + self.eps
        
        # Calculate element-wise relative absolute errors
        relative_absolute_errors = abs_diff / denominator
        
        # Calculate mean across all elements to get a scalar
        loss = torch.mean(relative_absolute_errors)
        
        return loss
    
class RelMAL(nn.Module):
    """
    Relative Maximal Absolute Loss (MAL) loss function.
    Computes the MAL between the predicted and true values.
    """
    def __init__(self, eps=1e-8):
        super(RelMAL, self).__init__()
        self.eps = eps  # small constant to avoid division by zero

    def forward(self, y_pred, y_true):
        """ Compute the relative Maximal Absolute Loss (MAL) loss according to the formula:
        max(abs(y_true - y_pred) / (abs(y_true) + eps))
                """
        # Calculate absolute differences
        abs_diff = torch.abs(y_true - y_pred)
        
        # Calculate denominator (abs(y_true)) with epsilon for numerical stability
        denominator = torch.abs(y_true) + self.eps
        
        # Calculate element-wise relative absolute errors
        relative_abs_diff = abs_diff / denominator
        
        # Calculate maximum value of relative absolute differences
        loss = torch.max(relative_abs_diff)
        
        return loss
    
class CrossCorrelationShiftLoss(nn.Module):
    """
    Cross-Correlation Shift Loss.
    Penalizes the time lag between prediction and target by computing the squared lag 
    at which the cross-correlation peaks. This helps prevent time-shifted predictions.
    """
    def __init__(self, max_shift=None):
        """
        Args:
            max_shift (int, optional): If specified, clips the computed lag to this maximum value.
        """
        super().__init__()
        self.max_shift = max_shift

    def forward(self, pred, target):
        """
        Compute the shift-based loss using the peak of the cross-correlation function.
        The loss is the squared normalized lag corresponding to the peak.

        Args:
            pred (Tensor): Predicted signal of shape (batch_size, signal_length).
            target (Tensor): Ground truth signal of the same shape.

        Returns:
            Tensor: Scalar loss value.
        """
        pred = pred - pred.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)

        fft_len = 2 * pred.shape[-1]
        pred_fft = torch.fft.fft(pred, n=fft_len)
        target_fft = torch.fft.fft(target, n=fft_len)
        cc = torch.fft.ifft(pred_fft * torch.conj(target_fft)).real
        cc = torch.fft.fftshift(cc, dim=-1)

        lags = torch.arange(-pred.shape[-1], pred.shape[-1], device=pred.device)
        max_idx = torch.argmax(cc, dim=-1)
        lag = lags[max_idx]

        if self.max_shift is not None:
            lag = torch.clamp(lag, -self.max_shift, self.max_shift)

        norm_lag = (lag.float() ** 2) / (pred.shape[-1] ** 2)
        return norm_lag.mean()
    
class SpectralMSELoss(nn.Module):
    """
    Spectral Mean Squared Error (MSE) loss.
    Computes the MSE between the magnitude spectra of predicted and true signals
    using the Fast Fourier Transform (FFT).
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """
        Compute the frequency-domain MSE between the prediction and target magnitudes.

        Args:
            pred (Tensor): Predicted signal of shape (batch_size, signal_length).
            target (Tensor): Ground truth signal of the same shape.

        Returns:
            Tensor: Scalar loss value.
        """
        pred_fft = torch.fft.fft(pred, dim=-1)
        target_fft = torch.fft.fft(target, dim=-1)

        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        return nn.functional.mse_loss(pred_mag, target_mag)