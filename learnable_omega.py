import torch
import torch.nn as nn

class LearnableOmegaOperator(nn.Module):
    """
    Learnable Omega Operator
    ========================
    A functional weighting and numerical integration module that computes
    a complex-valued response (Omega) from an input signal.
    """
    def __init__(self, kernel_size):
        super().__init__()

        # Learnable parameters
        self.log_D = nn.Parameter(torch.zeros(1))     # ensures D > 0
        self.phi = nn.Parameter(torch.zeros(1))       # phase
        self.kernel = nn.Parameter(torch.randn(kernel_size))  # functional kernel

    def forward(self, A_values, v_values, delta_exec=1.0):
        # Functional weighting
        weighted_signal = A_values * self.kernel

        # Numerical integration using trapezoidal rule
        # Note: torch.trapezoid is the modern equivalent of torch.trapz
        integral = torch.trapezoid(weighted_signal, v_values)

        # Stable impedance
        D = torch.exp(self.log_D) + 1e-6

        # Complex phase
        phase = torch.exp(1j * self.phi)

        # Response
        response = integral / (D * phase)

        # Execution gate
        omega = response * delta_exec

        return omega
