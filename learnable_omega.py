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

class MultiHeadOmegaOperator(nn.Module):
    """
    Multi-Head Omega Operator
    =========================
    Extends the Learnable Omega Operator to support multiple independent heads,
    each with its own impedance, phase, and functional kernel.
    """
    def __init__(self, num_heads, kernel_size):
        super().__init__()

        self.num_heads = num_heads

        # Each head has its own parameters
        self.log_D = nn.Parameter(torch.zeros(num_heads))
        self.phi = nn.Parameter(torch.zeros(num_heads))
        self.kernel = nn.Parameter(torch.randn(num_heads, kernel_size))

        # Domain awareness
        self.register_buffer("v_domain", torch.linspace(0, 1, kernel_size))

    def forward(self, A_values, v_values, delta_exec=1.0):
        """
        A_values: (N,)
        v_values: (N,)
        Returns:
            (num_heads,) complex tensor
        """
        # Ensure A_values is (1, N) for broadcasting
        A_reshaped = A_values.unsqueeze(0)

        # Apply kernels (broadcast over heads) and incorporate domain
        weighted = A_reshaped * self.kernel * self.v_domain  # (H, N)

        # Integrate per head along the N dimension
        integral = torch.trapezoid(weighted, v_values, dim=1)  # (H,)

        # Ensure complex type for complex division
        integral = integral.to(torch.complex64)

        # Parameters
        D = torch.exp(self.log_D) + 1e-6  # (H,)
        phase = torch.exp(1j * self.phi)  # (H,)

        # Compute response
        omega = (integral / (D * phase)) * delta_exec  # (H,)

        return omega
