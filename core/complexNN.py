import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from complexPyTorch.complexFunctions import complex_relu

class ComplexLinear(nn.Module):
    """Complex-valued Linear Layer."""
    def __init__(self, in_features, out_features, bias=True):
        super(ComplexLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Real and imaginary parts of the weights
        self.real_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.imag_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.real_bias = nn.Parameter(torch.Tensor(out_features))
            self.imag_bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('real_bias', None)
            self.register_parameter('imag_bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.real_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.imag_weight, a=math.sqrt(5))
        if self.real_bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.real_weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.real_bias, -bound, bound)
            nn.init.uniform_(self.imag_bias, -bound, bound)

    def forward(self, input):
        # Input is complex
        real_input = input.real
        imag_input = input.imag

        real_output = F.linear(real_input, self.real_weight, self.real_bias) - \
                      F.linear(imag_input, self.imag_weight, self.imag_bias)
        imag_output = F.linear(real_input, self.imag_weight, self.imag_bias) + \
                      F.linear(imag_input, self.real_weight, self.real_bias)
        output = torch.complex(real_output, imag_output)
        return output

class ModReLU(nn.Module):
    """modReLU activation function for complex-valued neural networks."""
    def __init__(self):
        super(ModReLU, self).__init__()
        # Bias is a learnable parameter
        self.bias = None

    def forward(self, input):
        # initbias if it hasn't been created yet
        if self.bias is None:
            # Get the feature size from input
            out_features = input.size(-1)
            self.bias = nn.Parameter(torch.Tensor(out_features))
            # Initialize the bias
            self.bias.data.uniform_(-0.01, 0.01)
        
        modulus = torch.abs(input)
        phase = input / (modulus + 1e-5)  # add epsilon to avoid division by zero
        activation = F.relu(modulus + self.bias)
        return activation * phase
    
class ComplexReLU(nn.Module):
    def forward(self, input):
        return complex_relu(input)