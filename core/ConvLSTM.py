"""
Purpose: ConvLSTM Implementation - continuation from meta_atom_rnn/utils/network/convlstm.py
Authors: Andy, Ethan
"""


#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import sys
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------

sys.path.append("../")


class ConvLSTMCell(LightningModule):

    def __init__(self, in_channels, out_channels,
                 kernel_size, padding, frame_size):

        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels + out_channels,
                              out_channels=4 * out_channels,
                              kernel_size=kernel_size, padding=padding)

        self.W_ci = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_co = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_cf = nn.Parameter(torch.Tensor(out_channels, *frame_size))

    def forward(self, x, h_prev, c_prev):

        conv_output = self.conv(torch.cat([x, h_prev], dim=1))

        i_conv, f_conv, c_conv, o_conv = torch.chunk(conv_output,
                                                     chunks=4, dim=1)

        input_gate = torch.sigmoid(i_conv + self.W_ci * c_prev)
        forget_gate = torch.sigmoid(f_conv + self.W_cf * c_prev)

        c = forget_gate * c_prev + input_gate * torch.tanh(c_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co * c)

        h = output_gate * torch.tanh(c)

        return h, c


class ConvLSTM(LightningModule):

    def __init__(self, in_channels, out_channels, seq_len,
                 kernel_size, padding, frame_size):

        super().__init__()

        self.target_len = seq_len
        self.convLSTMcell = ConvLSTMCell(in_channels, out_channels,
                                         kernel_size, padding, frame_size)

    def forward(self, x):

        batch_size, seq_len, channel, height, width = x.size()

        h = torch.zeros(batch_size, channel,
                        height, width, device=x.device)

        c = torch.zeros(batch_size, channel,
                        height, width, device=x.device)

        output = torch.zeros(batch_size, self.target_len,
                             channel, height, width, device=x.device)

        if seq_len == 1:
            x = x.squeeze(dim=1)
        else:
            raise NotImplementedError

        for time_step in range(self.target_len):
            h, c = self.convLSTMcell(x, h, c)
            output[:, time_step] = h

        return output
