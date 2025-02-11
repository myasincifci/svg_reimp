import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=10, skip=False):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.lstm_cells = nn.ModuleList(
            [
                nn.LSTMCell(in_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(num_layers)
            ]
        )
        self.linear = nn.Linear(hidden_dim, in_dim)
        self.hidden_dim = hidden_dim
        self.skip = skip

    def forward(self, input, future=0):
        B, T, D = input.size()
        outputs = []

        h_t = [
            torch.zeros(B, self.hidden_dim, dtype=torch.float, device=input.device)
            for _ in range(self.num_layers)
        ]
        c_t = [
            torch.zeros(B, self.hidden_dim, dtype=torch.float, device=input.device)
            for _ in range(self.num_layers)
        ]

        for input_t in input.split(1, dim=1):
            # TODO: two cases: on the conditioning part, no linear projection at the end
            # Inputs: X, T: Condition on X, predict T

            input_t = input_t.squeeze(1)
            h_t[0], c_t[0] = self.lstm_cells[0](input_t, (h_t[0], c_t[0]))
            for i in range(1, self.num_layers):
                h_t[i], c_t[i] = self.lstm_cells[i](h_t[i - 1], (h_t[i], c_t[i]))

            if self.skip:
                output = self.linear(h_t[-1]) + input_t
            else:
                output = self.linear(h_t[-1])

            output = output[:, None, :] 
            outputs += [output]

        for i in range(future):
            input_t = output.squeeze(1)
            h_t[0], c_t[0] = self.lstm_cells[0](input_t, (h_t[0], c_t[0]))
            for i in range(1, self.num_layers):
                h_t[i], c_t[i] = self.lstm_cells[i](h_t[i - 1], (h_t[i], c_t[i]))
            
            if self.skip:
                output = self.linear(h_t[-1]) + input_t
            else:
                output = self.linear(h_t[-1])

            output = output[:, None, :]
            outputs += [output]

        outputs = torch.cat(outputs, dim=1)
        return outputs