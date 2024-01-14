import numpy as np
import torch
from torch import nn

class EmbeddingProjection(nn.Module):

    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        # self.act_1 = nn.SiLU()
        # self.linear_2 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)
        # self.act_2 = nn.SiLU()
        # self.linear_3 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)

    def forward(self, embedding):
        hidden_states = self.linear_1(embedding)
        # hidden_states = self.act_1(hidden_states)
        # hidden_states = self.linear_2(hidden_states)
        # hidden_states = self.act_2(hidden_states)
        # hidden_states = self.linear_3(hidden_states)
        return hidden_states