import torch
from torch import nn, optim
import torch.nn.functional as F


class ParserModel(nn.Module):
    def __init__(self, feature_size, label_size, hidden_size=256, use_dropout=True):
        super(ParserModel, self).__init__()
        self.hidden_size = hidden_size
        self.use_dropout = use_dropout
        self.dropout_prob = 0.2
        self.input_to_hidden_layer = nn.Linear(feature_size, self.hidden_size)
        nn.init.xavier_uniform_(self.input_to_hidden_layer.weight)

        self.dropout = nn.Dropout(p=self.dropout_prob)

        self.hidden_to_logits_layer = nn.Linear(self.hidden_size, label_size)
        nn.init.xavier_uniform_(self.hidden_to_logits_layer.weight)

    def forward(self, input_features):
        out1 = F.relu(self.input_to_hidden_layer(input_features))
        if self.use_dropout:
            out1 = self.dropout(out1)
        out = self.hidden_to_logits_layer(out1)
#        return F.softmax(out, dim=0)
        return out
