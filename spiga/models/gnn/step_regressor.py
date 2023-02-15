import torch.nn as nn

from spiga.models.gnn.layers import MLP
from spiga.models.gnn.gat import GAT


class StepRegressor(nn.Module):

    def __init__(self, input_dim: int, feature_dim: int, nstack=4, decoding=[256, 128, 64, 32]):
        super(StepRegressor, self).__init__()
        assert nstack > 0
        self.nstack = nstack
        self.gat = nn.ModuleList([GAT(input_dim, feature_dim, 4)])
        for _ in range(nstack-1):
            self.gat.append(GAT(feature_dim, feature_dim, 4))
        self.decoder = OffsetDecoder(feature_dim, decoding)

    def forward(self, embedded, prob_list=[]):
        embedded = embedded.transpose(-1, -2)
        for i in range(self.nstack):
            embedded, prob = self.gat[i](embedded)
            prob_list.append(prob)
        offset = self.decoder(embedded)
        return offset.transpose(-1, -2), prob_list


class OffsetDecoder(nn.Module):
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.decoder = MLP([feature_dim] + layers + [2])

    def forward(self, embedded):
        return self.decoder(embedded)


class RelativePositionEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([input_dim] + layers + [feature_dim])

    def forward(self, feature):
        feature = feature.transpose(-1, -2)
        return self.encoder(feature).transpose(-1, -2)
