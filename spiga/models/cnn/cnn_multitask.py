from torch import nn
from spiga.models.cnn.layers import Conv, Residual
from spiga.models.cnn.hourglass import HourglassCore
from spiga.models.cnn.coord_conv import AddCoordsTh
from spiga.models.cnn.transform_e2p import E2Ptransform


class MultitaskCNN(nn.Module):

    def __init__(self, nstack=4, num_landmarks=98, num_edges=15, pose_req=True, **kwargs):
        super(MultitaskCNN, self).__init__()

        # Parameters
        self.img_res = 256                  # WxH input resolution
        self.ch_dim = 256                   # Default channel dimension
        self.out_res = 64                   # WxH output resolution
        self.nstack = nstack                # Hourglass modules stacked
        self.num_landmarks = num_landmarks  # Number of landmarks
        self.num_edges = num_edges          # Number of edges subsets (eyeR, eyeL, nose, etc)
        self.pose_required = pose_req       # Multitask flag

        # Image preprocessing
        self.pre = nn.Sequential(
            AddCoordsTh(x_dim=self.img_res, y_dim=self.img_res, with_r=True),
            Conv(6, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            Conv(128, 128, 2, 2, bn=True, relu=True),
            Residual(128, 128),
            Residual(128, self.ch_dim)
        )

        # Hourglass modules
        self.hgs = nn.ModuleList([HourglassCore(4, self.ch_dim) for i in range(self.nstack)])
        self.hgs_out = nn.ModuleList([
            nn.Sequential(
                Residual(self.ch_dim, self.ch_dim),
                Conv(self.ch_dim, self.ch_dim, 1, bn=True, relu=True)
            ) for i in range(nstack)])
        if self.pose_required:
            self.hgs_core = nn.ModuleList([
                nn.Sequential(
                    Residual(self.ch_dim, self.ch_dim),
                    Conv(self.ch_dim, self.ch_dim, 2, 2, bn=True, relu=True),
                    Residual(self.ch_dim, self.ch_dim),
                    Conv(self.ch_dim, self.ch_dim, 2, 2, bn=True, relu=True)
                ) for i in range(nstack)])

        # Attention module (ADnet style)
        self.outs_points = nn.ModuleList([nn.Sequential(Conv(self.ch_dim, self.num_landmarks, 1, relu=False, bn=False),
                                                        nn.Sigmoid()) for i in range(self.nstack - 1)])
        self.outs_edges = nn.ModuleList([nn.Sequential(Conv(self.ch_dim, self.num_edges, 1, relu=False, bn=False),
                                                       nn.Sigmoid()) for i in range(self.nstack - 1)])
        self.E2Ptransform = E2Ptransform(self.num_landmarks, self.num_edges, out_dim=self.out_res)

        self.outs_features = nn.ModuleList([Conv(self.ch_dim, self.num_landmarks, 1, relu=False, bn=False)for i in range(self.nstack - 1)])

        # Stacked Hourglass inputs (nstack > 1)
        self.merge_preds = nn.ModuleList([Conv(self.num_landmarks, self.ch_dim, 1, relu=False, bn=False) for i in range(self.nstack - 1)])
        self.merge_features = nn.ModuleList([Conv(self.ch_dim, self.ch_dim, 1, relu=False, bn=False) for i in range(self.nstack - 1)])

    def forward(self, imgs):

        x = self.pre(imgs)
        outputs = {'VisualField': [],
                   'HGcore': []}

        core_raw = []
        for i in range(self.nstack):
            # Hourglass
            hg, core_raw = self.hgs[i](x, core=core_raw)
            if self.pose_required:
                core = self.hgs_core[i](core_raw[-self.hgs[i].n])
                outputs['HGcore'].append(core)
            hg = self.hgs_out[i](hg)

            # Visual features
            outputs['VisualField'].append(hg)

            # Prepare next stacked input
            if i < self.nstack - 1:
                # Attentional modules
                points = self.outs_points[i](hg)
                edges = self.outs_edges[i](hg)
                edges_ext = self.E2Ptransform(edges)
                point_edges = points * edges_ext

                # Landmarks
                maps = self.outs_features[i](hg)
                preds = maps * point_edges

                # Outputs
                x = x + self.merge_preds[i](preds) + self.merge_features[i](hg)

        return outputs
