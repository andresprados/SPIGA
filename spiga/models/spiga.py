import torch
import torch.nn as nn
import torch.nn.functional as F

import spiga.models.gnn.pose_proj as pproj
from spiga.models.cnn.cnn_multitask import MultitaskCNN
from spiga.models.gnn.step_regressor import StepRegressor, RelativePositionEncoder


class SPIGA(nn.Module):
    def __init__(self, num_landmarks=98, num_edges=15, steps=3, **kwargs):

        super(SPIGA, self).__init__()

        # Model parameters
        self.steps = steps          # Cascaded regressors
        self.embedded_dim = 512     # GAT input channel
        self.nstack = 4             # Number of stacked GATs per step
        self.kwindow = 7            # Output cropped window dimension (kernel)
        self.swindow = 0.25         # Scale of the cropped window at first step (Dft. 25% w.r.t the input featuremap)
        self.offset_ratio = [self.swindow/(2**step)/2 for step in range(self.steps)]

        # CNN parameters
        self.num_landmarks = num_landmarks
        self.num_edges = num_edges

        # Initialize backbone
        self.visual_cnn = MultitaskCNN(num_landmarks=self.num_landmarks, num_edges=self.num_edges)
        # Features dimensions
        self.img_res = self.visual_cnn.img_res
        self.visual_res = self.visual_cnn.out_res
        self.visual_dim = self.visual_cnn.ch_dim

        # Initialize Pose head
        self.channels_pose = 6
        self.pose_fc = nn.Linear(self.visual_cnn.ch_dim, self.channels_pose)

        # Initialize feature extractors:
        # Relative positional encoder
        shape_dim = 2 * (self.num_landmarks - 1)
        shape_encoder = []
        for step in range(self.steps):
            shape_encoder.append(RelativePositionEncoder(shape_dim, self.embedded_dim, [256, 256]))
        self.shape_encoder = nn.ModuleList(shape_encoder)
        # Diagonal mask used to compute relative positions
        diagonal_mask = (torch.ones(self.num_landmarks, self.num_landmarks) - torch.eye(self.num_landmarks)).type(torch.bool)
        self.diagonal_mask = nn.parameter.Parameter(diagonal_mask, requires_grad=False)

        # Visual feature extractor
        conv_window = []
        theta_S = []
        for step in range(self.steps):
            # S matrix per step
            WH = self.visual_res                                  # Width/height of ftmap
            Wout = self.swindow / (2 ** step) * WH                # Width/height of the window
            K = self.kwindow                                      # Kernel or resolution of the window
            scale = K / WH * (Wout - 1) / (K - 1)                 # Scale of the affine transformation
            # Rescale matrix S
            theta_S_stp = torch.tensor([[scale, 0], [0, scale]])
            theta_S.append(nn.parameter.Parameter(theta_S_stp, requires_grad=False))

            # Convolutional to embedded to BxLxCx1x1
            conv_window.append(nn.Conv2d(self.visual_dim, self.embedded_dim, self.kwindow))

        self.theta_S = nn.ParameterList(theta_S)
        self.conv_window = nn.ModuleList(conv_window)

        # Initialize GAT modules
        self.gcn = nn.ModuleList([StepRegressor(self.embedded_dim, 256, self.nstack) for i in range(self.steps)])

    def forward(self, data):
        # Inputs: Visual features and points projections
        pts_proj, features = self.backbone_forward(data)
        # Visual field
        visual_field = features['VisualField'][-1]

        # Params compute only once
        gat_prob = []
        features['Landmarks'] = []
        for step in range(self.steps):
            # Features generation
            embedded_ft = self.extract_embedded(pts_proj, visual_field, step)

            # GAT inference
            offset, gat_prob = self.gcn[step](embedded_ft, gat_prob)
            offset = F.hardtanh(offset)

            # Update coordinates
            pts_proj = pts_proj + self.offset_ratio[step] * offset
            features['Landmarks'].append(pts_proj.clone())

        features['GATProb'] = gat_prob
        return features

    def backbone_forward(self, data):
        # Inputs: Image and model3D
        imgs = data[0]
        model3d = data[1]
        cam_matrix = data[2]

        # HourGlass Forward
        features = self.visual_cnn(imgs)

        # Head pose estimation
        pose_raw = features['HGcore'][-1]
        B, L, _, _ = pose_raw.shape
        pose = pose_raw.reshape(B, L)
        pose = self.pose_fc(pose)
        features['Pose'] = pose.clone()

        # Project model 3D
        euler = pose[:, 0:3]
        trl = pose[:, 3:]
        rot = pproj.euler_to_rotation_matrix(euler)
        pts_proj = pproj.projectPoints(model3d, rot, trl, cam_matrix)
        pts_proj = pts_proj / self.visual_res

        return pts_proj, features

    def extract_embedded(self, pts_proj, receptive_field, step):
        # Visual features
        visual_ft = self.extract_visual_embedded(pts_proj, receptive_field, step)
        # Shape features
        shape_ft = self.calculate_distances(pts_proj)
        shape_ft = self.shape_encoder[step](shape_ft)
        # Addition
        embedded_ft = visual_ft + shape_ft
        return embedded_ft

    def extract_visual_embedded(self, pts_proj, receptive_field, step):
        # Affine matrix generation
        B, L, _ = pts_proj.shape  # Pts_proj range:[0,1]
        centers = pts_proj + 0.5 / self.visual_res  # BxLx2
        centers = centers.reshape(B * L, 2)  # B*Lx2
        theta_trl = (-1 + centers * 2).unsqueeze(-1)  # BxLx2x1
        theta_s = self.theta_S[step]  # 2x2
        theta_s = theta_s.repeat(B * L, 1, 1)  # B*Lx2x2
        theta = torch.cat((theta_s, theta_trl), -1)  # B*Lx2x3

        # Generate crop grid
        B, C, _, _ = receptive_field.shape
        grid = torch.nn.functional.affine_grid(theta, (B * L, C, self.kwindow, self.kwindow))
        grid = grid.reshape(B, L, self.kwindow, self.kwindow, 2)
        grid = grid.reshape(B, L, self.kwindow * self.kwindow, 2)

        # Crop windows
        crops = torch.nn.functional.grid_sample(receptive_field, grid, padding_mode="border")  # BxCxLxK*K
        crops = crops.transpose(1, 2)  # BxLxCxK*K
        crops = crops.reshape(B * L, C, self.kwindow, self.kwindow)

        # Flatten features
        visual_ft = self.conv_window[step](crops)
        _, Cout, _, _ = visual_ft.shape
        visual_ft = visual_ft.reshape(B, L, Cout)

        return visual_ft

    def calculate_distances(self, pts_proj):
        B, L, _ = pts_proj.shape    # BxLx2
        pts_a = pts_proj.unsqueeze(-2).repeat(1, 1, L, 1)
        pts_b = pts_a.transpose(1, 2)
        dist = pts_a - pts_b
        dist_wo_self = dist[:, self.diagonal_mask, :].reshape(B, L, -1)
        return dist_wo_self







