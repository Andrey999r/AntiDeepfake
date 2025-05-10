import cv2
import torch
import warnings
import numpy as np
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class Config:
    frame_step = 10
    num_frames = 20
    img_size = 224
    trans_dim = 512
    trans_heads = 8
    trans_layers = 2
    focal_alpha = 0.25
    focal_gamma = 2.0

class FocalLoss(nn.Module):
    def __init__(self, alpha=Config.focal_alpha, gamma=Config.focal_gamma):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1 - prob) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss
        return loss.mean()

class TwoStreamDeepFakeDetector(nn.Module):
    """
    Two-stream DeepFake detector: RGB + Optical Flow backbones + Temporal Transformer.
    Uses gradient checkpointing to reduce VRAM footprint.
    """
    def __init__(self):
        super().__init__()
        # RGB and Flow backbones
        self.rgb_backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        self.flow_backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        for m in (self.rgb_backbone, self.flow_backbone):
            m.classifier = nn.Identity()
        feat_dim = 1792
        # Feature projections
        self.proj_rgb = nn.Linear(feat_dim, Config.trans_dim)
        self.proj_flow = nn.Linear(feat_dim, Config.trans_dim)
        # Transformer over concatenated streams
        layer = nn.TransformerEncoderLayer(
            d_model=Config.trans_dim * 2,
            nhead=Config.trans_heads,
            dim_feedforward=Config.trans_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=Config.trans_layers)
        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(Config.trans_dim * 2),
            nn.Linear(Config.trans_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def backbone_forward(self, backbone, x):
        # Gradient checkpointing on CNN backbone
        seq = list(backbone.features.children())
        feats = checkpoint_sequential(seq, segments=Config.trans_layers, input=x, use_reentrant=False)
        feats = backbone.avgpool(feats)
        feats = torch.flatten(feats, 1)
        return feats

    def forward(self, x_rgb, x_flow):
        # x_*: [B, T, 3, H, W]
        B, T, C, H, W = x_rgb.shape
        xr = x_rgb.view(-1, C, H, W)
        xf = x_flow.view(-1, C, H, W)
        fr = self.backbone_forward(self.rgb_backbone, xr)
        ff = self.backbone_forward(self.flow_backbone, xf)
        fr = fr.view(B, T, -1)
        ff = ff.view(B, T, -1)
        pr = self.proj_rgb(fr)
        pf = self.proj_flow(ff)
        seq = torch.cat([pr, pf], dim=-1)  # [B, T, 2*trans_dim]
        seq = self.transformer(seq)        # [B, T, 2*trans_dim]
        pooled = seq.mean(dim=1)           # [B, 2*trans_dim]
        logits = self.classifier(pooled)    # [B,1]
        return logits.squeeze(1)

# Preprocessing utilities
opt_flow_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def compute_optical_flow(frames):
    flows = []
    for i in range(1, len(frames)):
        prev = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
        curr = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv = np.zeros((*prev.shape, 3), dtype=np.uint8)
        hsv[...,0] = ang * 180 / np.pi / 2
        hsv[...,1] = 255
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        flows.append(rgb)
    while len(flows) < Config.num_frames:
        flows.append(np.zeros_like(flows[0]))
    return flows

def preprocess_two_stream(video_path: str):
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    while cap.isOpened() and len(frames) < Config.num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % Config.frame_step == 0:
            frame = cv2.resize(frame, (Config.img_size, Config.img_size), interpolation=cv2.INTER_CUBIC)
            frames.append(frame)
        idx += 1
    cap.release()
    while len(frames) < Config.num_frames:
        frames.append(np.zeros_like(frames[0]))
    # RGB tensor
    tf_rgb = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    rgb_t = torch.stack([tf_rgb(f) for f in frames]).unsqueeze(0)
    # Flow tensor
    flows = compute_optical_flow(frames)
    flow_t = torch.stack([opt_flow_transform(f) for f in flows]).unsqueeze(0)
    return rgb_t, flow_t