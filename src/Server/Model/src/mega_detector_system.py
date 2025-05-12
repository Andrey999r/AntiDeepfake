import os
import sys
import cv2
import warnings
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from scipy.signal import find_peaks
from torch.cuda.amp import autocast
from tqdm import tqdm

# ───────── CONFIG ─────────────
class Config:
    real_dir   = os.path.join(os.getcwd(), 'data', 'real')
    fake_dir   = os.path.join(os.getcwd(), 'data', 'fake')
    num_frames = 10                # для ускорения
    frame_step = max(1, 100 // num_frames)
    img_size   = 224
    device     = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_sample   = 500               # для ускорения

warnings.filterwarnings('ignore')
device = torch.device(Config.device)

# ───────── FACE DETECT ─────────
_HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = _HAAR.detectMultiScale(gray, 1.1, 5)
    if len(rects) == 0:
        return Image.new('RGB', (Config.img_size, Config.img_size))
    x, y, w, h = rects[0]
    face = frame[y:y+h, x:x+w]
    return Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

# ───────── SIMPLE SIGNAL METRIC ─────────
def flow_score(frames):
    mags = []
    for i in range(1, len(frames)):
        p = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
        c = cv2.cvtColor(frames[i],   cv2.COLOR_BGR2GRAY)
        f = cv2.calcOpticalFlowFarneback(p, c, None, 0.5,3,15,3,5,1.2,0)
        m, _ = cv2.cartToPolar(f[...,0], f[...,1])
        mags.append(m.mean())
    peaks, _ = find_peaks(mags, distance=2)
    return len(peaks) / max(1, len(mags))

# ───────── SPATIAL MODEL ─────────
class SpatialNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.feature = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(base.classifier[1].in_features, 1)
        self.tf = transforms.Compose([
            transforms.Resize((Config.img_size, Config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])

    def forward(self, faces):
        x = torch.stack([self.tf(f) for f in faces]).to(device)
        feats = self.feature(x).flatten(1)
        logits = self.fc(feats)
        return torch.sigmoid(logits).mean().item()

# ───────── MAIN DETECTOR ─────────
class MegaDeepFake:
    def __init__(self):
        self.net = SpatialNet().to(device).eval()

    def predict(self, path):
        cap = cv2.VideoCapture(path)
        frames, faces = [], []
        cnt = 0
        while len(frames) < Config.num_frames:
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
            if cnt % Config.frame_step == 0:
                faces.append(extract_face(frame))
            cnt += 1
        cap.release()
        faces += [Image.new('RGB', (Config.img_size,)*2)] * (Config.num_frames - len(faces))

        with torch.no_grad(), autocast():
            spatial_score = self.net(faces)
        flow = flow_score(frames)
        return 0.7 * spatial_score + 0.3 * flow

# ───────── TRAINER & EVAL ─────────
from torch.utils.data import Dataset, random_split
from random import sample

class VideoDataset(Dataset):
    def __init__(self, real_dir, fake_dir):
        self.paths, self.labels = [], []
        for d, l in [(real_dir, 0), (fake_dir, 1)]:
            files = os.listdir(d)[:Config.n_sample]
            for fn in files:
                self.paths.append(os.path.join(d, fn))
                self.labels.append(l)
    def __len__(self): return len(self.paths)
    def __getitem__(self, i): return self.paths[i], self.labels[i]

class Trainer:
    def __init__(self):
        ds = VideoDataset(Config.real_dir, Config.fake_dir)
        n = len(ds)
        t = int(0.8 * n)
        self.tr, self.vl = random_split(ds, [t, n - t])
        self.model = MegaDeepFake()

    def eval(self, split, name):
        corr = 0
        for path, lbl in tqdm(split, desc=f"Eval {name}"):
            score = self.model.predict(path)
            corr += (int(score > 0.5) == lbl)
        print(f"{name} Acc: {corr/len(split):.3f}")

    def run(self):
        self.eval(self.tr, 'Train')
        self.eval(self.vl, 'Val')

if __name__ == '__main__':
    if len(sys.argv) == 2:
        score = MegaDeepFake().predict(sys.argv[1])
        lbl = 'FAKE' if score > 0.5 else 'REAL'
        print(f"Score: {score:.3f} → {lbl}")
        sys.exit(int(score > 0.5))
    else:
        Trainer().run()
