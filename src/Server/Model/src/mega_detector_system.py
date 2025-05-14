import os
import sys
import cv2
import logging
import warnings
import numpy as np
import librosa
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from timm import create_model
import face_alignment
from scipy.signal import find_peaks
from scipy.fftpack import fft2, fftshift
from torch.cuda.amp import autocast
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from random import sample

# ───────── CONFIG ─────────────
class Config:
    real_dir    = os.path.join(os.getcwd(), 'data', 'real')
    fake_dir    = os.path.join(os.getcwd(), 'data', 'fake')
    num_frames  = 20
    frame_step  = max(1, 200 // num_frames)
    img_size    = 224
    device      = "cuda:0"
    dr          = 16
    n_sample    = 2000
    log_path    = "logs/train.log"
    save_model_path = os.path.join(os.getcwd(), 'mega_detector_model.pth')

# ───────── LOGGING ─────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=Config.log_path, level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s', datefmt='%H:%M:%S'
)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s','%H:%M:%S'))
logging.getLogger().addHandler(ch)
warnings.filterwarnings('ignore')

device = torch.device(Config.device if torch.cuda.is_available() else "cpu")

# ───────── FACE DETECT & ALIGN ─────────
_HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
fa    = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,
                                     device=device.type)

def extract_face(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = _HAAR.detectMultiScale(gray, 1.1, 5, minSize=(50,50))
    if len(rects)==0:
        return None
    x,y,w,h = rects[0]
    face = frame[y:y+h, x:x+w]
    return Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

# ───────── SIGNAL METRICS ─────────
def flow_score(frames):
    mags = []
    for i in range(1, len(frames)):
        p = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
        c = cv2.cvtColor(frames[i],   cv2.COLOR_BGR2GRAY)
        f = cv2.calcOpticalFlowFarneback(p, c, None,
                                          0.5,3,15,3,5,1.2,0)
        m,_ = cv2.cartToPolar(f[...,0], f[...,1])
        mags.append(m.mean())
    peaks,_ = find_peaks(mags, distance=2)
    return len(peaks)/max(1,len(mags))

def freq_score(frames):
    h,w,_ = frames[0].shape
    patch = [f[h//4:3*h//4:Config.dr, w//4:3*w//4:Config.dr,1] for f in frames]
    fftm  = [np.abs(fftshift(fft2(p))) for p in patch]
    return float(np.mean([m[:m.shape[0]//2].max() for m in fftm]))

def rppg_score(frames):
    vals = [f[:,:,1].mean() for f in frames]
    return float(np.std(vals)/(np.mean(vals)+1e-6))

def eye_blink_score(frames):
    ear = []
    for fr in frames:
        lm = fa.get_landmarks(np.array(fr))
        if lm is None:
            ear.append(0.3)
            continue
        pts = lm[0]
        A = np.linalg.norm(pts[37]-pts[41])
        B = np.linalg.norm(pts[38]-pts[40])
        C = np.linalg.norm(pts[36]-pts[39])
        ear.append((A+B)/(2*C+1e-6))
    peaks,_ = find_peaks(-np.array(ear), height=0.2, distance=Config.frame_step//2)
    return len(peaks)/max(1,len(frames))

def lipsync_score(path):
    heights = []
    cap = cv2.VideoCapture(path)
    while True:
        ret,fr = cap.read()
        if not ret: break
        lm = fa.get_landmarks(fr)
        if lm is None: continue
        lm=lm[0]; top,bot = lm[48:60,1].min(), lm[48:60,1].max()
        heights.append(bot-top)
    cap.release()
    wav = path.rsplit('.',1)[0]+'.wav'
    if not os.path.isfile(wav):
        return 0.0
    y,_ = librosa.load(wav, sr=16000)
    mf = librosa.feature.mfcc(y, sr=16000, n_mfcc=13).mean(axis=0)
    L = min(len(heights), mf.shape[0])
    return float(np.corrcoef(heights[:L], mf[:L])[0,1]) if L>1 else 0.0

# ───────── SPATIAL ENSEMBLE ─────────
class SpatialEnsemble(nn.Module):
    def __init__(self):
        super().__init__()
        e = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1).to(device)
        e.classifier = nn.Identity().to(device)
        x = create_model("xception", pretrained=True, num_classes=0).to(device)
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.e, self.x, self.clip, self.proc = e, x, clip, proc
        total_dim = 1792 + x.num_features + 512
        self.proj = nn.Linear(total_dim, Config.dr*Config.dr).to(device)
        self.pool = nn.AdaptiveAvgPool2d(1).to(device)

        self.tf = transforms.Compose([
            transforms.Resize((Config.img_size,Config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def forward(self, faces):
        imgs = torch.stack([self.tf(f) for f in faces]).to(device)
        f1 = self.e.avgpool(self.e.features(imgs)).flatten(1)
        f2 = self.pool(self.x.forward_features(imgs)).flatten(1)
        with torch.no_grad():
            inp = self.proc(images=faces, return_tensors="pt").to(device)
            f3 = self.clip.get_image_features(**inp)
        feats = torch.cat([f1,f2,f3], dim=1)
        return self.proj(feats)

# ───────── TEMPORAL TRANSFORMER ─────────
class TemporalNet(nn.Module):
    def __init__(self):
        super().__init__()
        D   = Config.dr*Config.dr
        enc = nn.TransformerEncoderLayer(
            d_model=D, nhead=8,
            dim_feedforward=4*D,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        ).to(device)
        self.trans = nn.TransformerEncoder(enc, num_layers=2).to(device)
        self.head  = nn.Linear(D,1).to(device)

    def forward(self, x):
        y = self.trans(x.unsqueeze(0))
        return self.head(y.mean(1)).squeeze(1)

# ───────── MAIN ENSEMBLE ─────────
class MegaDeepFake:
    def __init__(self):
        self.sp  = SpatialEnsemble().eval()
        self.tmp = TemporalNet().eval()

    def predict(self, path):
        cap, raw, faces = cv2.VideoCapture(path), [], []
        cnt = 0
        while len(raw) < Config.num_frames:
            ret, fr = cap.read()
            if not ret: break
            raw.append(fr)
            if cnt % Config.frame_step == 0:
                f = extract_face(fr) or Image.new('RGB',(Config.img_size,)*2)
                faces.append(f)
            cnt += 1
        cap.release()
        faces += [Image.new('RGB',(Config.img_size,)*2)] * (Config.num_frames - len(faces))

        with torch.no_grad(), autocast():
            sp_feats = self.sp(faces)
            v_sp     = torch.sigmoid(self.tmp(sp_feats)).item()

        # signals on CPU
        ofs  = flow_score(raw)
        ffs  = freq_score(raw)
        rpp  = rppg_score(raw)
        eb   = eye_blink_score(raw)
        ls   = lipsync_score(path)

        return (0.5*v_sp +
                0.1*ofs + 0.1*ffs +
                0.1*rpp + 0.1*eb +
                0.1*ls)

    def save(self, path=None):
        save_path = path or Config.save_model_path
        torch.save({
            'spatial': self.sp.state_dict(),
            'temporal': self.tmp.state_dict()
        }, save_path)
        logging.info(f"Model saved to {save_path}")
        print(f"Model saved to {save_path}")

# ───────── DATASET & EVALUATOR ─────────
class VideoTrainDataset(Dataset):
    def __init__(self, rd, fd):
        self.paths, self.labels = [], []
        for d,l in [(rd,0),(fd,1)]:
            for fn in os.listdir(d):
                self.paths.append(os.path.join(d,fn))
                self.labels.append(l)
    def __len__(self): return len(self.paths)
    def __getitem__(self,i): return self.paths[i], torch.tensor(self.labels[i],dtype=torch.float32)

class Trainer:
    def __init__(self):
        logging.info("Initializing Trainer...")
        ds = VideoTrainDataset(Config.real_dir, Config.fake_dir)
        logging.info(f"Total dataset size: {len(ds)} samples")
        total = len(ds)
        split = int(0.8 * total)
        self.tr, self.vl = random_split(ds, [split, total - split])
        logging.info(f"Training set size: {len(self.tr)} — Validation set size: {len(self.vl)}")
        self.model = MegaDeepFake()
        self.model.sp.to(device)
        self.model.tmp.to(device)

    def eval_split(self, split, name=""):
        logging.info(f"Evaluating {name} split on {len(split)} samples...")
        corr = 0
        for i, (path, lbl) in enumerate(split, 1):
            label = int(lbl.item())
            pred_score = self.model.predict(path)
            pred_label = int(pred_score > 0.5)
            is_correct = (pred_label == label)
            logging.info(
                f"[{name}] {i}/{len(split)}: {os.path.basename(path)} "
                f"→ True={label}, Pred={pred_score:.4f} ({'FAKE' if pred_label else 'REAL'}) "
                f"{'✔' if is_correct else '✘'}"
            )
            corr += is_correct
        acc = corr / len(split) if split else 0
        logging.info(f"[{name}] Accuracy: {acc:.4f}")
        return acc

    def run(self):
        logging.info("=== Starting training/validation process ===")
        t_acc = self.eval_split(self.tr, name="Train")
        v_acc = self.eval_split(self.vl, name="Val")
        msg = f"Final Results — Train Acc: {t_acc:.4f} — Val Acc: {v_acc:.4f}"
        print(msg)
        logging.info(msg)
        logging.info("Saving model to disk...")
        self.model.save()
        logging.info("Training complete.")

if __name__=="__main__":
    if len(sys.argv)==2 and os.path.isfile(sys.argv[1]):
        s = MegaDeepFake().predict(sys.argv[1])
        print(f"DeepFake Score: {s:.4f} →", "FAKE" if s>0.5 else "REAL")
    else:
        Trainer().run()
