"""④ 训练 CNN：读清单 CSV -> 图像 Dataset -> 训练/验证。"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import config
from src.models.cnn import SignalClassifier


class SignalImageDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, channels: int = 1):
        self.df = manifest
        self.channels = channels
        mode = "RGB" if channels == 3 else "L"
        self.to_tensor = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
        ])
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert(self.mode)
        x = self.to_tensor(img)
        if self.channels == 1 and x.shape[0] == 3:
            x = x[:1]
        y = torch.tensor(float(row["label"]), dtype=torch.float32)
        return x, y


def train_one_epoch(model, loader, opt, crit, device):
    model.train()
    total_loss, n = 0.0, 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x).squeeze(-1)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n


@torch.no_grad()
def collect_preds(model, loader, device):
    """跑一遍 loader，返回 (sigmoid 概率, 标签) 两个 numpy 数组。"""
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x).squeeze(-1)
        preds.append(torch.sigmoid(logits).cpu())
        labels.append(y)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def evaluate(model, loader, device):
    preds, labels = collect_preds(model, loader, device)
    pred_label = (preds > 0.5).astype(int)
    acc = (pred_label == labels).mean()
    from sklearn.metrics import roc_auc_score, f1_score
    auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else float("nan")
    f1 = f1_score(labels, pred_label, zero_division=0)
    return acc, f1, auc


def main():
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    channel = config.SIGNAL_CHANNELS[config.ACTIVE_CHANNEL_IDX]
    method = config.METHOD
    train_df = pd.read_csv(config.SPLITS_DIR / f"train_{channel}_{method}.csv")
    val_df = pd.read_csv(config.SPLITS_DIR / f"dev_{channel}_{method}.csv")

    train_ds = SignalImageDataset(train_df, config.N_CHANNELS)
    val_ds = SignalImageDataset(val_df, config.N_CHANNELS)
    train_loader = DataLoader(train_ds, batch_size=config.TRAIN_BATCH, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=config.VAL_BATCH, shuffle=False, num_workers=config.NUM_WORKERS)

    model = SignalClassifier(in_channels=config.N_CHANNELS).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    crit = torch.nn.BCEWithLogitsLoss()

    best_auc = 0.0
    for epoch in range(config.EPOCHS):
        loss = train_one_epoch(model, train_loader, opt, crit, device)
        acc, f1, auc = evaluate(model, val_loader, device)
        print(f"[epoch {epoch+1}/{config.EPOCHS}] loss={loss:.4f} val_acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            ckpt = config.CKPT_DIR / f"{channel}_{method}.pt"
            torch.save({"state_dict": model.state_dict(), "config": {
                "channel": channel, "method": method, "in_channels": config.N_CHANNELS}},
                       ckpt)
    print(f"best val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()