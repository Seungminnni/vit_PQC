import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
import numpy as np
from tqdm import tqdm

class ToyLWEColumnViT(nn.Module):
    def __init__(self, M=64, n=16, embed_dim=128, num_heads=4, depth=2, dropout=0.1):
        super().__init__()
        self.M = M
        self.n = n
        
        # 장난감 규격에 맞게 커널 사이즈 축소
        self.patch_embed = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=(M, 1))
        self.pos_embed = nn.Parameter(torch.randn(1, n + 1, embed_dim))

        self.embed_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x):
        x = self.patch_embed(x)          
        x = x.squeeze(2).transpose(1, 2) 
        x = x + self.pos_embed
        x = self.embed_dropout(x)
        x = self.transformer(x)
        x = x[:, :self.n, :]             
        logits = self.classifier(x).squeeze(-1)
        return logits

def parse_args():
    parser = argparse.ArgumentParser(description="Train/test split toy binary LWE column ViT.")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--M", type=int, default=64)
    parser.add_argument("--q", type=int, default=127)
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--sigma_e", type=float, default=2.0)
    parser.add_argument("--secret_mode", default="fixed", choices=["fixed", "per_sample", "fixed_h"])
    parser.add_argument("--h", type=int, default=3)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def run_toy_experiment():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 [독자 노선] AI가 원시 LWE 암호를 깰 수 있는지 직접 증명합니다! (Device: {device})")
	    
    # 1. Toy LWE 파라미터 (n=16으로 대폭 축소)
    n = args.n
    M = args.M
    q = args.q
    num_samples = args.num_samples
    train_fraction = args.train_fraction
    batch_size = args.batch_size
	    
    print(
        f"1. Toy LWE 데이터 생성: n={n}, M={M}, q={q}, samples={num_samples}, "
        f"sigma_e={args.sigma_e}, secret_mode={args.secret_mode}, h={args.h}, "
        f"embed_dim={args.embed_dim}, heads={args.num_heads}, depth={args.depth}, "
        f"dropout={args.dropout}, wd={args.weight_decay}"
    )
    A = torch.randint(0, q, (num_samples, M, n), dtype=torch.float32)
    if args.secret_mode == "fixed":
        s_fixed = torch.randint(0, 2, (1, n, 1), dtype=torch.float32)
        s = s_fixed.expand(num_samples, -1, -1).clone()
        print(f"   fixed secret h={int(s_fixed.sum().item())}/{n}")
    elif args.secret_mode == "per_sample":
        s = torch.randint(0, 2, (num_samples, n, 1), dtype=torch.float32) # 이진 비밀키
        print(f"   per-sample secret mean h={float(s.sum(dim=1).mean().item()):.2f}/{n}")
    else:
        if args.h < 0 or args.h > n:
            raise ValueError(f"h must be in [0, n], got h={args.h}, n={n}")
        scores = torch.rand(num_samples, n)
        topk = torch.topk(scores, k=args.h, dim=1).indices
        s = torch.zeros(num_samples, n)
        s.scatter_(1, topk, 1.0)
        s = s.unsqueeze(2)
        print(f"   fixed_h secret: h={args.h}/{n} (independent per sample)")
	    
    # 정규분포를 따르는 에러(노이즈) e 생성
    e = torch.round(torch.randn(num_samples, M, 1) * args.sigma_e)
    
    # LWE 방정식: b = (A * s + e) mod q
    b = (torch.bmm(A, s) + e) % q
    
    print("2. 2D 이미지 텐서로 변환 중...")
    matrix = torch.cat([A, b], dim=2)
    ch1 = matrix / q
    ch2 = torch.abs(matrix - (q / 2)) / (q / 2)
    ch3 = torch.sin(2 * np.pi * matrix / q)
    images = torch.stack([ch1, ch2, ch3], dim=1) 
    labels = s.squeeze(2)
	    
    dataset = TensorDataset(images, labels)
    generator = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(num_samples, generator=generator)
    train_size = int(num_samples * train_fraction)
    train_dataset = Subset(dataset, perm[:train_size].tolist())
    test_dataset = Subset(dataset, perm[train_size:].tolist())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"   train/test split: {len(train_dataset)}/{len(test_dataset)}")
    
    print("3. 모델 학습 시작!")
    model = ToyLWEColumnViT(
        M=M,
        n=n,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        dropout=args.dropout,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 데이터가 쉬워졌으므로 50배 페널티는 제거하고 순수 실력으로 승부합니다.
    criterion = nn.BCEWithLogitsLoss() 

    def run_epoch(loader, training):
        model.train(training)
        total_loss = 0.0
        total_coord_correct = 0.0
        total_coords = 0
        total_exact = 0.0
        total_samples = 0

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            progress = tqdm(loader, desc="train" if training else "test", leave=False)
            for imgs, lbls in progress:
                imgs, lbls = imgs.to(device), lbls.to(device)

                if training:
                    optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, lbls)
                if training:
                    loss.backward()
                    optimizer.step()

                preds = (logits > 0).float()
                batch_size_now = lbls.shape[0]
                total_loss += loss.item() * batch_size_now
                total_coord_correct += (preds == lbls).float().sum().item()
                total_coords += lbls.numel()
                total_exact += (preds == lbls).all(dim=1).float().sum().item()
                total_samples += batch_size_now
                progress.set_postfix({"loss": f"{loss.item():.4f}"})

        return {
            "loss": total_loss / total_samples,
            "coord_acc": total_coord_correct / total_coords,
            "exact_match": total_exact / total_samples,
        }
    
    epochs = args.epochs
    for epoch in range(epochs):
        train_metrics = run_epoch(train_loader, training=True)
        test_metrics = run_epoch(test_loader, training=False)
        print(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['coord_acc']*100:.2f}% "
            f"train_exact={train_metrics['exact_match']*100:.2f}% | "
            f"test_loss={test_metrics['loss']:.4f} "
            f"test_acc={test_metrics['coord_acc']*100:.2f}% "
            f"test_exact={test_metrics['exact_match']*100:.2f}%"
        )

if __name__ == "__main__":
    run_toy_experiment()
