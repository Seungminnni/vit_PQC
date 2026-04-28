import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# [희소 키 생성기 - 동일]
def generate_sparse_s(batch_size, n, min_hw=3, max_hw=5, device='cuda'):
    s = torch.zeros(batch_size, n, 1, dtype=torch.float32, device=device)
    hws = torch.randint(min_hw, max_hw + 1, (batch_size,), device=device)
    for i in range(batch_size):
        hw = hws[i].item()
        idx = torch.randperm(n, device=device)[:hw]
        vals = torch.randint(1, 4, (hw,), dtype=torch.float32, device=device)
        signs = (torch.randint(0, 2, (hw,), device=device).float() * 2 - 1)
        s[i, idx, 0] = vals * signs
    return s

# [ViT 모델 - 512차원 유지]
class LWEIntegerViT(nn.Module):
    def __init__(self, M=512, n=16, embed_dim=512, num_heads=8, depth=6):
        super().__init__()
        self.n = n
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=(M, 1))
        self.pos_embed = nn.Parameter(torch.randn(1, n + 1, embed_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=0.1, batch_first=True), 
            num_layers=depth
        )
        self.classifier = nn.Linear(embed_dim, 7)

    def forward(self, x):
        x = self.patch_embed(x).squeeze(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.transformer(x)
        return self.classifier(x[:, :self.n, :])

def encode_to_img_gpu(matrix, q):
    ch1, ch2, ch3 = matrix/q, torch.abs(matrix-q/2)/(q/2), (torch.sin(2*np.pi*matrix/q)+1)/2
    return torch.stack([ch1, ch2, ch3], dim=1)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n, M, q = 16, 512, 127
    
    model = LWEIntegerViT(M=M, n=n).to(device)
    
    # 🚨 [핵심 변경] Learning Rate를 1e-4로 확 줄여서 섬세한 학습 유도
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    
    # [클래스 가중치 유지]
    class_weights = torch.tensor([5.0, 5.0, 5.0, 1.0, 5.0, 5.0, 5.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    target_s = generate_sparse_s(1, n, 3, 5, device)
    hw_count = (target_s != 0).sum().item()

    print(f"🔥 [System] 최후의 수술: 보폭 축소 (lr=1e-4)")
    print(f"🎯 픽스된 타겟 S (HW={hw_count}): {target_s.squeeze().cpu().numpy().astype(int)}")

    for epoch in range(100):
        model.train()
        pbar = tqdm(range(800), desc=f"Epoch {epoch+1:2d}")
        for _ in pbar:
            s = generate_sparse_s(128, n, 3, 5, device)
            A = torch.randint(0, q, (128, M, n), dtype=torch.float32, device=device)
            b = (torch.bmm(A, s)) % q  # 노이즈 0.0
            
            imgs = encode_to_img_gpu(torch.cat([A, b], dim=2), q)
            lbls = (s.squeeze(-1) + 3).long()

            optimizer.zero_grad()
            loss = criterion(model(imgs).view(-1, 7), lbls.view(-1))
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            test_A = torch.randint(0, q, (1, M, n), dtype=torch.float32, device=device)
            test_b = (torch.matmul(test_A, target_s)) % q
            test_img = encode_to_img_gpu(torch.cat([test_A, test_b], dim=2), q)
            
            preds = torch.argmax(model(test_img), dim=-1)
            acc = ((preds == (target_s.squeeze(-1) + 3)).float().mean().item()) * 100
            
            print(f"\n📊 Epoch {epoch+1} 실력 측정 결과")
            print(f"   예측: {(preds.cpu().numpy().flatten() - 3)}")
            print(f"   실제: {target_s.squeeze().cpu().numpy().astype(int)}")
            print(f"   정확도: {acc:.2f}%")

            if acc == 100.0:
                print("✨ 와 씨, 드디어 뚫었다!!")
                break

if __name__ == "__main__":
    main()