import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from dataset import VERDERawImageDataset
from model import LWEColumnViT
import numpy as np

def run_sanity_check():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 [산성 테스트] 딱 16개 샘플로 '과적합(Overfitting)' 능력을 검증합니다! (Device: {device})")
    
    # 1. 데이터 로드
    data_folder = "./data/n256_logq20_binary_for_release"
    dataset = VERDERawImageDataset(data_dir=data_folder)
    
    # [핵심 1] 전체 데이터 중 딱 16개만 잘라내서 미니 데이터셋을 만듭니다.
    subset_indices = list(range(16))
    overfit_dataset = Subset(dataset, subset_indices)
    
    # Shuffle을 False로 꺼서 매번 똑같은 순서로 똑같은 문제를 풀게 합니다.
    dataloader = DataLoader(overfit_dataset, batch_size=16, shuffle=False)
    
    # 2. 모델 세팅 (학습률을 조금 높여서 빠르게 외우도록 유도)
    model = LWEColumnViT(M=1024, n=256).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    
    # [핵심 2] AI가 꼼수를 부리지 못하도록 50배 페널티를 10배로 줄입니다.
    pos_weight = torch.tensor([10.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 200번 반복 학습
    epochs = 200
    print("-" * 60)
    for epoch in range(epochs):
        model.train()
        
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            # --- 로직 계산 및 채점 ---
            with torch.no_grad():
                preds = (logits > 0).float()
                batch_acc = (preds == labels).float().mean().item()
                
                k_values = labels.sum(dim=1).long() 
                batch_recall = 0
                for i in range(labels.size(0)):
                    k = k_values[i].item()
                    if k == 0: continue
                    _, top_k_indices = torch.topk(logits[i], k)
                    true_indices = torch.nonzero(labels[i]).squeeze(-1)
                    hits = len(np.intersect1d(top_k_indices.cpu().numpy(), true_indices.cpu().numpy()))
                    batch_recall += hits / k
                avg_recall = batch_recall / labels.size(0)

        # 너무 길어지지 않게 20번마다 한 번씩 결과 출력
        if (epoch + 1) % 20 == 0 or epoch == 0:
            sample_true = "".join([str(int(x)) for x in labels[0].cpu().numpy()[:64]])
            sample_pred = "".join([str(int(x)) for x in preds[0].cpu().numpy()[:64]])
            
            print(f"✅ Epoch [{epoch+1:3d}/{epochs}] | Loss: {loss.item():.4f} | Accuracy: {batch_acc*100:.1f}% | ✨ Recall: {avg_recall*100:.1f}%")
            if (epoch + 1) % 40 == 0:
                print(f"   [정답]: {sample_true}...")
                print(f"   [예측]: {sample_pred}...")
                print("-" * 60)

if __name__ == "__main__":
    run_sanity_check()