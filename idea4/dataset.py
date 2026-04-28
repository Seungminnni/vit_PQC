import os
import torch
import numpy as np
from torch.utils.data import Dataset

class VERDERawImageDataset(Dataset):
    def __init__(self, data_dir, q=2**20):
        self.q = q
        
        print("공통 A 행렬과 380개의 다중 비밀키 데이터 로드 중... ⚡")
        
        # 데이터 불러오기
        self.A_data = np.load(os.path.join(data_dir, "orig_A.npy"))   # shape: (1024, 256)
        self.b_data = np.load(os.path.join(data_dir, "orig_B.npy"))   # shape: (1024, 380)
        self.secrets = np.load(os.path.join(data_dir, "secret.npy"))  # shape: (256, 380)
        
        # 380개의 서로 다른 LWE 문제(샘플)가 존재함
        self.num_samples = self.b_data.shape[1] # 380
        self.M = self.A_data.shape[0]           # 1024 (방정식 개수)
        self.n = self.A_data.shape[1]           # 256 (차원)
        
        print(f"데이터 로드 완료! 총 {self.num_samples}개의 학습 샘플 생성 가능 (M={self.M}, n={self.n})")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # A는 모든 샘플이 공유하므로 그대로 가져옴
        A_mat = torch.tensor(self.A_data, dtype=torch.float32)
        
        # b와 secret은 380개 중 idx번째 열(Column)만 썰어서 가져옴
        # b_data[:, idx:idx+1]로 슬라이싱하여 (1024, 1) 형태 유지
        b_vec = torch.tensor(self.b_data[:, idx:idx+1], dtype=torch.float32)
        
        # [A | b] 형태로 가로로 이어붙이기 -> shape: (1024, 257)
        matrix = torch.cat([A_mat, b_vec], dim=1)
        
        # 3-Channel 비전 이미지 인코딩
        ch1 = matrix / self.q
        ch2 = torch.abs(matrix - (self.q / 2)) / (self.q / 2)
        ch3 = torch.sin(2 * np.pi * matrix / self.q)
        
        image_tensor = torch.stack([ch1, ch2, ch3], dim=0) # shape: (3, 1024, 257)
        
        # idx번째 정답(비밀키) 1차원 벡터 추출 -> shape: (256,)
        label = torch.tensor(self.secrets[:, idx], dtype=torch.float32)
        
        return image_tensor, label

# === 테스트 실행 블록 ===
if __name__ == "__main__":
    data_folder = "./data/n256_logq20_binary_for_release" 
    
    dataset = VERDERawImageDataset(data_dir=data_folder)
    img, label = dataset[0] # 첫 번째 샘플(문제) 꺼내기
    
    print("-" * 50)
    print("비전 모델에 들어갈 첫 번째 이미지 텐서 형태:", img.shape) 
    print("첫 번째 정답지(비밀키) 형태:", label.shape)
    print("-" * 50)