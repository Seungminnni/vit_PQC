import os
import torch
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm

def dump_all_images():
    # 저장할 폴더 생성
    output_dir = "visualized_dataset"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 데이터 파라미터 (n=16 Toy 규격)
    n, M, q = 16, 64, 127
    num_samples = 10000 

    print(f"🚀 {num_samples}개의 데이터를 이미지로 변환 중...")

    # 데이터 생성
    torch.manual_seed(42)
    A = torch.randint(0, q, (num_samples, M, n), dtype=torch.float32)
    s = torch.randint(0, 2, (num_samples, n, 1), dtype=torch.float32)
    e = torch.round(torch.randn(num_samples, M, 1) * 2.0)
    b = (torch.bmm(A, s) + e) % q

    # [A | b] 결합 및 3채널 인코딩
    matrix = torch.cat([A, b], dim=2)
    ch1 = matrix / q
    ch2 = torch.abs(matrix - (q / 2)) / (q / 2)
    ch3 = (torch.sin(2 * np.pi * matrix / q) + 1) / 2
    
    images = torch.stack([ch1, ch2, ch3], dim=1) 

    # 이미지 파일로 저장
    for i in tqdm(range(num_samples), desc="갤러리 구축 중"):
        img = images[i]
        file_path = os.path.join(output_dir, f"sample_{i:05d}.png")
        save_image(img, file_path)

    print(f"\n✨ 완료! '{output_dir}' 폴더를 확인해 보세요.")

if __name__ == "__main__":
    dump_all_images()