import numpy as np
import matplotlib.pyplot as plt

# 1. LWE 파라미터 설정 (우리가 사용한 Toy Scale)
n = 16
M = 64
q = 127

# 2. 샘플 데이터 1개 생성
np.random.seed(42) # 재현을 위해 시드 고정
A = np.random.randint(0, q, (M, n))
s = np.random.randint(0, 2, (n, 1))
e = np.round(np.random.normal(0, 2.0, (M, 1)))
b = (np.dot(A, s) + e) % q

# [A | b] 형태로 결합 (64x17 행렬)
matrix = np.hstack([A, b])

# 3. 3개 채널 인코딩 로직
# 채널 1: 0~1 정규화 (수치 정보)
ch1 = matrix / q

# 채널 2: 중앙값으로부터의 거리 (변동성 정보)
ch2 = np.abs(matrix - (q / 2)) / (q / 2)

# 채널 3: 사인파 인코딩 (주기성 정보)
# 시각화를 위해 -1~1 범위를 0~1로 변환
ch3 = (np.sin(2 * np.pi * matrix / q) + 1) / 2

# 4. RGB 이미지로 병합 (64, 17, 3)
rgb_img = np.stack([ch1, ch2, ch3], axis=-1)

# 5. 시각화 및 저장
fig, axes = plt.subplots(1, 4, figsize=(18, 10))

# 각 채널별 출력
axes[0].imshow(ch1, cmap='Reds', aspect='auto')
axes[0].set_title("Channel 1: Value (Red)")

axes[1].imshow(ch2, cmap='Greens', aspect='auto')
axes[1].set_title("Channel 2: Distance (Green)")

axes[2].imshow(ch3, cmap='Blues', aspect='auto')
axes[2].set_title("Channel 3: Sine (Blue)")

# 최종 통합 이미지
axes[3].imshow(rgb_img, aspect='auto')
axes[3].set_title("Final Input: RGB Combined")

# 꾸미기 (축 라벨 추가)
for ax in axes:
    ax.set_xlabel("n + 1 (Columns)")
axes[0].set_ylabel("M (Rows)")

plt.tight_layout()
plt.show() # 화면에 출력
# plt.savefig('lwe_vision_input.png', dpi=300) # 고화질 저장용