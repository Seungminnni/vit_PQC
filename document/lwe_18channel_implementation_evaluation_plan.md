# 18채널 다중 특징 기반 Toy LWE 비밀키 복구 구현 및 평가 계획서

## 1. 목적

본 문서는 Toy LWE 환경에서 sparse small-integer secret \(s\)를 복구하기 위한 딥러닝 기반 실험 구현 계획을 정리한다.

핵심 목표는 다음과 같다.

- 기존 3채널 입력 구조의 한계를 개선한다.
- \(A\), \(b\), \(A\)-\(b\) 관계, 후보값 적합도, modular feature를 포함한 18채널 입력을 구성한다.
- Conv layer를 이미지 특징 추출기가 아니라 feature projection 및 token embedding 용도로 사용한다.
- Transformer 기반 coordinate-wise prediction 구조를 구현한다.
- 노이즈가 있는 경우에는 모델 예측 결과에 residual 검증기를 결합한다.
- 최종 평가는 coordinate accuracy가 아니라 full-key exact match와 residual score를 중심으로 수행한다.

---

## 2. 전체 구현 단계 요약

전체 구현은 다음 순서로 진행한다.

```text
Step 1. Toy LWE 데이터 생성기 구현
Step 2. 18채널 feature encoder 구현
Step 3. 18채널 입력을 받는 모델 구현
Step 4. 기본 supervised 학습 루프 구현
Step 5. 평가 지표 구현
Step 6. centroid 기반 예측기 구현
Step 7. top-k candidate generator 구현
Step 8. residual 기반 verifier/reranker 구현
Step 9. noise curriculum 실험
Step 10. ablation study 수행
```

---

## 3. 문제 설정

Toy LWE 식은 다음과 같다.

\[
b = As + e \pmod q
\]

여기서 각 기호의 의미는 다음과 같다.

| 기호 | 의미 |
|---|---|
| \(A\) | 공개 행렬, shape = \([M, n]\) |
| \(s\) | 비밀키 벡터, shape = \([n]\) |
| \(e\) | 작은 정수형 노이즈 |
| \(b\) | 응답 벡터, shape = \([M]\) |
| \(q\) | modular modulus |
| \(M\) | LWE 방정식 개수 |
| \(n\) | secret 차원 |

본 실험의 기본 목표는 각 좌표 \(s_j\)를 다음 7개 클래스 중 하나로 복구하는 것이다.

\[
s_j \in \{-3,-2,-1,0,1,2,3\}
\]

---

## 4. 추천 실험 파라미터

초기 실험은 너무 어렵게 시작하지 않는다. 반드시 낮은 난이도에서 구조가 정상 동작하는지 확인한 뒤 점진적으로 난이도를 올린다.

| 단계 | \(n\) | \(M\) | \(q\) | \(h\) | noise | secret range |
|---|---:|---:|---:|---:|---:|---|
| Stage 1 | 10 | 64 | 127 | 1 | 0 | \(\{-3,-2,-1,0,1,2,3\}\) |
| Stage 2 | 10 | 64 | 127 | 2 | 0 | \(\{-3,-2,-1,0,1,2,3\}\) |
| Stage 3 | 10 | 128 | 127 | 3 | 0 | \(\{-3,-2,-1,0,1,2,3\}\) |
| Stage 4 | 16 | 128 | 127 | 3 | 0 | \(\{-3,-2,-1,0,1,2,3\}\) |
| Stage 5 | 16 | 256 | 127 | 3 | 1.0 | \(\{-3,-2,-1,0,1,2,3\}\) |
| Stage 6 | 16 | 512 | 127 | 3 | 1.5 | \(\{-3,-2,-1,0,1,2,3\}\) |

여기서 \(h\)는 secret의 해밍가중치이다. 즉, \(s\)에서 nonzero 좌표의 개수이다.

---

## 5. 구현 단계 1: Toy LWE 데이터 생성기

### 5.1 Secret 생성

비밀키 \(s\)는 sparse small-integer secret으로 생성한다.

```python
def generate_sparse_s(
    batch_size,
    n,
    min_hw=3,
    max_hw=3,
    values=(-3, -2, -1, 1, 2, 3),
    device="cuda"
):
    import torch

    s = torch.zeros(batch_size, n, 1, dtype=torch.float32, device=device)
    values_tensor = torch.tensor(values, dtype=torch.float32, device=device)

    for i in range(batch_size):
        hw = torch.randint(min_hw, max_hw + 1, (1,), device=device).item()
        idx = torch.randperm(n, device=device)[:hw]
        val_idx = torch.randint(0, len(values), (hw,), device=device)
        s[i, idx, 0] = values_tensor[val_idx]

    return s
```

### 5.2 LWE 샘플 생성

```python
def generate_lwe_batch(
    batch_size,
    M,
    n,
    q,
    sigma_e=0.0,
    min_hw=3,
    max_hw=3,
    device="cuda"
):
    import torch

    s = generate_sparse_s(
        batch_size=batch_size,
        n=n,
        min_hw=min_hw,
        max_hw=max_hw,
        device=device
    )

    A = torch.randint(
        low=0,
        high=q,
        size=(batch_size, M, n),
        dtype=torch.float32,
        device=device
    )

    if sigma_e > 0:
        e = torch.round(torch.randn(batch_size, M, 1, device=device) * sigma_e)
    else:
        e = torch.zeros(batch_size, M, 1, device=device)

    b = (torch.bmm(A, s) + e) % q

    return A, b, s
```

---

## 6. 구현 단계 2: 18채널 Feature Encoder

### 6.1 핵심 설계 원칙

기존 방식처럼 \(b\)를 마지막 열로 붙이지 않는다.

잘못된 방식:

```python
X = torch.cat([A, b], dim=2)
```

제안 방식:

```python
b_exp = b.expand(-1, -1, n)
```

즉, \(b_i\)를 모든 \(j\) 위치에 broadcast하여 각 좌표 \((i,j)\)에서 \(A_{ij}\)와 \(b_i\)의 관계를 직접 계산한다.

입력 및 출력 shape는 다음과 같다.

```text
A: [B, M, n]
b: [B, M, 1]
X: [B, 18, M, n]
```

---

### 6.2 18채널 구성

| 채널 | 이름 | 정의 | 목적 |
|---:|---|---|---|
| 1 | centered A | \(c(A_{ij})\) | modular centered 값 |
| 2 | abs centered A | \(|c(A_{ij})|\) | 크기 정보 |
| 3 | sin A | \(\sin(2\pi A_{ij}/q)\) | 주기 구조 |
| 4 | cos A | \(\cos(2\pi A_{ij}/q)\) | 주기 구조 보완 |
| 5 | centered b | \(c(b_i)\) | 응답값 정보 |
| 6 | abs centered b | \(|c(b_i)|\) | 응답값 크기 |
| 7 | A-b product | \(c(A_{ij})c(b_i)\) | 방향성 관계 |
| 8 | A-b distance | \(d_q(A_{ij},b_i)\) | modular 거리 |
| 9 | compat -3 | \(d_q(b_i,-3A_{ij})\) | 후보값 -3 적합도 |
| 10 | compat -2 | \(d_q(b_i,-2A_{ij})\) | 후보값 -2 적합도 |
| 11 | compat -1 | \(d_q(b_i,-A_{ij})\) | 후보값 -1 적합도 |
| 12 | compat 1 | \(d_q(b_i,A_{ij})\) | 후보값 1 적합도 |
| 13 | compat 2 | \(d_q(b_i,2A_{ij})\) | 후보값 2 적합도 |
| 14 | compat 3 | \(d_q(b_i,3A_{ij})\) | 후보값 3 적합도 |
| 15 | parity A | \(A_{ij} \bmod 2\) | 홀짝 정보 |
| 16 | parity b | \(b_i \bmod 2\) | 홀짝 정보 |
| 17 | square A | \(A_{ij}^2 \bmod q\) | 비선형 modular 단서 |
| 18 | product A·b | \(A_{ij}b_i \bmod q\) | 결합 정보 |

---

### 6.3 Encoder 코드

```python
import torch
import numpy as np

def centered(x, q):
    half = q // 2
    return (((x + half) % q) - half) / half

def centered_int(x, q):
    half = q // 2
    return ((x + half) % q) - half

def mod_dist(x, y, q):
    d = torch.abs((x - y) % q)
    return torch.minimum(d, q - d) / (q / 2)

def encode_lwe_18ch(A, b, q):
    # A: [B, M, n]
    # b: [B, M, 1]
    # return X: [B, 18, M, n]
    B, M, n = A.shape

    b_exp = b.expand(-1, -1, n)

    cA = centered(A, q)
    cB = centered(b_exp, q)

    channels = []

    # 1~4: A 정보
    channels.append(cA)
    channels.append(torch.abs(cA))
    channels.append(torch.sin(2 * np.pi * A / q))
    channels.append(torch.cos(2 * np.pi * A / q))

    # 5~6: b 정보
    channels.append(cB)
    channels.append(torch.abs(cB))

    # 7~8: A-b 관계
    channels.append(cA * cB)
    channels.append(mod_dist(A, b_exp, q))

    # 9~14: 후보값별 compatibility
    for v in [-3, -2, -1, 1, 2, 3]:
        channels.append(mod_dist(b_exp, (v * A) % q, q))

    # 15~18: 추가 modular feature
    parity_A = ((A % 2) * 2 - 1).float()
    parity_b = ((b_exp % 2) * 2 - 1).float()
    square_A = centered((A * A) % q, q)
    prod_Ab = centered((A * b_exp) % q, q)

    channels.append(parity_A)
    channels.append(parity_b)
    channels.append(square_A)
    channels.append(prod_Ab)

    X = torch.stack(channels, dim=1)
    return X
```

---

## 7. 구현 단계 3: 모델 레이어 설계

## 7.1 전체 모델 구조

추천 모델 구조는 다음과 같다.

```text
Input X: [B, 18, M, n]

1. Channel Mixer
   - 1x1 Conv
   - 같은 좌표 (i,j)의 18개 feature를 섞음

2. Column Patch Embedding
   - Conv2d(kernel_size=(M,1))
   - 각 secret 좌표 j에 대해 M개의 방정식 evidence를 하나의 token으로 압축

3. Positional Embedding
   - 각 secret 좌표 j의 위치 정보 부여

4. Transformer Encoder
   - coordinate token들 사이의 관계 학습
   - sparse secret에서 좌표 간 상호작용 반영

5. Classifier Head
   - 각 좌표 s_j를 7-class로 분류

Output logits: [B, n, 7]
```

---

## 7.2 왜 Conv를 사용하는가

본 구조에서 Conv layer는 자연 이미지의 local pattern을 뽑기 위한 CNN 모듈이 아니다.

Conv의 역할은 다음과 같다.

| Conv 종류 | 역할 | 사용 여부 |
|---|---|---|
| \(1 \times 1\) Conv | 같은 \((i,j)\) 좌표의 18채널 feature 혼합 | 권장 |
| \((M,1)\) Conv | 각 \(j\) 좌표의 \(M\)개 방정식 evidence 압축 | 권장 |
| \(3 \times 3\) Conv | 이미지식 local pattern 추출 | 비권장 |
| \((1,n)\) Conv | row equation 전체 embedding | 후속 실험용 |

따라서 본 구현에서는 다음 조합을 기본 구조로 사용한다.

```text
1x1 Conv + (M,1) Conv + Transformer
```

---

## 7.3 기본 모델 코드

```python
import torch
import torch.nn as nn

class LWEModel18(nn.Module):
    def __init__(
        self,
        M=64,
        n=10,
        in_ch=18,
        hidden_ch=64,
        embed_dim=128,
        num_heads=4,
        depth=3,
        dropout=0.1
    ):
        super().__init__()

        self.M = M
        self.n = n
        self.embed_dim = embed_dim

        # 1. 같은 좌표의 18개 feature를 섞는 channel mixer
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=1),
            nn.GELU()
        )

        # 2. 각 column j를 하나의 token으로 압축
        self.patch_embed = nn.Conv2d(
            hidden_ch,
            embed_dim,
            kernel_size=(M, 1)
        )

        # 3. coordinate positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, n, embed_dim) * 0.02)

        # 4. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

        # 5. 7-class classifier
        self.classifier = nn.Linear(embed_dim, 7)

    def forward(self, x, return_embedding=False):
        # x: [B, 18, M, n]

        x = self.channel_mixer(x)
        # [B, hidden_ch, M, n]

        z = self.patch_embed(x)
        # [B, embed_dim, 1, n]

        z = z.squeeze(2).transpose(1, 2)
        # [B, n, embed_dim]

        z = z + self.pos_embed

        z = self.transformer(z)
        # [B, n, embed_dim]

        logits = self.classifier(z)
        # [B, n, 7]

        if return_embedding:
            return logits, z

        return logits
```

---

## 8. 구현 단계 4: 라벨 인코딩

각 secret 값을 7개 class index로 변환한다.

| secret 값 | class index |
|---:|---:|
| -3 | 0 |
| -2 | 1 |
| -1 | 2 |
| 0 | 3 |
| 1 | 4 |
| 2 | 5 |
| 3 | 6 |

```python
def secret_to_label(s):
    # s: [B, n, 1]
    # return: [B, n]
    return (s.squeeze(-1) + 3).long()

def label_to_secret(label):
    # label: [B, n]
    # return: [B, n]
    return label - 3
```

---

## 9. 구현 단계 5: 기본 학습 루프

### 9.1 Loss 설계

Sparse secret에서는 0 class가 많기 때문에 class imbalance가 발생한다.

따라서 class weight를 적용한다.

```python
class_weights = torch.tensor(
    [5.0, 5.0, 5.0, 1.0, 5.0, 5.0, 5.0],
    device=device
)

criterion = nn.CrossEntropyLoss(weight=class_weights)
```

여기서 class index 3이 secret 값 0에 해당한다.

---

### 9.2 One-step 학습 함수

```python
def train_one_step(
    model,
    optimizer,
    criterion,
    q,
    M,
    n,
    batch_size,
    sigma_e,
    min_hw,
    max_hw,
    device
):
    model.train()

    A, b, s = generate_lwe_batch(
        batch_size=batch_size,
        M=M,
        n=n,
        q=q,
        sigma_e=sigma_e,
        min_hw=min_hw,
        max_hw=max_hw,
        device=device
    )

    X = encode_lwe_18ch(A, b, q)
    labels = secret_to_label(s)

    logits = model(X)

    loss = criterion(
        logits.reshape(-1, 7),
        labels.reshape(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    with torch.no_grad():
        pred_labels = torch.argmax(logits, dim=-1)
        coord_acc = (pred_labels == labels).float().mean().item()

    return loss.item(), coord_acc
```

---

### 9.3 전체 학습 루프

```python
def train_model():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = 10
    M = 64
    q = 127
    batch_size = 128
    epochs = 100
    steps_per_epoch = 500

    sigma_e = 0.0
    min_hw = 1
    max_hw = 1

    model = LWEModel18(
        M=M,
        n=n,
        in_ch=18,
        hidden_ch=64,
        embed_dim=128,
        num_heads=4,
        depth=3,
        dropout=0.1
    ).to(device)

    class_weights = torch.tensor(
        [5.0, 5.0, 5.0, 1.0, 5.0, 5.0, 5.0],
        device=device
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.05
    )

    for epoch in range(epochs):
        losses = []
        accs = []

        for _ in range(steps_per_epoch):
            loss, acc = train_one_step(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                q=q,
                M=M,
                n=n,
                batch_size=batch_size,
                sigma_e=sigma_e,
                min_hw=min_hw,
                max_hw=max_hw,
                device=device
            )

            losses.append(loss)
            accs.append(acc)

        print(
            f"Epoch {epoch+1:03d} | "
            f"loss={sum(losses)/len(losses):.4f} | "
            f"coord_acc={sum(accs)/len(accs)*100:.2f}%"
        )

    return model
```

---

## 10. 구현 단계 6: 평가 지표

Coordinate accuracy만 평가하면 안 된다. sparse secret에서는 0이 많기 때문에 0만 잘 맞춰도 accuracy가 높게 나올 수 있다.

따라서 다음 지표를 모두 계산한다.

| 지표 | 의미 |
|---|---|
| Coordinate Accuracy | 각 \(s_j\) 값의 평균 정확도 |
| Support Accuracy | \(s_j \neq 0\) 위치를 맞췄는지 |
| Nonzero Precision | nonzero로 예측한 좌표 중 실제 nonzero 비율 |
| Nonzero Recall | 실제 nonzero 좌표 중 맞게 찾은 비율 |
| Sign Accuracy | nonzero 좌표에서 부호를 맞췄는지 |
| Magnitude Accuracy | nonzero 좌표에서 크기를 맞췄는지 |
| Full-key Exact Match | 전체 \(s\) 벡터를 완전히 맞춘 비율 |
| Residual Score | \(b - A\hat{s}\)가 얼마나 작은지 |
| Top-k Hit Rate | 정답 좌표값이 top-k 후보 안에 있는지 |

---

## 11. 평가 지표 구현

```python
def centered_mod_int(x, q):
    half = q // 2
    return ((x + half) % q) - half

def residual_score(A, b, s_hat, q):
    # A: [B, M, n]
    # b: [B, M, 1]
    # s_hat: [B, n]
    # return score: [B]

    s_hat = s_hat.float().unsqueeze(-1)
    pred_b = torch.bmm(A, s_hat) % q
    r = centered_mod_int(b - pred_b, q)
    score = (r.float() ** 2).mean(dim=1).squeeze(-1)
    return score
```

```python
@torch.no_grad()
def evaluate_model(
    model,
    q,
    M,
    n,
    batch_size,
    sigma_e,
    min_hw,
    max_hw,
    device,
    num_batches=100
):
    model.eval()

    metric_sum = {
        "coord_acc": 0.0,
        "support_acc": 0.0,
        "nonzero_precision": 0.0,
        "nonzero_recall": 0.0,
        "sign_acc": 0.0,
        "mag_acc": 0.0,
        "full_match": 0.0,
        "residual_score": 0.0
    }

    eps = 1e-8

    for _ in range(num_batches):
        A, b, s = generate_lwe_batch(
            batch_size=batch_size,
            M=M,
            n=n,
            q=q,
            sigma_e=sigma_e,
            min_hw=min_hw,
            max_hw=max_hw,
            device=device
        )

        X = encode_lwe_18ch(A, b, q)
        labels = secret_to_label(s)

        logits = model(X)
        pred_labels = torch.argmax(logits, dim=-1)

        pred_s = label_to_secret(pred_labels)
        true_s = s.squeeze(-1).long()

        coord_acc = (pred_s == true_s).float().mean()

        pred_support = pred_s != 0
        true_support = true_s != 0

        support_acc = (pred_support == true_support).float().mean()

        tp = (pred_support & true_support).float().sum()
        fp = (pred_support & ~true_support).float().sum()
        fn = (~pred_support & true_support).float().sum()

        nonzero_precision = tp / (tp + fp + eps)
        nonzero_recall = tp / (tp + fn + eps)

        nonzero_mask = true_support

        if nonzero_mask.sum() > 0:
            sign_acc = (
                torch.sign(pred_s[nonzero_mask].float())
                == torch.sign(true_s[nonzero_mask].float())
            ).float().mean()

            mag_acc = (
                torch.abs(pred_s[nonzero_mask])
                == torch.abs(true_s[nonzero_mask])
            ).float().mean()
        else:
            sign_acc = torch.tensor(0.0, device=device)
            mag_acc = torch.tensor(0.0, device=device)

        full_match = (pred_s == true_s).all(dim=1).float().mean()

        r_score = residual_score(A, b, pred_s, q).mean()

        metric_sum["coord_acc"] += coord_acc.item()
        metric_sum["support_acc"] += support_acc.item()
        metric_sum["nonzero_precision"] += nonzero_precision.item()
        metric_sum["nonzero_recall"] += nonzero_recall.item()
        metric_sum["sign_acc"] += sign_acc.item()
        metric_sum["mag_acc"] += mag_acc.item()
        metric_sum["full_match"] += full_match.item()
        metric_sum["residual_score"] += r_score.item()

    metrics = {
        k: v / num_batches
        for k, v in metric_sum.items()
    }

    return metrics
```

---

## 12. 구현 단계 7: Centroid 기반 예측기

Softmax classifier만 사용하는 방식은 모델의 마지막 linear head에 의존한다.

대안으로 각 좌표 임베딩 \(z_j\)를 추출하고, 클래스별 중심 벡터 \(\mu_c\)를 계산한 뒤 가장 가까운 중심으로 분류할 수 있다.

\[
\hat{s}_j = \arg\min_{c \in C} \|z_j - \mu_c\|_2
\]

여기서

\[
C = \{-3,-2,-1,0,1,2,3\}
\]

이다.

---

### 12.1 Centroid 계산

```python
@torch.no_grad()
def compute_centroids(
    model,
    q,
    M,
    n,
    batch_size,
    sigma_e,
    min_hw,
    max_hw,
    device,
    num_batches=200
):
    model.eval()

    embeddings_by_class = [[] for _ in range(7)]

    for _ in range(num_batches):
        A, b, s = generate_lwe_batch(
            batch_size=batch_size,
            M=M,
            n=n,
            q=q,
            sigma_e=sigma_e,
            min_hw=min_hw,
            max_hw=max_hw,
            device=device
        )

        X = encode_lwe_18ch(A, b, q)
        labels = secret_to_label(s)

        _, Z = model(X, return_embedding=True)
        # Z: [B, n, d]

        for c in range(7):
            mask = labels == c
            if mask.sum() > 0:
                embeddings_by_class[c].append(Z[mask])

    centroids = []

    for c in range(7):
        class_emb = torch.cat(embeddings_by_class[c], dim=0)
        mu_c = class_emb.mean(dim=0)
        centroids.append(mu_c)

    centroids = torch.stack(centroids, dim=0)
    return centroids
```

---

### 12.2 Centroid 예측

```python
@torch.no_grad()
def predict_by_centroid(model, X, centroids):
    # X: [B, 18, M, n]
    # centroids: [7, d]
    # return pred_s: [B, n]

    model.eval()

    _, Z = model(X, return_embedding=True)
    # Z: [B, n, d]

    B = Z.size(0)

    centroids_expand = centroids.unsqueeze(0).expand(B, -1, -1)
    dist = torch.cdist(Z, centroids_expand)
    # dist: [B, n, 7]

    pred_labels = torch.argmin(dist, dim=-1)
    pred_s = label_to_secret(pred_labels)

    return pred_s
```

---

## 13. 구현 단계 8: Top-k Candidate + Residual Reranking

노이즈가 있는 경우 모델의 argmax 결과만 신뢰하면 안 된다.

모델은 각 좌표별 후보를 줄여주는 candidate generator 역할을 하고, 최종 선택은 LWE residual 검증기로 수행한다.

---

### 13.1 Top-k 후보 추출

```python
@torch.no_grad()
def get_topk_candidates(logits, k=2):
    # logits: [B, n, 7]
    # return:
    # topk_values: [B, n, k]
    # topk_probs: [B, n, k]

    probs = torch.softmax(logits, dim=-1)
    topk_probs, topk_labels = torch.topk(probs, k=k, dim=-1)
    topk_values = label_to_secret(topk_labels)

    return topk_values, topk_probs
```

---

### 13.2 후보 조합 생성

주의: \(k^n\) 후보가 생기므로 n이 커지면 폭발한다. 따라서 처음에는 \(n=10\), \(k=2\) 정도에서만 사용한다.

```python
def enumerate_candidates_for_one_sample(topk_values_one):
    # topk_values_one: [n, k]
    # return candidates: [num_candidates, n]

    import itertools
    import torch

    n, k = topk_values_one.shape

    choices = [
        topk_values_one[j].tolist()
        for j in range(n)
    ]

    candidates = list(itertools.product(*choices))
    candidates = torch.tensor(
        candidates,
        dtype=torch.float32,
        device=topk_values_one.device
    )

    return candidates
```

---

### 13.3 Residual Reranking

```python
@torch.no_grad()
def rerank_candidates_by_residual(A_one, b_one, candidates, q):
    # A_one: [M, n]
    # b_one: [M, 1]
    # candidates: [C, n]
    # return:
    # best_s: [n]
    # best_score: scalar

    C = candidates.size(0)
    M, n = A_one.shape

    A_expand = A_one.unsqueeze(0).expand(C, -1, -1)
    b_expand = b_one.unsqueeze(0).expand(C, -1, -1)

    s_expand = candidates.unsqueeze(-1)

    pred_b = torch.bmm(A_expand, s_expand) % q

    r = centered_mod_int(b_expand - pred_b, q)
    scores = (r.float() ** 2).mean(dim=1).squeeze(-1)

    best_idx = torch.argmin(scores)
    best_s = candidates[best_idx]
    best_score = scores[best_idx]

    return best_s.long(), best_score
```

---

### 13.4 전체 Top-k + Residual 예측

```python
@torch.no_grad()
def predict_with_topk_residual(model, A, b, q, k=2):
    # A: [B, M, n]
    # b: [B, M, 1]
    # return pred_s: [B, n]

    X = encode_lwe_18ch(A, b, q)
    logits = model(X)

    topk_values, _ = get_topk_candidates(logits, k=k)

    B, n, _ = topk_values.shape

    pred_list = []

    for i in range(B):
        candidates = enumerate_candidates_for_one_sample(topk_values[i])
        best_s, best_score = rerank_candidates_by_residual(
            A_one=A[i],
            b_one=b[i],
            candidates=candidates,
            q=q
        )
        pred_list.append(best_s)

    pred_s = torch.stack(pred_list, dim=0)
    return pred_s
```

---

## 14. 노이즈 대응 전략

노이즈가 있으면 정답 secret도 residual 0을 만들지 않는다.

\[
b - A\hat{s} \approx e \pmod q
\]

따라서 노이즈 환경에서는 다음 구조를 사용한다.

```text
18채널 모델
→ 좌표별 후보 확률 생성
→ top-k 후보 선택
→ 후보 secret 조합 생성
→ residual score 계산
→ residual이 가장 작은 secret 선택
```

즉, 모델은 solver가 아니라 candidate generator로 사용하고, residual verifier가 최종 선택을 담당한다.

---

## 15. Noise Curriculum

처음부터 큰 노이즈로 학습하지 않는다.

다음 순서로 학습 난이도를 올린다.

```text
Stage 1. sigma_e = 0.0
Stage 2. sigma_e = 0.25
Stage 3. sigma_e = 0.5
Stage 4. sigma_e = 1.0
Stage 5. sigma_e = 1.5
Stage 6. sigma_e = 2.0
```

각 단계에서 다음을 기록한다.

```text
coordinate accuracy
support accuracy
nonzero precision
nonzero recall
sign accuracy
magnitude accuracy
full-key exact match
residual score
top-k hit rate
```

---

## 16. Ablation Study 계획

다음 비교 실험을 수행한다.

| 실험 | 입력 채널 | 모델 구조 | 목적 |
|---|---:|---|---|
| Baseline | 3 | 기존 ViT | 기존 코드 성능 |
| Exp 1 | 14 | Conv + Transformer | 기본 다중채널 성능 |
| Exp 2 | 18 | Conv + Transformer | 추가 feature 효과 |
| Exp 3 | 18 | 1x1 Conv 제거 | channel mixer 효과 |
| Exp 4 | 18 | Transformer 제거 | 좌표 간 attention 효과 |
| Exp 5 | 18 | softmax only | 분류기 단독 성능 |
| Exp 6 | 18 | centroid | embedding space 분리 성능 |
| Exp 7 | 18 | top-k + residual | 노이즈 대응 성능 |

---

## 17. 실험 결과 표 양식

### 17.1 채널 수 비교

| 채널 수 | Coordinate Acc | Support Acc | Sign Acc | Magnitude Acc | Full-key EM | Residual Score |
|---:|---:|---:|---:|---:|---:|---:|
| 3 |  |  |  |  |  |  |
| 14 |  |  |  |  |  |  |
| 18 |  |  |  |  |  |  |

---

### 17.2 노이즈 수준 비교

| \(\sigma_e\) | Coordinate Acc | Support Recall | Full-key EM | Top-2 Hit | Residual Score |
|---:|---:|---:|---:|---:|---:|
| 0.0 |  |  |  |  |  |
| 0.25 |  |  |  |  |  |
| 0.5 |  |  |  |  |  |
| 1.0 |  |  |  |  |  |
| 1.5 |  |  |  |  |  |
| 2.0 |  |  |  |  |  |

---

### 17.3 해밍가중치 비교

| \(h\) | Coordinate Acc | Support Acc | Full-key EM | Residual Score |
|---:|---:|---:|---:|---:|
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |
| 4 |  |  |  |  |
| 5 |  |  |  |  |

---

## 18. 구현 시 주의점

### 18.1 3x3 Conv는 기본 구조로 사용하지 않는다

LWE 행렬은 자연 이미지가 아니다. row \(i\)와 row \(i+1\), column \(j\)와 column \(j+1\)이 이미지 픽셀처럼 공간적으로 인접한 의미를 갖는다고 보기 어렵다.

따라서 일반 CNN식 `Conv2d(kernel_size=3)` 구조는 기본 구조로 사용하지 않는다.

권장 구조는 다음이다.

```text
1x1 Conv: 채널 혼합
(M,1) Conv: coordinate token 생성
Transformer: 좌표 간 관계 학습
```

---

### 18.2 Compatibility feature는 완전한 likelihood가 아니다

\(d_q(b_i, vA_{ij})\)는 \(s_j=v\)에 대한 직접적인 likelihood가 아니다.

실제 식은 다음과 같다.

\[
b_i = A_{ij}s_j + \sum_{k \ne j} A_{ik}s_k + e_i \pmod q
\]

따라서 \(b_i\)에는 다른 secret 좌표의 기여와 noise가 함께 포함된다.

그러므로 compatibility feature는 힌트일 뿐이며, 최종적으로는 Transformer의 좌표 간 관계 학습과 residual verifier가 필요하다.

---

### 18.3 Full-key Exact Match를 반드시 기록한다

Coordinate accuracy가 높아도 전체 secret을 하나라도 틀리면 실제 복구는 실패다.

따라서 가장 중요한 지표는 다음이다.

\[
EM = \mathbb{E}[\mathbf{1}(\hat{s}=s)]
\]

---

## 19. 최종 구현 목표

본 구현의 최종 목표는 다음 구조를 완성하는 것이다.

```text
A, b 생성
→ 18채널 feature tensor 생성
→ 1x1 Conv channel mixing
→ (M,1) Conv coordinate token embedding
→ Transformer encoder
→ coordinate-wise 7-class prediction
→ optional centroid prediction
→ optional top-k candidate generation
→ residual-based reranking
→ full-key recovery 평가
```

---

## 20. 최종 결론

18채널 다중 특징 기반 구조는 기존 3채널 이미지형 입력보다 LWE의 modular relation을 더 많이 보존한다. 특히 \(b_i\)를 각 좌표 \((i,j)\)에 broadcast하여 \(A_{ij}\)와의 관계를 직접 feature로 제공하는 점이 핵심이다.

Conv layer는 사용 가능하지만, 자연 이미지용 local pattern extractor가 아니라 feature projection 및 token embedding 용도로 사용해야 한다. 기본 모델은 `1x1 Conv + (M,1) Conv + Transformer` 구조가 가장 적절하다.

노이즈가 없는 toy LWE에서는 모델 단독 복구가 가능할 수 있다. 그러나 노이즈가 들어가면 모델의 argmax 예측만으로는 불안정하므로, top-k candidate generation과 residual-based reranking을 결합해야 한다.

따라서 본 연구의 구현 방향은 다음과 같이 정리된다.

```text
모델 = 후보 생성기
Residual verifier = 최종 수학적 검증기
Full-key exact match = 최종 평가 지표
```

본 실험은 실제 보안 파라미터 LWE를 공격하는 실용적 공격이라기보다, 제한된 toy setting에서 neural representation이 modular linear relation을 얼마나 학습할 수 있는지 검증하는 실험적 프레임워크로 해석해야 한다.
