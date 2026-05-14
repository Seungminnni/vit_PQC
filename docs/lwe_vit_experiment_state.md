# LWE ViT Experiment State

이 문서는 현재까지의 LWE ViT 실험 상태를 잊지 않기 위한 고정 기록이다.

## 핵심 결론

- 현재 명확히 성공한 구조는 `row_block + block_cols=16`이다.
- `fourier_k=2`의 추가 harmonic은 핵심이 아니었다.
- `phase k=1`, 즉 `centered + sin(2*pi*x/q) + cos(2*pi*x/q)`만으로도 거의 같은 성공을 냈다.
- `raw`, 즉 `x/(q-1)`만 넣는 방식도 `m=256`, `epochs=150`에서는 성공했다.
- 따라서 현재 기준 핵심은 `k=2`가 아니라 row-equation token 구조이며, phase encoding은 raw보다 학습을 훨씬 빠르게 만드는 입력 좌표계로 보는 것이 맞다.

## 성공한 실험

### Raw Row-Block, bc16, m=256

경로:

```text
runs/lwe_vit/raw_bc16_n16_m256_100k_4096_4096_ep150/seed_0
```

설정:

```text
model=row_block
residue_encoding=raw
n=16
m=256
q=257
fixed_h=2
block_rows=1
block_cols=16
num_train=100000
num_val=4096
num_test=4096
epochs=150
batch_size=512
embed_dim=128
depth=4
num_heads=8
```

결과:

```text
best_epoch=147
test_exact_match=0.998291
test_support_f1=0.999390
test_coord_acc=0.999847
test_support_pred_rate=0.125092
test_pred_residual_std_mean=1.159504
test_oracle_residual_std_mean=1.036468
test_residual_success_rate=0.998291
```

해석:

```text
raw도 성공했다.
다만 초반에는 거의 못 배우다가 100 epoch 이후 급격히 열렸고, phase보다 훨씬 긴 학습과 더 많은 방정식 수가 필요했다.
```

### Phase10 Row-Block, Old k=2 Run

경로:

```text
runs/lwe_vit/phase10_k2_bc16_n16_m128_100k_4096_4096_ep80/seed_0
```

설정:

```text
model=row_block
n=16
m=128
q=257
fixed_h=2
block_rows=1
block_cols=16
old fourier_k=2
num_train=100000
num_val=4096
num_test=4096
epochs=80
batch_size=128
embed_dim=128
depth=4
num_heads=8
```

결과:

```text
best_epoch=38
test_exact_match=0.999512
test_support_f1=0.999756
test_coord_acc=0.999939
test_pred_residual_std_mean=1.068697
test_residual_success_rate=0.999512
```

해석:

```text
거의 완전 복원 성공.
예측 secret의 residual std가 oracle noise 수준인 약 1 근처로 내려감.
```

### Phase6 Row-Block, Old k=1 Run

경로:

```text
runs/lwe_vit/phase6_bc16_n16_m128_100k_4096_4096_ep80/seed_0
```

결과:

```text
best_epoch=54
test_exact_match=0.999268
test_support_f1=0.999817
test_coord_acc=0.999954
test_pred_residual_std_mean=1.086710
test_residual_success_rate=0.999268
```

해석:

```text
k=2 없이도 성공.
따라서 추가 Fourier harmonic은 핵심이 아님.
현재 코드에서는 fourier_k 기능을 제거하고 phase를 centered+sin+cos로 고정했다.
```

## 실패하거나 부족했던 실험

아래 run들은 2026-05-13 삭제 전에 결과를 기록해 둔 목록이다.
삭제 기준은 명확한 실패, smoke-only 실행, 또는 `summary.json`이 없는 불완전 실행이다.

| 삭제 전 run | 상태 | 핵심 결과/사유 |
| --- | --- | --- |
| `phase10_k2_bc16_n16_m128_64_32_32_ep1` | smoke | `test_exact_match=0.0`, `support_f1=0.244318`, `pred_residual_std=73.810524` |
| `phase6_bc16_n16_m128_64_32_32_ep1` | smoke | `test_exact_match=0.0`, `support_f1=0.212500`, `pred_residual_std=73.404533` |
| `raw_bc16_n16_m128_64_32_32_ep1` | smoke | `test_exact_match=0.0`, `support_f1=0.166667`, `pred_residual_std=73.884781` |
| `phase10_k2_bc16_n16_m128_8192_2048_2048_ep30` | failed/pilot | `test_exact_match=0.0`, `support_f1=0.242986`, `pred_residual_std=73.742691` |
| `raw_bc16_n16_m128_100k_4096_4096_ep50` | failed | `test_exact_match=0.0`, `support_f1=0.222222`, `support_pred_rate=1.0`, `pred_residual_std=73.901389` |
| `raw_bc1_n16_m128_10k_1024_1024_ep20_b64` | failed | `test_exact_match=0.0`, `support_f1=0.219143`, `coord_acc=0.264038`, `pred_residual_std=73.935602` |
| `raw_bc1_n16_m128_10k_1024_1024_ep20_b2` | incomplete | `summary.json` 없음. 작은 batch의 raw bc1 실험 잔여물 |
| `phase10_k2_bc4_n16_m128_50k_4096_4096_ep60` | incomplete | `summary.json` 없음. bc4 중간 실험 잔여물 |

### Raw Row-Block, bc16, m=128

삭제 전 경로:

```text
runs/lwe_vit/raw_bc16_n16_m128_100k_4096_4096_ep50/seed_0
```

설정:

```text
model=row_block
block_cols=16
residue_encoding=raw
num_train=100000
epochs=50
batch_size=128
embed_dim=128
depth=4
num_heads=8
```

결과:

```text
test_exact_match=0.0
test_support_f1=0.222222
test_coord_acc=0.125000
test_support_pred_rate=1.0
test_pred_residual_std_mean=73.901389
test_residual_success_rate=0.0
```

해석:

```text
50 epoch에서는 모델이 모든 좌표를 active로 찍는 방향으로 무너짐.
이후 m=256, 150 epoch raw 실험이 성공했으므로 raw 자체가 불가능한 것은 아니고, 이 설정은 학습량/방정식 수가 부족했던 실험으로 보는 것이 맞음.
```

## 현재 구조 정리

### Large Dataset Mode

`--on-the-fly`를 주면 `A,b,s,e` 전체를 RAM에 저장하지 않는다.
각 sample은 `split seed + idx * 1000003` 형태의 deterministic seed로 `__getitem__`에서 즉시 생성된다.

```text
train seed = run_seed * 1000 + 11
val seed   = run_seed * 1000 + 23
test seed  = run_seed * 1000 + 37
```

따라서 train/val/test는 서로 다른 seed 공간을 쓰고, 같은 split의 같은 idx는 재실행/worker 수/shuffle 여부와 무관하게 같은 LWE sample을 만든다.
이 방식은 CPU RAM 문제를 해결하지만, VRAM은 여전히 `batch_size * token_count^2`에 지배되므로 bc1 대형 실험에서는 batch를 낮춰야 한다.

새 로그 기준으로는 `input_encoding`이 실제 모델 입력 인코딩을 뜻한다.

```text
row_block:
  input_encoding=residue_encoding
  residue_encoding=raw|centered|phase
  ViT-style relation-grid block token 모델

equation_transformer:
  input_encoding=residue_encoding
  일반 row-equation Transformer baseline

row_cnn:
  input_encoding=residue_encoding
  같은 relation row를 row-local Conv1d CNN으로 처리하는 baseline
```

과거 row_block 로그에는 `use_phase=True`가 함께 찍힌 경우가 있다.
그 값은 legacy image encoder 플래그였고, row_block의 실제 입력은 `residue_encoding`이 결정한다.

현재 active model set은 위 세 개로 정리한다. 과거 실험용 `pair_token`,
`vit_patch` 모델 코드는 제거했다.

### Checkpoint/Resume

장기 실험은 중단될 수 있으므로, 현재 train script는 각 epoch 종료 후 다음 파일을 저장한다.

```text
runs/lwe_vit/<run_name>/seed_<seed>/latest.pt
```

같은 명령어를 다시 실행하면 기본적으로 `latest.pt`에서 자동 재시작한다.
이미 `summary.json`까지 생성된 완료 run이면 같은 명령어는 재학습하지 않고 기존 summary를 재사용한다.

```text
default:
  resume enabled

disable:
  --no-resume
```

저장되는 항목:

```text
model_state
optimizer_state
history
best_state
best_epoch
best_val_score
best train/val metrics
RNG state
```

최종 test는 마지막 epoch이 아니라 `best_epoch`의 `best_state`를 load해서 수행한다.
따라서 resume 이후에도 기존 원칙은 유지된다.

현재 row-block 모델은 내부적으로 LWE grid를 만든다.

```text
X[i,j] = [ENC(A_ij), ENC(b_i)]
```

`b_i`는 해당 row의 모든 column에 broadcast된다.

### raw

```text
ENC(x) = x / (q - 1)
pixel_dim = 2
```

### centered

```text
ENC(x) = centered_mod(x, q) / (q/2)
pixel_dim = 2
```

### phase6

```text
ENC(x) = [
  centered_mod(x, q) / (q/2),
  sin(2*pi*x/q),
  cos(2*pi*x/q)
]
pixel_dim = 6
```

과거 `fourier_k=2` 실험은 harmonic 2개를 썼으므로:

```text
phase10_k2:
  per residue = centered + sin/cos(k=1) + sin/cos(k=2) = 5
  A,b pair pixel_dim = 10
```

## bc16, bc4, bc1 의미

`block_cols`는 한 token이 row 안에서 몇 개 column을 묶는지 뜻한다.

```text
bc16:
  block_cols=16
  n=16이면 row 전체가 token 하나
  data tokens = m
  현재 성공한 구조
  row-equation Transformer에 가까움

bc4:
  block_cols=4
  row 하나를 4개 token으로 나눔
  data tokens = m * 4
  더 ViT스러운 중간 구조

bc1:
  block_cols=1
  각 A_ij 위치가 사실상 token 하나
  data tokens = m * n
  사용자의 원래 아이디어인 "A_ij와 b_i broadcast를 전부 token으로 넣기"에 가장 가까움
```

## bc16 세부 해석

### bc16도 모든 행을 본다

`bc16`은 한 행만 보는 모델이 아니다.
`n=16`에서 `block_cols=16`이라는 뜻은 row 하나의 모든 column을 token 하나로 묶는다는 뜻이다.

```text
bc16:
  모든 행 i=1..m을 본다.
  단, 한 행 전체를 token 하나로 묶는다.
  data tokens = m
  query tokens = n

bc1:
  모든 행 i=1..m, 모든 열 j=1..n을 본다.
  pixel 하나를 token 하나로 둔다.
  data tokens = m*n
  query tokens = n
```

예를 들어 `n=16,m=128`이면:

```text
bc16 token 수 = 128 + 16 = 144
bc1  token 수 = 128*16 + 16 = 2064
```

Transformer attention 비용은 대략 `token_count^2`에 비례한다.
따라서 같은 모델 depth, head, embed_dim을 쓰더라도 `bc1`은 `bc16`보다 훨씬 무겁다.

```text
(2064 / 144)^2 ~= 205
```

즉 `bc1`이 느린 이유는 모든 행을 봐서가 아니라, row 내부의 16개 column을 token 16개로 펼치기 때문이다.
`bc16`도 모든 row equation을 보지만, row 하나를 하나의 patch token으로 압축한다.

### bc16 token은 정보 폐기라기보다 equation embedding이다

현재 LWE relation grid는 다음과 같이 정의된다.

```text
X[i,j] = [ENC(A_ij), ENC(b_i)]
```

`b_i`는 같은 row의 모든 column에 broadcast된다.
`bc16`에서 `n=16`이면 한 row token은 다음과 같다.

```text
token_i input = [
  ENC(A_i1), ENC(b_i),
  ENC(A_i2), ENC(b_i),
  ...
  ENC(A_i16), ENC(b_i)
]
```

`raw`일 때는 `ENC(x)=x/(q-1)`이므로 feature 길이는 `2*n=32`다.
이 feature는 바로 Transformer에 들어가는 것이 아니라 학습 가능한 linear projection을 통과한다.

```text
token_i = Linear(row_features_i)
```

현재 성공한 설정에서는:

```text
input dim = 32
embed_dim = 128
```

따라서 `bc16`은 평균 pooling처럼 정보를 단순히 버리는 압축이 아니다.
한 row equation 전체를 하나의 learned equation embedding으로 바꾸는 과정이다.
오히려 feature 차원은 `32 -> 128`로 늘어난다.

문제가 될 수 있는 지점은 정보량 손실이라기보다, row 내부의 pixel-level attention을 직접 하지 않는다는 점이다.
`bc1`은 `(i,j)` pixel token끼리 attention할 수 있지만, `bc16`은 row 전체를 하나의 token으로 만든 뒤 row 간 attention을 수행한다.

### 왜 bc16이 LWE와 잘 맞을 수 있는가

LWE에서 자연스러운 관측 단위는 pixel 하나가 아니라 row equation 하나다.

```text
b_i = A_i1*s_1 + A_i2*s_2 + ... + A_i16*s_16 + e_i mod q
```

한 행 하나만 보면 미지수 `s_1..s_16`이 모두 섞여 있으므로 어느 좌표가 1인지 알 수 없다.
하지만 `m`개의 row equation을 모두 보면 secret support에 대한 반복적인 증거가 생긴다.

```text
b_1   = A_11*s_1   + ... + A_1,16*s_16   + e_1
b_2   = A_21*s_1   + ... + A_2,16*s_16   + e_2
...
b_256 = A_256,1*s_1 + ... + A_256,16*s_16 + e_256
```

따라서 `bc16`의 역할은 다음과 같이 볼 수 있다.

```text
한 row token = 하나의 LWE 방정식 evidence
Transformer = 모든 equation evidence를 모으는 장치
secret query_j = j번째 secret 좌표가 0인지 1인지 묻는 query
```

모델 입력 sequence는 다음 구조다.

```text
[equation token 1]
[equation token 2]
...
[equation token m]
[secret query 1]
[secret query 2]
...
[secret query n]
```

각 `secret query_j`는 positional embedding을 통해 자신이 몇 번째 secret 좌표를 예측해야 하는지 구분한다.
Transformer attention을 통해 query는 모든 row equation token을 읽고, `s_j`에 해당하는 증거를 모아 좌표별 logits를 만든다.

### 어느 자리가 1인지 어떻게 구분하는가

`bc16`은 row 안의 column 위치를 없애지 않는다.
row feature는 순서가 있는 벡터다.

```text
[A_i1,b_i, A_i2,b_i, ..., A_i16,b_i]
```

따라서 `A_i3`은 항상 row feature의 3번째 A 위치에 있고, `A_i7`은 항상 7번째 A 위치에 있다.
Linear projection의 weight는 이 위치별 값을 다르게 다룰 수 있다.

또한 출력 쪽에는 `n`개의 secret query token이 있으며, 각 query에는 column 위치 embedding이 붙는다.

```text
query_1 -> s_1 예측
query_2 -> s_2 예측
...
query_16 -> s_16 예측
```

즉 column 정보는 다음 두 경로로 보존된다.

```text
1. row token 내부의 feature 순서
2. secret query_j의 column positional embedding
```

그래서 `bc16`은 한 행에서 바로 답을 찾는 구조가 아니다.
한 행은 하나의 evidence이고, 모든 행 evidence를 Transformer가 모은 뒤, 각 `secret query_j`가 자기 좌표에 대한 판단을 수행한다.

### 현재 해석

`bc16`은 원래의 LWE image idea를 완전히 벗어난 구조가 아니다.
이미지 관점에서는 `1 x 16` horizontal patch를 하나의 token으로 만드는 ViT-style tokenization이다.

```text
bc1:
  pixel-level ViT
  token(i,j) = [A_ij,b_i]
  가장 세밀하지만 매우 비쌈

bc4:
  small horizontal patch ViT
  row를 4개 patch로 나눔

bc16:
  row-patch ViT / equation-row patch
  한 LWE equation row를 token 하나로 만듦
  모든 row를 보며 계산 비용이 낮음
```

따라서 현재 가장 깔끔한 연구 해석은 다음과 같다.

```text
LWE를 2D relation grid로 만든다.
그 grid 위에서 patch granularity를 조절한다.
bc1은 pixel token, bc16은 row patch token이다.
실험상 row patch가 계산 효율과 성능 모두 좋았다.
```

## 중요한 해석

현재 성공한 `bc16`은 수학적으로 깔끔하며, LWE relation grid에서 row 전체를 하나의 patch로 잡은 구조다.
다만 원래 아이디어인 위치별 pixel tokenization과 비교하면 더 큰 patch granularity를 쓴다.

```text
bc16 성공:
  한 LWE row equation 전체를 token 하나로 embedding
  Transformer가 m개 equation token과 n개 secret query token을 처리

원래 아이디어에 가까운 방향:
  phase encoding을 유지한 채 bc4, bc1을 비교
  특히 bc1은 A_ij + b_i broadcast 위치별 token 구조
```

따라서 앞으로 방향은 `bc16 -> bc4 -> bc1`로 token granularity를 바꿔 비교하는 것이다.
이 비교를 통해 row-patch가 좋은 inductive bias인지, pixel-level token이 추가 이득을 주는지 확인할 수 있다.

## 다음 실험 로드맵

현재는 `bc1`을 메인으로 밀기보다, 이미 성공한 `bc16 row-patch ViT`를 중심에 두고 난이도와 baseline을 넓히는 것이 더 현실적이다.
`bc1`은 pixel-level tokenization ablation으로 남기되, 계산 비용이 너무 크므로 AMP나 더 가벼운 attention을 넣은 뒤 다시 보는 편이 좋다.

### 1. bc16 유지 + 샘플 수 증가

목적:

```text
raw bc16 성공이 train sample 수 증가에서 더 안정적이거나 더 빠르게 열리는지 확인.
on-the-fly dataset이 500k/1m에서도 RAM 문제 없이 동작하는지 확인.
```

후보 run:

```text
raw_bc16_n16_m256_500k_10k_10k_ep80
raw_bc16_n16_m256_1m_10k_10k_ep80
```

해석:

```text
성공하면:
  row-patch 구조가 더 큰 synthetic sample regime에서도 안정적이라는 근거.

시간만 늘고 성능 차이가 작으면:
  현재 n=16,h=2에서는 100k sample로 이미 충분할 수 있음.
```

### 2. bc16 + variable_h

목적:

```text
fixed_h=2라는 강한 prior를 약화한다.
정확히 active가 2개라는 정보를 쓰지 않아도 학습되는지 확인한다.
```

후보 run:

```text
raw_bc16_n16_m256_varh1to4_100k_10k_10k_ep150
phase6_bc16_n16_m128_varh1to4_100k_10k_10k_ep80
```

해석:

```text
성공하면:
  고정 top-k prior에만 기대는 것이 아니라,
  A,b relation에서 support 크기 변화까지 어느 정도 처리한다는 의미.

실패하면:
  현재 성공은 fixed_h=2 prior에 많이 의존했을 가능성이 큼.
```

### 3. bc16 + 더 큰 Hamming weight

목적:

```text
sparse support recovery가 아니라 더 밀도 높은 binary secret에서도 되는지 확인.
```

후보 run:

```text
raw_bc16_n16_m256_h4_100k_10k_10k_ep150
raw_bc16_n16_m256_h8_100k_10k_10k_ep150
phase6_bc16_n16_m128_h8_100k_10k_10k_ep80
```

해석:

```text
h=4 성공:
  fixed_h=2보다 어려운 sparse secret에서도 가능.

h=8 성공:
  n=16의 절반이 active인 regime.
  단순 sparse prior를 넘어선 신호를 배운다는 의미가 커짐.
```

### 4. raw vs phase6 비교

목적:

```text
raw도 성공했지만 늦게 열린다.
어려운 조건(variable_h, h 증가)에서 phase6가 학습 속도와 안정성을 얼마나 개선하는지 확인한다.
```

비교 축:

```text
raw:
  ENC(x)=x/(q-1)
  A,b만 쓰는 가장 단순한 입력.

phase6:
  ENC(x)=centered + sin(2*pi*x/q) + cos(2*pi*x/q)
  A,b만 쓰지만 modulo geometry를 더 잘 펼친 입력.
```

주의:

```text
phase6도 oracle 정보를 추가하는 것은 아니다.
입력 정보는 여전히 A,b뿐이고, 같은 residue를 다른 좌표계로 표현하는 것이다.
```

### 5. Plain Equation Transformer Baseline

목적:

```text
현재 성공이 "ViT-style relation grid" 덕인지,
아니면 row equation token Transformer만으로도 충분한지 확인한다.
```

비교:

```text
row-patch ViT bc16:
  token_i = [A_i1,b_i, A_i2,b_i, ..., A_i16,b_i]

plain equation Transformer:
  token_i = [A_i1, A_i2, ..., A_i16, b_i]
```

해석:

```text
둘 다 잘 되면:
  한 row equation을 token으로 보는 구조 자체가 핵심일 수 있음.

row-patch ViT가 더 좋으면:
  b_i broadcast와 [A_ij,b_i] relation pixelization이 실제 이득을 준다는 근거.
```

구현 상태:

```text
구현 완료.
CLI: --model equation_transformer
기존 loss/metrics/residual 검증은 그대로 공유.
```

구현 플랜:

```text
목표:
  row-patch ViT가 잘 되는 이유가 relation-grid tokenization 때문인지,
  아니면 m개 equation row를 Transformer가 보는 것만으로 충분한지 분리한다.

공정성 원칙:
  데이터 생성은 완전히 동일하게 둔다.
  입력 정보도 A,b만 사용한다.
  secret/noise/top-k oracle 정보는 넣지 않는다.
  residue_encoding(raw/centered/phase)은 row_block과 같은 옵션을 쓴다.
  loss, class weight, residual metric, exact/support metrics는 그대로 공유한다.
  secret query token n개를 추가하는 head 구조도 row_block과 동일하게 둔다.
```

입력 차이는 의도적으로 하나만 둔다.

```text
row-patch ViT bc16:
  relation pixel을 먼저 만든다.
  pixel(i,j) = [ENC(A_ij), ENC(b_i)]
  b_i를 모든 column j에 broadcast한다.
  row token = [A_i1,b_i, A_i2,b_i, ..., A_i16,b_i]

plain equation Transformer:
  row equation을 바로 token으로 만든다.
  token_i = [ENC(A_i1), ENC(A_i2), ..., ENC(A_i16), ENC(b_i)]
  b_i는 row RHS로 한 번만 붙인다.
```

따라서 plain baseline은 "같은 A,b와 같은 residue encoding을 쓰되, relation pixel broadcast를 제거한 row Transformer"다.
이렇게 해야 row-patch ViT의 `[A_ij,b_i]` pair/broadcast 설계가 실제 이득인지 분리해서 볼 수 있다.

구조:

```text
A,b
-> ENC(A), ENC(b)
-> row token_i = concat(ENC(A_i,:), ENC(b_i))
-> Linear(row_feature_dim -> embed_dim)
-> row positional embedding
-> n개 secret query + secret column positional embedding
-> TransformerEncoder
-> query output
-> coordinate-wise s_logits
```

예상 token 수:

```text
n=16,m=256:
  equation tokens = 256
  secret query tokens = 16
  total = 272
```

이는 `bc16`과 token 수가 같다.
따라서 계산 비용은 비슷하게 맞추고, 입력 표현의 차이만 비교할 수 있다.

1차 비교 조건:

```text
raw_bc16_n16_m256_100k_4096_4096_ep150
raw_equation_n16_m256_100k_4096_4096_ep150
```

2차 비교 조건:

```text
phase6_bc16_n16_m128_100k_4096_4096_ep80
phase6_equation_n16_m128_100k_4096_4096_ep80
```

### 6. CNN Grid Baseline

목적:

```text
Transformer attention 없이도 LWE relation row에서 local/column feature만으로 복원이 되는지 확인한다.
```

입력:

```text
row_pixel(i,j) = [ENC(A_ij), ENC(b_i)]
shape = [B*m, C, n]
```

모델 후보:

```text
Conv1D row-local stack
-> row-wise pooling
-> n개 secret coordinate logits
```

해석:

```text
CNN도 잘 되면:
  relation grid 자체의 feature가 강하고, attention은 필수는 아닐 수 있음.

Transformer만 잘 되면:
  여러 row equation evidence를 query로 모으는 attention 구조가 핵심일 가능성.
```

구현 상태:

```text
구현 완료.
CLI: --model row_cnn
기존 loss/metrics/residual 검증은 그대로 공유.
```

구현 플랜:

```text
목표:
  Transformer attention 없이도 각 LWE row 내부의 CNN feature만으로
  secret support를 복원할 수 있는지 확인한다.

공정성 원칙:
  row_block과 같은 relation row를 그대로 사용한다.
  pixel(i,j) = [ENC(A_ij), ENC(b_i)]
  b_i broadcast 방식도 동일하게 둔다.
  다만 CNN feature extraction은 한 row 안에서만 수행한다.
  서로 다른 LWE row 사이에는 이미지 인접성이 없으므로 2D Conv로 row 이웃을 섞지 않는다.
  raw/centered/phase residue_encoding 옵션도 동일하게 둔다.
  loss/metrics/residual 검증은 train_lwe_vit.py의 기존 로직을 공유한다.
```

입력:

```text
A,b
-> ENC(A), ENC(b)
-> relation pixels X[i,j] = [ENC(A_ij), ENC(b_i)]
-> row tensor shape [B*m, C, n]

raw:
  C=2

centered:
  C=2

phase6:
  C=6
```

모델 구조 후보:

```text
Conv1D row stem:
  [B*m,C,n] -> [B*m,embed_dim,n]

Row-local Conv blocks:
  한 LWE 방정식 row 안에서만 column-direction feature extraction

Column head:
  [B*m,embed_dim,n] -> [B,m,embed_dim,n]
  각 secret coordinate j에 대해 row 방향으로 pooling
  feature_j = mean_m(features[:,:,:,j])
  Linear(feature_j -> num_secret_classes)

Output:
  s_logits shape = [B,n,num_secret_classes]
```

중요한 점:

```text
CNN baseline은 같은 relation row를 보지만,
row equation 전체를 attention으로 aggregation하지 않는다.
따라서 CNN이 실패하고 row_block이 성공하면 query-attention aggregation이 핵심일 가능성이 커진다.
CNN도 성공하면 row-local relation feature와 row pooling만으로도 강한 신호가 있다는 의미가 된다.
```

1차 비교 조건:

```text
raw_bc16_n16_m256_100k_4096_4096_ep150
raw_rowcnn_n16_m256_100k_4096_4096_ep150
```

2차 비교 조건:

```text
phase6_bc16_n16_m128_100k_4096_4096_ep80
phase6_rowcnn_n16_m128_100k_4096_4096_ep80
```

### Baseline 구현 순서

```text
1. residue encoding 공통 함수 정리 [done]
   raw/centered/phase를 row_block, equation_transformer, row_cnn이 같은 방식으로 쓰게 한다.

2. EquationTransformer 추가 [done]
   CLI: --model equation_transformer
   token_i = [ENC(A_i1), ..., ENC(A_in), ENC(b_i)]
   n개 secret query head는 row_block과 동일하게 둔다.

3. RowLocalCNNModel 추가 [done]
   CLI: --model row_cnn
   입력은 relation row [B*m,C,n]
   row-local Conv1d 후 row pooling으로 좌표별 logits를 만든다.

4. train_lwe_vit.py 연결 [done]
   model 선택지만 추가하고,
   run_epoch/loss/metrics/residual code path는 최대한 공유한다.

5. smoke test [done]
   작은 n,m에서 forward shape, loss 계산, residual metric이 동작하는지 확인한다.

6. 동일 조건 비교 실험 [next]
   raw 기준 성공 run과 같은 n,m,q,h,train/val/test,epoch 조건으로 비교한다.
```

### 구현된 baseline 명령 이름

```text
--model equation_transformer
  일반 row-token Transformer.
  token_i = [ENC(A_i1), ..., ENC(A_in), ENC(b_i)]
  sequence length = m + n query.

--model row_cnn
  row-local relation CNN.
  pixel(i,j) = [ENC(A_ij), ENC(b_i)]
  internal shape = [B*m,C,n], raw/centered는 C=2, phase는 C=6.
  Conv1d는 한 row 내부에서만 작동하고, 이후 row 방향 pooling으로 좌표별 s_j logits를 만든다.
```

### 7. bc1/bc4는 비싼 ablation으로 유지

목적:

```text
patch granularity가 작아질 때 성능이 좋아지는지 확인.
원래 아이디어인 pixel-level tokenization에 가까운 방향을 평가.
```

현재 판단:

```text
bc1:
  가장 세밀하지만 attention 비용이 너무 큼.
  raw_bc1_n16_m128_100k_4096_4096_ep50는 epoch당 약 20분으로 비쌈.
  초기 2 epoch에서는 residual 신호가 없었음.

bc4:
  bc16과 bc1 사이의 현실적 중간점.
  나중에 phase6로 먼저 보는 것이 좋음.
```

우선순위:

```text
1. bc16 확장 실험
2. plain Transformer/CNN baseline 구현
3. variable_h/h 증가
4. 그 후 bc4/bc1 ablation
```

## 기준 지표

성공 여부는 다음을 우선으로 본다.

```text
exact_match:
  전체 secret 좌표를 모두 맞춘 비율

support_f1:
  active support 예측 품질

pred_residual_std_mean:
  예측 secret으로 b - A*s_hat residual을 계산했을 때 std
  성공하면 oracle noise std인 약 1 근처
  실패하면 random residual 수준인 약 70대

residual_success_rate:
  residual std가 noise bound 안에 들어온 sample 비율
```

## Loss 정책

대표 실험의 학습 loss는 다음으로 둔다.

```text
loss = weighted cross entropy
```

이유:

```text
목표가 A,b에서 secret 좌표 s_j를 맞추는 supervised classification이므로
좌표별 cross entropy가 가장 직접적이다.

fixed_h=2, n=16에서는 0:1 비율이 14:2라 class imbalance가 크므로
inverse_prior class weight는 유지한다.

residual consistency를 학습 loss에 넣으면 모델 구조 효과와
수학 보조 loss 효과가 섞인다. 따라서 공정 비교용 main result에서는
residual loss를 넣지 않는다.
```

현재 train script 기본값:

```text
--class-weight-mode inverse_prior
--residual-loss-weight 0
```

따라서 기본 objective는:

```text
loss_objective = weighted_ce
```

residual consistency는 계속 계산하고 기록한다.

```text
train_residual_loss
val_residual_loss
test_residual_loss
pred_residual_std_mean
residual_success_rate
```

다만 이는 기본적으로 평가/검증용이다. 보조 loss ablation을 할 때만:

```text
--residual-loss-weight 0.05
```

처럼 명시적으로 켠다. 이런 ablation run은 기본 run name에
`_resw0p05` suffix가 붙는다.

## 주의할 점

현재 `n=16,h=2`는 가능한 support가 작다.

```text
C(16,2)=120
```

따라서 `train=100k`에서는 모든 support 조합을 많이 반복해서 본다.
이 실험은 "작은 fixed-h synthetic regime에서 복원 가능"으로 해석해야 한다.

더 강한 검증은 다음 단계다.

```text
n=32,h=2
n=32,h=4
variable_h
support holdout split
```
