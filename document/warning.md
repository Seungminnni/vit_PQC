뉴립스식 실험은 한마디로 **“결과가 좋아 보이는 실험”보다 “다른 사람이 믿고 재현할 수 있는 실험”**을 선호함. NeurIPS 체크리스트도 재현성, 실험 세부정보, 통계적 유의성, 컴퓨팅 자원, 윤리·사회적 영향까지 확인하게 되어 있음. 특히 실험이 있는 논문에서 재현성 항목을 `No`로 답하면 리뷰어에게 좋게 보이지 않는다고 명시되어 있음. 

아래는 네가 LWE/ML 공격 논문이나 QR 피싱 탐지 논문을 낼 때 지켜야 할 실험 원칙임.

---

# 1. 데이터셋은 “무조건 크기만 큰 것”보다 “충분하고 설계가 명확한 것”이 중요함

데이터셋이 크면 좋긴 한데, 뉴립스가 진짜 보는 건 **크기 자체가 아니라 실험 주장을 뒷받침할 만큼 충분한가**임.

나쁜 예:

> 데이터 100만 개로 실험했다.

좋은 예:

> 각 파라미터 설정마다 train/validation/test를 분리했고, 데이터 크기에 따른 성능 변화를 learning curve로 보고했다. 모델 성능이 특정 데이터 크기 이상에서 안정화되는지 확인했다.

즉 “크다”보다 중요한 건 이거임.

| 원칙            | 설명                          |
| ------------- | --------------------------- |
| 충분한 규모        | 결과가 우연히 나온 게 아니게 해야 함       |
| 분포 다양성        | 쉬운 케이스만 넣으면 안 됨             |
| 난이도 변화        | 쉬움/중간/어려움 설정을 나눠야 함         |
| held-out test | 학습 때 본 적 없는 조건에서 평가해야 함     |
| leakage 방지    | train과 test에 같은 정보가 섞이면 안 됨 |

---

# 2. LWE 실험이면 데이터 분리 방식을 특히 조심해야 함

네 LWE 연구에서 제일 위험한 건 **train/test leakage**임.

예를 들어 LWE 샘플이:

[
b = As + e \pmod q
]

형태라고 할 때, 같은 secret vector (s)에서 나온 row들을 train/test에 섞으면 모델이 진짜 일반화한 게 아니라 특정 secret 구조를 외웠을 가능성이 생김.

그래서 추천 방식은 이거임.

## 나쁜 분리

```text
하나의 secret s에서 많은 LWE sample 생성
→ row 단위로 train/test split
```

이러면 같은 (s)가 train과 test에 동시에 들어갈 수 있음.

## 좋은 분리

```text
secret vector 단위로 split
train secrets / validation secrets / test secrets 완전 분리
```

즉 test에는 학습 때 한 번도 보지 않은 secret vector가 들어가야 함.

네가 “전체 비밀키 복구”를 주장하려면 특히 이게 중요함.

---

# 3. 기본 split은 70/15/15 또는 80/10/10이 무난함

보통은 이렇게 감.

| 방식       | 의미                                  |
| -------- | ----------------------------------- |
| 70/15/15 | train 70%, validation 15%, test 15% |
| 80/10/10 | train 80%, validation 10%, test 10% |
| 80/20    | 단순 비교 실험에서는 가능하지만 validation 없으면 약함 |

뉴립스급으로 가려면 validation set을 두는 게 좋음.

왜냐하면 hyperparameter를 test set 보고 고르면 안 되기 때문임.

정석은 이거임.

```text
train set: 모델 학습
validation set: hyperparameter 선택
test set: 최종 성능 보고
```

test set은 마지막에 한 번만 봐야 함.

---

# 4. 시드는 최소 3개, 가능하면 5개, 강하게 하려면 10개

시드 하나로 나온 결과는 약함.

뉴립스 체크리스트도 error bar, confidence interval, statistical significance test 같은 걸 요구하고, error bar가 어떤 변동성을 반영하는지도 설명하라고 함. 예를 들어 train/test split, 초기화, 랜덤 파라미터 생성, 전체 run 변동성을 명시해야 한다고 되어 있음. 

## 추천

| 수준 |     시드 수 | 평가         |
| -- | -------: | ---------- |
| 최소 |  3 seeds | 간신히 가능     |
| 권장 |  5 seeds | 논문용으로 무난   |
| 강함 | 10 seeds | 리뷰어 설득력 높음 |

네 경우는 우선 **5 seeds**를 추천함.

예:

```text
Seeds = {0, 1, 2, 3, 4}
```

그리고 결과는 이렇게 써야 함.

```text
Coordinate accuracy: 0.612 ± 0.018
Full recovery rate: 0.084 ± 0.011
```

여기서 반드시 밝혀야 함.

```text
The error bars indicate standard deviation over five random seeds.
```

또는

```text
We report mean ± standard error over five independent runs.
```

표준편차인지 표준오차인지 헷갈리게 쓰면 안 됨.

---

# 5. 시드는 “모델 초기화”만 고정하는 게 아님

시드가 영향을 주는 곳은 많음.

네 실험에서는 최소 아래를 모두 통제해야 함.

| 랜덤 요소                      | 설명          |
| -------------------------- | ----------- |
| secret 생성                  | (s) 샘플링     |
| matrix 생성                  | (A) 샘플링     |
| noise 생성                   | (e) 샘플링     |
| train/val/test split       | 데이터 분리      |
| model initialization       | 모델 가중치 초기화  |
| batch shuffle              | 학습 중 데이터 순서 |
| negative/positive sampling | 있다면 샘플링 방식  |

논문에는 이렇게 적으면 좋음.

```text
For each seed, we independently regenerate the LWE instances, train/validation/test split, model initialization, and minibatch order.
```

이렇게 해야 결과가 진짜 안정적인지 볼 수 있음.

---

# 6. 데이터 크기 하나만 보여주지 말고 scaling curve를 보여주는 게 좋음

뉴립스 리뷰어는 보통 이렇게 물음.

> 데이터가 커지면 성능이 계속 좋아지는가?
> 작은 데이터에서만 되는가?
> 차원이 커지면 무너지는가?
> noise가 커지면 어떻게 되는가?

그래서 하나의 setting만 보여주면 약함.

LWE 실험이면 최소 이런 축을 바꿔야 함.

| 축              | 예시                                        |
| -------------- | ----------------------------------------- |
| 차원 (n)         | 32, 64, 128                               |
| modulus (q)    | 257, 512, 769                             |
| noise (\sigma) | 1, 2, 3, 4                                |
| sample 수 (m)   | 1k, 5k, 10k, 50k                          |
| secret 범위      | binary, ternary, bounded integer          |
| secret 분포      | uniform / sparse / centered small integer |

네가 지금 LWE 공격 쪽이면 최소 이 정도는 추천함.

```text
n ∈ {32, 64, 128}
q ∈ {257, 512}
σ ∈ {1, 2, 3}
secret ∈ {-3, -2, -1, 0, 1, 2, 3}
seeds = {0, 1, 2, 3, 4}
```

처음부터 너무 크게 하면 실험이 터지니까, 본문에는 핵심 setting을 넣고 appendix에 확장 실험을 넣으면 됨.

---

# 7. baseline은 약한 것만 고르면 안 됨

뉴립스에서 제일 싫어하는 실험 중 하나가 **허수아비 baseline**임.

즉 일부러 약한 비교대상을 세워놓고 “우리 방법이 좋다”고 하면 바로 까임.

좋은 baseline은 최소 3종류가 필요함.

## LWE/ML 공격 논문 기준

| baseline             | 의미                                     |
| -------------------- | -------------------------------------- |
| Random guess         | 클래스 수 기준 최소 성능                         |
| Linear model         | logistic regression, linear classifier |
| MLP                  | 기본 neural baseline                     |
| Transformer/SALSA 계열 | 관련 연구 기준                               |
| Nearest centroid     | 네가 말한 임베딩 평균 기반 방식과 비교 가능              |
| Classical solver     | least squares, lattice/BKZ 가능한 범위면 비교  |

네 아이디어가 “자리별 임베딩 + 클래스 평균”이면 적어도 아래와 비교해야 함.

```text
Random
Linear classifier
MLP
Nearest-centroid without proposed feature
Nearest-centroid with proposed feature
SALSA-style model, if applicable
```

그래야 “그냥 모델 크게 해서 좋아진 거 아님?”이라는 공격을 막을 수 있음.

---

# 8. ablation study는 거의 필수라고 보면 됨

Ablation은 네 방법에서 구성요소 하나씩 빼보는 실험임.

리뷰어가 궁금해하는 건 이거임.

> 성능 향상이 정확히 어떤 요소 때문에 나온 건가?

예를 들어 네 LWE 아이디어가 다음으로 구성되어 있다고 하자.

1. coordinate-wise feature (z_j)
2. class prototype (\mu_c)
3. nearest-centroid prediction
4. full secret recovery
5. hamming 제거 또는 정수 기반 평가

그러면 ablation은 이렇게 설계함.

| 실험                                    | 목적                |
| ------------------------------------- | ----------------- |
| full model                            | 전체 방법             |
| without prototype                     | prototype 효과 확인   |
| without coordinate-wise feature       | 자리별 feature 효과 확인 |
| linear classifier instead of centroid | centroid가 필요한지 확인 |
| digit-wise only vs full-vector        | 전체 복구 전략의 효과 확인   |
| different embedding dimension         | 임베딩 차원 민감도 확인     |

Ablation이 없으면 리뷰어가 “뭐가 contribution인지 모르겠다”고 할 가능성이 큼.

---

# 9. metric은 하나만 쓰면 약함

LWE secret recovery 논문에서 accuracy 하나만 쓰면 부족함.

추천 metric은 이거임.

| Metric                       | 의미                          |
| ---------------------------- | --------------------------- |
| coordinate accuracy          | 각 좌표 (s_j)를 맞춘 비율           |
| full secret recovery rate    | 전체 secret vector를 완전히 맞춘 비율 |
| macro-F1                     | 클래스 불균형 있을 때 중요             |
| per-class accuracy           | -3, -2, ..., 3 각각 얼마나 맞추는지  |
| top-k accuracy               | 후보군 안에 정답이 있는지              |
| runtime                      | 공격 시간                       |
| memory                       | 실험 비용                       |
| success rate by (n,q,\sigma) | 파라미터별 성공률                   |

특히 네가 “전체 비밀키 복구”를 주장하려면 **full secret recovery rate**를 반드시 넣어야 함.

좌표 정확도가 90%여도 (n=128)이면 전체 secret을 다 맞출 확률은 낮을 수 있음.

예를 들어 좌표별 정확도가 0.95라도 독립이라고 단순 가정하면:

[
0.95^{128} \approx 0.0014
]

즉 좌표 정확도만 높다고 전체 복구가 된다고 주장하면 안 됨.

---

# 10. 결과는 평균만 쓰지 말고 error bar를 붙여야 함

뉴립스 체크리스트는 error bar나 confidence interval 또는 statistical significance test를 요구하고, error bar 계산 방식까지 설명하라고 함. 또한 error bar가 standard deviation인지 standard error인지 명확히 해야 한다고 되어 있음. ([NeurIPS][1])

좋은 표 예시는 이거임.

| Method |       Coord. Acc. |     Full Recovery |          Macro-F1 |
| ------ | ----------------: | ----------------: | ----------------: |
| Linear |     0.531 ± 0.012 |     0.000 ± 0.000 |     0.501 ± 0.014 |
| MLP    |     0.574 ± 0.018 |     0.004 ± 0.002 |     0.552 ± 0.016 |
| Ours   | **0.621 ± 0.015** | **0.037 ± 0.009** | **0.604 ± 0.013** |

그리고 표 아래에 이렇게 써야 함.

```text
All results are averaged over five independent seeds. Error bars denote one standard deviation.
```

---

# 11. hyperparameter tuning은 공정해야 함

리뷰어가 자주 보는 부분임.

나쁜 방식:

```text
우리 방법만 열심히 튜닝하고 baseline은 기본값 사용
```

좋은 방식:

```text
모든 방법에 동일한 validation protocol을 적용
각 모델은 같은 search budget 안에서 hyperparameter 선택
```

예:

| 항목                | 공정한 방식                         |
| ----------------- | ------------------------------ |
| learning rate     | 모든 모델에서 {1e-4, 3e-4, 1e-3} 탐색  |
| batch size        | {64, 128, 256}                 |
| hidden dim        | {128, 256, 512}                |
| epoch             | 동일한 maximum epoch              |
| early stopping    | 동일한 patience                   |
| validation metric | 동일하게 macro-F1 또는 full recovery |

논문에는 이렇게 쓰면 좋음.

```text
Hyperparameters for all methods were selected using the validation set under the same search budget.
```

---

# 12. test set으로 모델 고르면 안 됨

이거 진짜 중요함.

test 성능 보고 설정 바꾸면 안 됨.

실험 순서는 이렇게 해야 함.

```text
1. train set으로 학습
2. validation set으로 hyperparameter 선택
3. test set은 마지막 최종 평가에만 사용
```

만약 test를 여러 번 보고 계속 고치면, test set에 과적합된 결과가 됨.

논문에 이런 문장 넣으면 좋음.

```text
The test set was used only once for final evaluation after hyperparameter selection on the validation set.
```

---

# 13. compute resource를 반드시 써야 함

NeurIPS 체크리스트는 각 실험에 대해 CPU/GPU 종류, 메모리, 실행 시간, 전체 compute를 충분히 제공하라고 함. 또 실제 논문에 보고된 실험보다 preliminary/failed experiments에 더 많은 compute가 들었는지도 공개하라고 되어 있음. 

예시:

```text
All experiments were run on a workstation with an Intel i7 CPU, 64GB RAM, and a single NVIDIA RTX 4090 GPU with 24GB VRAM. Each training run took approximately 2.1 hours for n=64 and 4.8 hours for n=128. The total compute for the reported experiments was approximately 180 GPU-hours.
```

이걸 쓰면 리뷰어가 “재현 가능한 규모인가?”를 판단할 수 있음.

---

# 14. 코드와 데이터는 가능하면 공개, 못 하면 재현 경로라도 줘야 함

NeurIPS는 코드 공개를 무조건 강제하지는 않지만, 실험 결과가 핵심 contribution이면 그 결과를 재현할 수 있는 코드 제공이 best practice라고 설명함. 제출용 code/data는 익명화된 zip으로 포함할 수 있고, 작은 데이터셋은 100MB 미만 zip에 넣을 수 있으며 큰 데이터셋은 anonymous URL로 연결할 수 있음. ([NeurIPS][2])

즉 가장 좋은 방식은:

```text
anonymous GitHub or anonymous zip
+ environment.yml
+ requirements.txt
+ run script
+ data generation script
+ exact commands
```

예:

```bash
python generate_lwe.py --n 64 --q 257 --sigma 3 --m 50000 --seed 0
python train.py --config configs/lwe_n64_q257_sigma3.yaml --seed 0
python evaluate.py --checkpoint runs/seed0/best.pt
```

이런 식으로 명령어까지 주면 좋음.

---

# 15. 실험 setting은 본문에 충분히, 세부는 appendix에

NeurIPS 템플릿은 핵심 논문이 9페이지 제한이고, acknowledgments, references, checklist, optional technical appendices는 content page에 포함되지 않는다고 설명함. 다만 appendix는 optional reading이어야 하며, 핵심 주장을 뒷받침하는 중요한 실험을 appendix에만 넣는 것은 부적절하다고 되어 있음.  

따라서 구조는 이렇게 가야 함.

## 본문에 반드시 넣을 것

* 핵심 실험 setting
* 주요 결과 표
* 핵심 baseline 비교
* 주요 ablation
* 한계 요약

## appendix에 넣을 것

* 전체 hyperparameter 표
* 추가 seed 결과
* 추가 파라미터 sweep
* 실패 실험
* proof 또는 세부 알고리즘
* 전체 per-class confusion matrix

---

# 16. claims는 실험 범위를 넘으면 안 됨

NeurIPS 체크리스트는 abstract/introduction의 주장이 실제 contribution과 scope를 정확히 반영해야 하고, 실험 결과와 일반화 가능 범위에 맞아야 한다고 요구함. 

네 LWE 논문에서 절대 이렇게 쓰면 안 됨.

```text
We break LWE.
```

이건 너무 큼.

대신 이렇게 써야 함.

```text
We study learning-based secret recovery under bounded-integer secret distributions in controlled synthetic LWE settings.
```

또는

```text
Our results show improved recovery accuracy for small-dimensional bounded-secret LWE instances under specified noise and modulus regimes.
```

이렇게 써야 안전함.

---

# 17. limitation은 숨기지 말고 먼저 써야 함

뉴립스 체크리스트는 limitation section을 따로 만드는 것을 권장하고, 강한 가정, 실제 적용 가능성, 데이터셋/실행 횟수의 제한, 계산 효율성, privacy/fairness 한계 등을 논의하라고 함. 또한 한계를 솔직히 쓴다고 불이익을 주지 않도록 리뷰어에게 안내한다고 되어 있음. 

네 LWE 논문 limitation 예시:

```text
Our experiments are limited to synthetic LWE instances with bounded-integer secrets and moderate dimensions. We do not claim practical attacks against standardized PQC schemes. The scalability of the method to larger dimensions, larger modulus values, and implementation-level side-channel settings remains future work.
```

이렇게 써야 함.

---

# 18. QR 피싱 논문이면 데이터 윤리와 안전장치가 더 중요함

QR 피싱 탐지 논문은 실제 피싱 URL, HTML, credential form, 동적 분석이 들어갈 수 있음.

그럼 아래를 지켜야 함.

| 항목              | 원칙                                  |
| --------------- | ----------------------------------- |
| 실제 피싱 URL       | active URL 그대로 공개하지 않기              |
| credential form | 실제 개인정보 입력 금지                       |
| 동적 분석           | 샌드박스/격리 환경에서 실행                     |
| 패킷 캡처           | 민감정보 저장 금지                          |
| 데이터 공개          | raw phishing page 대신 feature만 공개 고려 |
| LLM teacher     | 사용했다면 LLM 사용 선언 필요                  |
| human study     | 사람에게 QR 스캔시키면 IRB 검토 필요             |

NeurIPS 체크리스트도 high-risk data/model release에는 safeguards를 설명하라고 하고, malicious/unintended use, privacy, security 같은 negative societal impact를 논의하라고 함. 

---

# 19. 뉴립스가 좋아하는 실험 구성 한 세트

네 논문 실험 섹션은 이렇게 만들면 강함.

```text
4. Experiments

4.1 Experimental setup
- dataset generation
- train/val/test split
- parameter settings
- model architecture
- training details
- hardware

4.2 Baselines
- random
- classical baseline
- ML baseline
- prior work baseline

4.3 Main results
- main performance table
- mean ± std over 5 seeds

4.4 Scaling analysis
- n, q, sigma, m 변화

4.5 Ablation study
- component removal

4.6 Robustness analysis
- OOD parameter
- noise sensitivity
- distribution shift

4.7 Compute and efficiency
- runtime
- memory
- GPU-hours
```

이 구조면 리뷰어가 좋아하는 질문에 대부분 답이 됨.

---

# 20. 네 LWE 실험 기준 추천 세팅

현실적으로 처음 논문 실험을 짠다면 이렇게 추천함.

## Main setting

```text
n = 32, 64
q = 257, 512
sigma = 1, 2, 3
s_i ∈ {-3, -2, -1, 0, 1, 2, 3}
m = 10k, 50k, 100k
seeds = 0, 1, 2, 3, 4
split = secret-vector-level 70/15/15
```

## Metrics

```text
coordinate accuracy
full secret recovery rate
macro-F1
per-class accuracy
runtime
memory
```

## Baselines

```text
random guess
least-squares rounding
linear classifier
MLP
nearest-centroid baseline
SALSA-style baseline, if feasible
```

## Tables

본문에는 최소 3개.

```text
Table 1: main comparison
Table 2: scaling with n/q/sigma
Table 3: ablation study
```

appendix에는 추가.

```text
per-class confusion matrix
all seed results
hyperparameter table
extra parameter sweeps
```

---

# 21. 결과 표기 예시

논문에 이렇게 쓰면 좋음.

```text
We report mean and standard deviation over five independent random seeds. Each seed regenerates the LWE instances, train/validation/test split, model initialization, and minibatch ordering. Hyperparameters are selected on the validation set and the test set is used only for final evaluation.
```

이 문장 하나가 엄청 중요함.

---

# 22. 실험할 때 절대 하면 안 되는 것

| 하지 말 것                                     | 이유              |
| ------------------------------------------ | --------------- |
| seed 하나만 보고 결론 내기                          | 우연일 수 있음        |
| test set 보고 hyperparameter 고르기             | test leakage    |
| baseline 약하게 설정하기                          | 불공정 비교          |
| 좋은 결과만 선택해서 보고                             | cherry-picking  |
| 실패한 조건 숨기기                                 | limitation 공격당함 |
| 데이터 생성 방식 생략                               | 재현 불가           |
| 코드 버전/환경 생략                                | 재현 불가           |
| 전체 secret 복구 주장하면서 coordinate accuracy만 보고 | 주장과 metric 불일치  |
| 작은 (n)만 해놓고 일반 LWE 공격처럼 주장                 | 과장              |
| appendix에 핵심 결과만 넣기                        | 본문 독립성 부족       |

---

# 23. 최종 체크리스트

실험 시작 전에 이걸 체크하면 됨.

```text
[ ] train/validation/test가 명확히 분리되어 있는가?
[ ] LWE에서는 secret vector 단위로 분리했는가?
[ ] 최소 5개 seed로 반복하는가?
[ ] mean ± std 또는 confidence interval을 보고하는가?
[ ] error bar가 무엇을 의미하는지 설명했는가?
[ ] baseline이 충분히 강하고 공정한가?
[ ] ablation study가 있는가?
[ ] n, q, sigma, m에 대한 scaling 실험이 있는가?
[ ] hyperparameter search를 validation set에서만 했는가?
[ ] test set은 최종 평가에만 사용했는가?
[ ] 데이터 생성 방식과 코드 실행 명령어를 제공하는가?
[ ] compute resource와 총 GPU-hour를 보고하는가?
[ ] limitation section을 따로 두었는가?
[ ] 공격/보안 논문이면 misuse와 safeguards를 썼는가?
```

---

한 줄로 정리하면, **뉴립스가 선호하는 실험은 큰 데이터 하나로 성능 자랑하는 실험이 아니라, 여러 seed·강한 baseline·공정한 split·error bar·ablation·scaling·재현 가능한 코드/설정·정직한 limitation이 모두 갖춰진 실험**임. 네 LWE 논문은 특히 **secret-vector-level split, 5 seeds, full secret recovery rate, n/q/sigma scaling, SALSA류 baseline, limitation 명확화**가 핵심임.

[1]: https://neurips.cc/public/guides/PaperChecklist?utm_source=chatgpt.com "NeurIPS Paper Checklist Guidelines"
[2]: https://neurips.cc/public/guides/CodeSubmissionPolicy?utm_source=chatgpt.com "Paper Information / Code Submission Policy"
