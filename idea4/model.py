import math
import torch
import torch.nn as nn

COLUMN_IMAGE_CHANNELS = 20


class LWEColumnViT(nn.Module):
    # Baseline ColumnViT: static image classifier on a single [A | b] snapshot.
    def __init__(self, M=1024, n=256, embed_dim=256, num_heads=8, depth=4):
        super().__init__()
        self.M = M
        self.n = n
        
        # [핵심] 가로 1, 세로 1024짜리 길쭉한 칼(Conv2d)로 이미지를 열(Column) 단위로 썹니다.
        # 즉, 1개의 열 데이터 전체가 1개의 트랜스포머 토큰(단어)으로 변환됩니다.
        self.patch_embed = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=(M, 1))
        
        # 위치 임베딩: 모델에게 "이 토큰이 몇 번째 열(비밀키 인덱스)이야" 라고 알려줍니다.
        self.pos_embed = nn.Parameter(torch.randn(1, n + 1, embed_dim))
        
        # 트랜스포머 인코더: 열들 간의 상관관계(특히 마지막 b' 열과의 관계)를 학습합니다.
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 최종 분류기: 각 열에 대해 "이 열의 비밀키가 1일 확률은?" 을 계산(0~1 사이 점수)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # 입력 x: (Batch, 3, 1024, 257)
        x = self.patch_embed(x)          # -> (Batch, 256, 1, 257)
        x = x.squeeze(2).transpose(1, 2) # -> (Batch, 257, 256)
        
        x = x + self.pos_embed           # 위치 정보 더하기
        x = self.transformer(x)          # 트랜스포머 연산 -> (Batch, 257, 256)
        
        # 마지막 257번째 열(b' 벡터 영역)은 힌트이므로 정답 예측에서는 빼고, 앞의 256개만 가져옵니다.
        x = x[:, :self.n, :]             # -> (Batch, 256, 256)
        
        logits = self.classifier(x).squeeze(-1) # -> (Batch, 256)
        return logits


def centered_mod(x, q):
    q_t = torch.tensor(float(q), device=x.device, dtype=x.dtype)
    half_q = q_t / 2.0
    return torch.remainder(x + half_q, q_t) - half_q


def circular_loss(residual, q):
    q_t = torch.tensor(float(q), device=residual.device, dtype=residual.dtype)
    return (1.0 - torch.cos(2.0 * math.pi * residual / q_t)).mean()


def make_column_image(A, b, p, q):
    # A: (B, M, n), b: (B, M), p: (B, n)
    b_hat = torch.bmm(A, p.unsqueeze(2)).squeeze(2)
    b_minus_bhat = b - b_hat
    residual = centered_mod(b_minus_bhat, q)

    q_t = torch.tensor(float(q), device=A.device, dtype=A.dtype)
    half_q = q_t / 2.0
    A_center = centered_mod(A, q) / half_q
    sin_A = torch.sin(2.0 * math.pi * A / q_t)
    cos_A = torch.cos(2.0 * math.pi * A / q_t)
    b_mod = centered_mod(b, q)
    b_img = (b_mod / half_q).unsqueeze(2).expand(-1, -1, A.shape[2])
    sin_b = torch.sin(2.0 * math.pi * b / q_t).unsqueeze(2).expand(-1, -1, A.shape[2])
    cos_b = torch.cos(2.0 * math.pi * b / q_t).unsqueeze(2).expand(-1, -1, A.shape[2])
    r_img = (residual / half_q).unsqueeze(2).expand(-1, -1, A.shape[2])
    sin_r = torch.sin(2.0 * math.pi * residual / q_t).unsqueeze(2).expand(-1, -1, A.shape[2])
    cos_r = torch.cos(2.0 * math.pi * residual / q_t).unsqueeze(2).expand(-1, -1, A.shape[2])
    p_img = p.unsqueeze(1).expand(-1, A.shape[1], -1)
    corr_img = A_center * r_img
    contrib_img = A_center * p_img

    single_residual = centered_mod(b.unsqueeze(2) - A, q)
    single_mean_abs = (single_residual.abs().mean(dim=1) / half_q).unsqueeze(1).expand(-1, A.shape[1], -1)
    single_circ = (
        1.0 - torch.cos(2.0 * math.pi * single_residual / q_t)
    ).mean(dim=1).unsqueeze(1).expand(-1, A.shape[1], -1)

    on_residual = centered_mod(b_minus_bhat.unsqueeze(2) - A * (1.0 - p_img), q)
    off_residual = centered_mod(b_minus_bhat.unsqueeze(2) + A * p_img, q)
    current_circ = (1.0 - torch.cos(2.0 * math.pi * residual / q_t)).mean(dim=1, keepdim=True)
    on_circ = (1.0 - torch.cos(2.0 * math.pi * on_residual / q_t)).mean(dim=1)
    off_circ = (1.0 - torch.cos(2.0 * math.pi * off_residual / q_t)).mean(dim=1)
    current_circ_img = current_circ.unsqueeze(2).expand(-1, A.shape[1], A.shape[2])
    on_delta_img = (current_circ - on_circ).unsqueeze(1).expand(-1, A.shape[1], -1)
    off_delta_img = (current_circ - off_circ).unsqueeze(1).expand(-1, A.shape[1], -1)
    on_residual_img = on_residual / half_q
    off_residual_img = off_residual / half_q

    X = torch.stack(
        [
            A_center,
            sin_A,
            cos_A,
            b_img,
            sin_b,
            cos_b,
            r_img,
            sin_r,
            cos_r,
            p_img,
            corr_img,
            contrib_img,
            single_mean_abs,
            single_circ,
            current_circ_img,
            on_delta_img,
            off_delta_img,
            on_residual_img,
            off_residual_img,
            on_residual_img - off_residual_img,
        ],
        dim=1,
    )
    return X, residual


def compute_pair_residual_features(A, b, pair_indices, q):
    # A: (B, M, n), b: (B, M), pair_indices: (P, 2)
    idx0 = pair_indices[:, 0]
    idx1 = pair_indices[:, 1]
    b_hat = (A[:, :, idx0] + A[:, :, idx1]) % q
    residual = centered_mod(b.unsqueeze(-1) - b_hat, q)

    q_t = torch.tensor(float(q), device=A.device, dtype=A.dtype)
    half_q = q_t / 2.0

    abs_residual = residual.abs()
    mean_abs = abs_residual.mean(dim=1) / half_q
    mean_sq = (residual ** 2).mean(dim=1) / (half_q ** 2)
    max_abs = abs_residual.amax(dim=1) / half_q
    std_abs = abs_residual.std(dim=1, unbiased=False) / half_q
    circ = (1.0 - torch.cos(2.0 * math.pi * residual / q_t)).mean(dim=1)

    residual_features = torch.stack([mean_abs, mean_sq, max_abs, std_abs, circ], dim=-1)
    residual_score = -mean_abs
    return residual_features, residual_score


class ResidualColumnViTCore(nn.Module):
    def __init__(self, M=64, n=16, embed_dim=128, num_heads=4, depth=2):
        super().__init__()
        self.M = M
        self.n = n

        self.patch_embed = nn.Conv2d(in_channels=COLUMN_IMAGE_CHANNELS, out_channels=embed_dim, kernel_size=(M, 1))
        self.pos_embed = nn.Parameter(torch.randn(1, n, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.squeeze(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x)
        return self.head(x).squeeze(-1)


class ResidualColumnEncoder(nn.Module):
    def __init__(self, M=64, n=16, embed_dim=128, num_heads=4, depth=2, dropout=0.1, use_pos_embed=True):
        super().__init__()
        self.M = M
        self.n = n
        self.use_pos_embed = use_pos_embed

        self.patch_embed = nn.Conv2d(in_channels=COLUMN_IMAGE_CHANNELS, out_channels=embed_dim, kernel_size=(M, 1))
        self.pos_embed = nn.Parameter(torch.randn(1, n, embed_dim))
        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.squeeze(2).transpose(1, 2)
        if self.use_pos_embed:
            x = x + self.pos_embed
        x = self.embed_dropout(x)
        x = self.transformer(x)
        return self.norm(x)


class PairResidualColumnViT(nn.Module):
    # Pairwise residual-guided solver for h=2 secrets.
    def __init__(
        self,
        M=32,
        n=8,
        q=127,
        T=3,
        embed_dim=128,
        num_heads=4,
        depth=2,
        dropout=0.1,
        pair_feature_mode="symmetric",
        residual_score_weight=0.0,
        use_pos_embed=True,
        h_prior=2,
    ):
        super().__init__()
        self.M = M
        self.n = n
        self.q = q
        self.T = T
        self.h_prior = h_prior
        self.pair_feature_mode = pair_feature_mode
        self.residual_score_weight = residual_score_weight
        self.encoder = ResidualColumnEncoder(
            M=M,
            n=n,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            dropout=dropout,
            use_pos_embed=use_pos_embed,
        )
        self.post_embed_dropout = nn.Dropout(dropout)

        if pair_feature_mode not in ("ordered", "symmetric"):
            raise ValueError(f"pair_feature_mode must be 'ordered' or 'symmetric', got {pair_feature_mode}")

        embed_mul = 4 if pair_feature_mode == "ordered" else 3
        pair_input_dim = embed_mul * embed_dim + 5

        self.pair_head = nn.Sequential(
            nn.Linear(pair_input_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1),
        )

        pair_indices = torch.combinations(torch.arange(n), r=2)
        pair_to_marginal = torch.zeros(pair_indices.size(0), n, dtype=torch.float32)
        pair_to_marginal.scatter_(1, pair_indices, 1.0)
        self.register_buffer("pair_indices", pair_indices)
        self.register_buffer("pair_to_marginal", pair_to_marginal)

    def forward(self, A, b):
        if b.dim() == 3 and b.size(-1) == 1:
            b = b.squeeze(-1)

        prior_p = float(self.h_prior) / float(self.n)
        prior_p = min(max(prior_p, 1e-4), 1.0 - 1e-4)
        p = torch.full((A.size(0), self.n), prior_p, device=A.device, dtype=A.dtype)

        residual_features, residual_score = compute_pair_residual_features(
            A, b, self.pair_indices, self.q
        )

        pair_scores = None
        for _ in range(self.T):
            X, _ = make_column_image(A, b, p, self.q)
            H = self.encoder(X)
            H = self.post_embed_dropout(H)
            idx0 = self.pair_indices[:, 0]
            idx1 = self.pair_indices[:, 1]
            H0 = H[:, idx0, :]
            H1 = H[:, idx1, :]
            if self.pair_feature_mode == "ordered":
                pair_embed = torch.cat([H0, H1, H0 * H1, (H0 - H1).abs()], dim=-1)
            else:
                pair_embed = torch.cat([H0 + H1, H0 * H1, (H0 - H1).abs()], dim=-1)

            pair_feat = torch.cat([pair_embed, residual_features], dim=-1)
            neural_score = self.pair_head(pair_feat).squeeze(-1)
            pair_scores = neural_score + self.residual_score_weight * residual_score
            pair_probs = torch.softmax(pair_scores, dim=-1)
            p = torch.matmul(pair_probs, self.pair_to_marginal)

        return pair_scores, self.pair_indices, p


class RecurrentResidualColumnViT(nn.Module):
    # Residual-Feedback ColumnViT: iterative residual-guided solver.
    def __init__(
        self,
        M=64,
        n=16,
        q=127,
        T=6,
        embed_dim=128,
        num_heads=4,
        depth=2,
        step_size=0.5,
        h_prior=None,
    ):
        super().__init__()
        self.M = M
        self.n = n
        self.q = q
        self.T = T
        self.step_size = step_size
        self.h_prior = h_prior
        self.core = ResidualColumnViTCore(
            M=M,
            n=n,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
        )

    def _init_logits(self, batch_size, device, dtype):
        if self.h_prior is None:
            prior_p = 0.5
        else:
            prior_p = float(self.h_prior) / float(self.n)
        prior_p = min(max(prior_p, 1e-4), 1.0 - 1e-4)
        prior_logit = math.log(prior_p / (1.0 - prior_p))
        return torch.full((batch_size, self.n), prior_logit, device=device, dtype=dtype)

    def forward(self, A, b, return_residuals=True):
        if b.dim() == 3 and b.size(-1) == 1:
            b = b.squeeze(-1)

        z = self._init_logits(A.size(0), A.device, A.dtype)
        residuals = []
        for _ in range(self.T):
            p = torch.sigmoid(z)
            X, residual = make_column_image(A, b, p, self.q)
            delta_z = self.core(X)
            z = z + self.step_size * torch.tanh(delta_z)
            residuals.append(residual)

        if return_residuals:
            return z, residuals
        return z
