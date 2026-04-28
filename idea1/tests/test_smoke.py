from __future__ import annotations

import torch

from src.attack.recover import recover_binary_from_topk
from src.embeddings.packet import build_crypto_image_packet
from src.lwe.generator import generate_lwe_batch, generator_sanity_check
from src.models.cnn import SimpleLWEImageCNN
from src.training.losses import support_bce_loss
from src.training.metrics import exact_support_recovery, top_h_recall


def test_generator_embedding_model_recovery_smoke():
    device = "cpu"
    n, m, q, h = 16, 64, 257, 3
    sanity = generator_sanity_check(4, n, m, q, h, 1, device)
    assert sanity["passed"] == 1.0
    A, b, s, _ = generate_lwe_batch(8, n, m, q, h, 1, device)
    packet = build_crypto_image_packet(A, b, q, use_raw=True, use_phase=True, use_rhie=True, use_gram=True)
    assert packet["raw"].shape == (8, 1, m, n + 1)
    assert packet["phase"].shape == (8, 6, m, n + 1)
    assert packet["rhie"].shape == (8, 8, m, n)
    assert packet["gram"].shape == (8, 2, n, n)
    model = SimpleLWEImageCNN(in_channels=8, hidden_dim=32, output_n=n)
    logits = model(packet["rhie"])
    y = (s != 0).float()
    loss = support_bce_loss(logits, y, n=n, h=h)
    assert torch.isfinite(loss)
    assert 0.0 <= top_h_recall(logits, y, h) <= 1.0
    assert 0.0 <= exact_support_recovery(logits, y, h) <= 1.0
    probs = y[0] * 0.9 + 0.05
    support, score = recover_binary_from_topk(A[0], b[0], probs, h=h, k=h, q=q)
    assert set(support) == set(torch.where(s[0] != 0)[0].tolist())
    assert score <= 1.0

