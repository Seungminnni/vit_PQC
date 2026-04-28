import itertools
import random

import torch


def all_binary_supports(n, h):
    return [tuple(int(x) for x in support) for support in itertools.combinations(range(n), h)]


def split_binary_supports(n, h, train_fraction=0.8, seed=1234):
    supports = all_binary_supports(n, h)
    if len(supports) < 2:
        raise ValueError(f"Need at least 2 supports for secret_split, got {len(supports)}")

    rng = random.Random(seed)
    rng.shuffle(supports)
    cut = max(1, min(len(supports) - 1, int(round(len(supports) * train_fraction))))
    return supports[:cut], supports[cut:]


def _split_sizes(total, train_fraction, val_fraction):
    train_size = int(round(total * train_fraction))
    val_size = int(round(total * val_fraction))
    train_size = max(1, min(total - 2, train_size))
    val_size = max(1, min(total - train_size - 1, val_size))
    test_size = total - train_size - val_size
    if test_size < 1:
        val_size -= 1
        test_size = 1
    return train_size, val_size, test_size


def _coordinate_counts(pool, n):
    counts = [0] * n
    for support in pool:
        for idx in support:
            counts[idx] += 1
    return counts


def _imbalance_score(pools, sizes, n, h):
    score = 0.0
    for pool, size in zip(pools, sizes):
        target = size * h / n
        counts = _coordinate_counts(pool, n)
        score += sum((count - target) ** 2 for count in counts)
        if target >= 1.0:
            score += 10.0 * sum(1 for count in counts if count == 0)
    return score


def split_binary_supports_balanced(
    n,
    h,
    train_fraction=0.8,
    val_fraction=0.1,
    seed=1234,
    trials=128,
):
    supports = all_binary_supports(n, h)
    if len(supports) < 3:
        raise ValueError(f"Need at least 3 supports for train/val/test split, got {len(supports)}")

    sizes = _split_sizes(len(supports), train_fraction, val_fraction)
    best_pools = None
    best_score = None

    for trial in range(max(1, trials)):
        rng = random.Random(seed + trial)
        remaining = supports[:]
        rng.shuffle(remaining)
        pools = [[], [], []]
        counts = [[0] * n for _ in range(3)]
        targets = [size * h / n for size in sizes]

        for split_idx in (1, 2, 0):
            size = sizes[split_idx]
            while len(pools[split_idx]) < size:
                candidates = []
                for support_idx, support in enumerate(remaining):
                    new_counts = counts[split_idx][:]
                    for coord in support:
                        new_counts[coord] += 1
                    split_score = sum((count - targets[split_idx]) ** 2 for count in new_counts)
                    if targets[split_idx] >= 1.0:
                        split_score += 10.0 * sum(1 for count in new_counts if count == 0)
                    jitter = rng.random() * 1e-6
                    candidates.append((split_score + jitter, support_idx, support, new_counts))

                _, support_idx, support, new_counts = min(candidates, key=lambda item: item[0])
                pools[split_idx].append(support)
                counts[split_idx] = new_counts
                remaining.pop(support_idx)

        score = _imbalance_score(pools, sizes, n, h)
        if best_score is None or score < best_score:
            best_score = score
            best_pools = pools

    return tuple(best_pools)


def support_split_summary(pools, n):
    summary = {}
    names = ["train", "val", "test"]
    for name, pool in zip(names, pools):
        counts = _coordinate_counts(pool, n)
        summary[f"{name}_supports"] = len(pool)
        summary[f"{name}_coord_min"] = min(counts)
        summary[f"{name}_coord_max"] = max(counts)
        summary[f"{name}_coord_counts"] = counts
    summary["train_val_overlap"] = support_overlap_count(pools[0], pools[1])
    summary["train_test_overlap"] = support_overlap_count(pools[0], pools[2])
    summary["val_test_overlap"] = support_overlap_count(pools[1], pools[2])
    return summary


def sample_fixed_h_from_pool(num_samples, n, support_pool, device=None):
    if not support_pool:
        raise ValueError("support_pool is empty")

    choices = torch.randint(0, len(support_pool), (num_samples,), device=device)
    s = torch.zeros(num_samples, n, device=device)
    for row, pool_idx in enumerate(choices.tolist()):
        s[row, list(support_pool[pool_idx])] = 1.0
    return s


def support_overlap_count(left, right):
    return len(set(left) & set(right))
