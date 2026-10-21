"""Long-tail-aware metrics and multi-seed aggregation (stdlib only)."""
import statistics


def per_class_recall(y_true, y_pred):
    tot, hit = {}, {}
    for t, p in zip(y_true, y_pred):
        tot[t] = tot.get(t, 0) + 1
        hit[t] = hit.get(t, 0) + (t == p)
    return {c: hit[c] / tot[c] for c in tot}


def balanced_accuracy(y_true, y_pred):
    rec = per_class_recall(y_true, y_pred)
    return sum(rec.values()) / len(rec) if rec else 0.0


def macro_f1(y_true, y_pred):
    classes = set(y_true) | set(y_pred)
    scores = []
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        d = 2 * tp + fp + fn
        scores.append(2 * tp / d if d else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def summarize_seeds(per_seed):
    """per_seed: {seed: {metric: value}} -> {metric: (mean, sample std)}."""
    metrics = {m for v in per_seed.values() for m in v}
    out = {}
    for m in sorted(metrics):
        vals = [v[m] for v in per_seed.values() if m in v]
        out[m] = (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0)
    return out
