"""Lesion-grouped splits: all images of a lesion land in exactly one partition."""
import random
from collections import defaultdict


def grouped_stratified_split(records, seed, fractions=(0.7, 0.15, 0.15),
                             names=("train", "val", "test")):
    """Split records into partitions by lesion_id, stratified by class label.

    records: iterable of dicts with 'image_id', 'lesion_id', 'label'.
    Each lesion is assigned the label of its first record; lesions whose images
    carry conflicting labels raise ValueError. Returns {name: [image_id, ...]}.
    """
    if len(fractions) != len(names) or abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("fractions must match names and sum to 1")
    lesions = {}
    images = defaultdict(list)
    for r in records:
        lid, label = r["lesion_id"], r["label"]
        if lesions.setdefault(lid, label) != label:
            raise ValueError(f"lesion {lid!r} has conflicting labels")
        images[lid].append(r["image_id"])
    by_class = defaultdict(list)
    for lid, label in lesions.items():
        by_class[label].append(lid)
    rng = random.Random(seed)
    out = {n: [] for n in names}
    for label in sorted(by_class, key=str):
        ids = sorted(by_class[label], key=str)
        rng.shuffle(ids)
        n = len(ids)
        cuts, acc = [], 0.0
        for f in fractions[:-1]:
            acc += f
            cuts.append(round(acc * n))
        bounds = [0] + cuts + [n]
        for name, lo, hi in zip(names, bounds[:-1], bounds[1:]):
            for lid in ids[lo:hi]:
                out[name].extend(images[lid])
    return out


def assert_no_leakage(records, split):
    """Raise AssertionError if any lesion or image appears in >1 partition."""
    lesion_of = {r["image_id"]: r["lesion_id"] for r in records}
    seen_lesion, seen_image = {}, {}
    for name, ids in split.items():
        for img in ids:
            if seen_image.setdefault(img, name) != name:
                raise AssertionError(f"image {img!r} in {seen_image[img]} and {name}")
            lid = lesion_of[img]
            if seen_lesion.setdefault(lid, name) != name:
                raise AssertionError(f"lesion {lid!r} in {seen_lesion[lid]} and {name}")


def multi_seed_splits(records, seeds, **kwargs):
    """Return {seed: split}, verifying leakage-freedom for each."""
    records = list(records)
    result = {}
    for s in seeds:
        sp = grouped_stratified_split(records, s, **kwargs)
        assert_no_leakage(records, sp)
        result[s] = sp
    return result
