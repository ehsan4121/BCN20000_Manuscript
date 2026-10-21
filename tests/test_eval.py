import unittest
from bcn20000_eval import (grouped_stratified_split, assert_no_leakage,
                           multi_seed_splits, macro_f1, balanced_accuracy,
                           summarize_seeds)


def make_records():
    recs = []
    for c, n in (("NV", 40), ("MEL", 20), ("DF", 6)):
        for i in range(n):
            for j in range(1 + i % 3):
                recs.append({"image_id": f"{c}{i}_{j}", "lesion_id": f"{c}{i}", "label": c})
    return recs


class T(unittest.TestCase):
    def test_no_leakage_multi_seed(self):
        recs = make_records()
        sp = multi_seed_splits(recs, [0, 1, 2])
        for s in sp.values():
            self.assertEqual(sum(map(len, s.values())), len(recs))
        self.assertNotEqual(sp[0], sp[1])

    def test_deterministic(self):
        r = make_records()
        self.assertEqual(grouped_stratified_split(r, 3), grouped_stratified_split(r, 3))

    def test_leak_detected(self):
        r = make_records()
        sp = grouped_stratified_split(r, 0)
        sp["test"].append(sp["train"][0])
        with self.assertRaises(AssertionError):
            assert_no_leakage(r, sp)

    def test_metrics(self):
        yt, yp = ["a", "a", "b", "b"], ["a", "b", "b", "b"]
        self.assertAlmostEqual(balanced_accuracy(yt, yp), 0.75)
        self.assertAlmostEqual(macro_f1(yt, yp), (2/3 + 0.8) / 2)
        m = summarize_seeds({0: {"f1": 0.5}, 1: {"f1": 0.7}})
        self.assertAlmostEqual(m["f1"][0], 0.6)


if __name__ == "__main__":
    unittest.main()
