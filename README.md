# BCN20000_Manuscript
Leakage-Safe Lesion-Grouped Multi-Seed Evaluation of Long-Tail Skin Lesion Classification on BCN20000

## Usage
`bcn20000_eval` provides lesion-grouped, class-stratified splits (all images of a
lesion stay in one partition), a leakage check, multi-seed split generation, and
long-tail metrics (macro-F1, balanced accuracy, per-class recall) with
mean/std aggregation across seeds. Pure standard library.

Run tests: `python -m unittest discover -s tests -t .`
