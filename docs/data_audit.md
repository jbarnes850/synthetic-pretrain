# Data Audit

This audit summarizes the released interleaved-thinking dataset:

- Hugging Face dataset: `Jarrodbarnes/qwen3-0.6B-interleaved-thinking-data`
- Rows: 8,704 total, with 8,192 train rows and 512 validation rows
- Source corpus: FineWeb-Edu `sample/10BT`
- Teacher: `qwen3.6-35b-a3b-mlx-int6-mlx`
- Format: raw pretraining chunks with short `<think>...</think>` spans inserted locally

The dataset was built for the supervised thinking-mid-training stage. It is not an instruction dataset. Its role is to install the interface: where short thoughts appear, how they connect to nearby text, and how the original document continues around them.

## Structural Checks

| Check | Result |
| --- | ---: |
| Malformed rows | 0 |
| Empty thought spans | 0 |
| Unexpected action tags after normalization | 0 |
| Rows with overlong thought spans above 90 words | 0 |
| Average thoughts per row | 4.39 |
| Median thought length | 14 words |
| Average raw word coverage | 99.98% |

## Preservation Caveat

Strict preservation checks flagged 66 rows out of 8,704. These rows are retained in the released dataset and should be treated as a known caveat. At corpus scale, preservation is strong; for work that depends on exact reconstruction of the raw chunk, filter or regenerate those rows.

## Intended Reading

The dataset supports small-scale research on interleaved thinking mid-training. It supports the claim that SFT can teach a small model the thought interface. It does not, by itself, prove that generated thoughts improve behavior; the reward gate and causal thought-use probe test that separately.
