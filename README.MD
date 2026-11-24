# Chunked execution for energy-limited inference and training

A small implementation and benchmark of the idea from
**"One Simple Trick for Improving the Performance of Energy-Limited Local Inference and Training"**
(Schultheis, Kleinegger, Alistarh) —
[OpenReview](https://openreview.net/forum?id=QSv1lTk06x) · [arXiv](https://arxiv.org/pdf/2609.11936).
All credit for the idea goes to the paper's authors; this repo is an independent, partial re-implementation.

## Idea
Workloads that alternate compute-bound matmuls with memory-bound ops (e.g. LM-head + cross-entropy) make GPU power
draw spiky and can trigger clock throttling. Splitting the work into row-chunks interleaves compute and memory
phases more finely, which smooths power draw.

## What is here
| File | Purpose |
|---|---|
| `chunked_ops.py` | chunked matmul → softmax → matmul; chunked, checkpointed LM-head cross-entropy |
| `energy.py` | optional NVML sampler (power, SM clock, temperature, estimated energy) |
| `benchmark.py` | chunk-count sweep (chunks=1 is the unmodified baseline), one JSONL row per config |
| `tests/test_chunked.py` | chunked outputs and gradients equal the unchunked baseline |
| `PAPER_CARD.md` | paper card (paper-reported vs locally measured, kept separate) |
| `results/` | T4 summary and raw rows |

## Run
```bash
pip install -r requirements.txt
python -I -m pytest -q tests
python benchmark.py --m 8192 --k 1024 --n 32768 --dtype float16 --chunks 1 2 4 8 16 32 --out results.jsonl
```
On CPU the benchmark only works as a smoke test (labelled `approximate`).

## Results (Kaggle Tesla T4, status: `partial`)
| chunks | median latency (ms) | speedup | peak mem (MB) | energy (J/iter) |
|---:|---:|---:|---:|---:|
| 1 (baseline) | 48.76 | 1.00x | 1176 | 3.30 |
| 2 | 47.13 | 1.03x | 1048 | 3.28 |
| 4 | 47.35 | 1.03x | 792 | 3.22 |
| 8 | 48.20 | 1.01x | 600 | 3.30 |
| 16 | 51.24 | 0.95x | 504 | 3.48 |
| 32 | 59.81 | 0.82x | 456 | 4.04 |

- 2–4 chunks: ~3% lower latency, ~2% lower energy (4 chunks), up to 33% lower peak memory.
- 16–32 chunks are slower. The T4 ran near its power cap (~68–69 W of 70 W), consistent with the paper's explanation.
- Outputs differ from the baseline by at most 0.004 (fp16 rounding).

## Limitations
- The latency gain is about the size of run-to-run noise (baseline varied ~3%); no firm speedup is claimed.
- Smaller shapes than the paper, shared T4 GPU; the paper's own gains were ~15–20% on DGX Spark and ~2.5% on L40S.
- The second matmul follows my reading of the paper's Listing 1 and should be checked against the paper.
- End-to-end LLMQ training and vLLM experiments are not reproduced.
- NVML energy is an estimate; best chunk size depends on hardware.

## Upstream
Contribution PR to the course repo: <add link to your PR in krsnoki/DA4022>

## License
MIT — see `LICENSE`.
