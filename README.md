# Lexsi Track 2 (team hum): refusal-direction steering on tiny-aya-fire

Team **hum**: Hassan Rehman (lead) and Mrunmayee Daware.

Entry for the Lexsi AI Safety and Alignment Hackathon (Track 2: AI Red Teaming, Safety and Alignment), Round 1.

## Result in one paragraph

On a matched 170-prompt multilingual harmful-prompt split (71 English, 71 Hindi, 28 Bengali) with deterministic decoding, the unmodified CohereLabs/tiny-aya-fire (revision `73500c80`) hard-refuses 92/170 prompts (54%). Inference-time refusal-direction steering with SafeTune - direction extracted from 16 harmful + 16 harmless contrast prompts per language, layer selected by a validation sweep over layers 7-28 (layer 18 selected), then applied across layers 16/18/20 at strength 2.0 - raises that to 110/170 (65%): 20 previously-compliant harmful prompts become clean refusals, with zero confirmed regressions after manual review of every classifier-flagged item (all flags were phrase changes, not real refusals lost) and no measurable fluency degradation (mean output tokens 100 -> 102, length-limited outputs 97 -> 94 of 170). Single-layer steering at layer 18 helps monotonically from strength 1.5 to 3.0 but plateaus at 101; multi-layer at strength 2.0 does best. Scope note: these findings are from a 170-prompt split with a phrase-based classifier plus manual review of flagged items - a feasibility result, not a semantic-safety audit.

## Contents

- `Lexsi_hum_baseline_preflight.ipynb` - Colab notebook (free-tier T4) used for the runs.
- `scripts/03_baseline_colab.py` - baseline generation over the matched split.
- `scripts/04_intervention_colab.py` - SafeTune refusal-direction extraction + steering sweep (`--strength`, `--layers`, `--skip-done`).
- `scripts/05_circuitkit_check_colab.py` - CircuitKIT load/discovery smoke check.
- `scripts/classify_v2.py` - phrase-based response classifier (flags regressions for manual review).
- `lexsi-track2-hum-writeup.pdf` - the full private writeup (method, sweep matrix, per-language breakdown, capability check, caveats).

Eval data, per-config responses and manifests, and the extracted refusal-direction artifact are published in the linked Hugging Face collections (see the submission / HF article).

## Reproduce

1. Open the notebook on a free Colab T4.
2. `pip install` the bundle requirements, run `03_baseline_colab.py` then `04_intervention_colab.py` per configuration.
3. Direction extraction caches to `/content/direction_cache.pt`; each 170-prompt generation pass takes about 20 minutes on a T4.
