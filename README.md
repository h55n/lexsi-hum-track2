# Lexsi Track 2a (team hum): refusal-direction steering on tiny-aya-fire

**Team hum: Mrunmayee Daware and Hassan Rehman, equal partners (50-50).**

Round 1 entry for the Lexsi AI Safety and Alignment Hackathon, Track 2a (AI Safety and Red Teaming).

## TL;DR (before -> after)

We make a small open-weights multilingual model meaningfully safer with inference-time steering, and show it with before-and-after metrics. On a matched 170-prompt harmful-prompt split (71 English, 71 Hindi, 28 Bengali) with deterministic decoding, the unmodified `CohereLabs/tiny-aya-fire` hard-refuses 92/170 prompts (54%). SafeTune refusal-direction steering (layers 16/18/20, strength 2.0) raises that to 110/170 (65%): 20 previously-compliant harmful prompts become clean refusals, with zero confirmed regressions after manual review of every classifier-flagged item. Scope: a feasibility result on one matched split with a phrase-based classifier plus manual review - not a semantic-safety audit.

## Method

1. **Model:** `CohereLabs/tiny-aya-fire` (dense, open-weights; revision `73500c80`). Our use case is multilingual safety (English, Hindi, Bengali).
2. **Toolkit:** SafeTune (Lexsi open-source stack) for refusal-direction extraction and inference-time steering; CircuitKIT load/discovery smoke check. No fine-tuning.
3. **Direction extraction:** contrast set of 16 harmful + 16 harmless prompts per language; steering layers selected by a validation sweep over layers 7-28 (layer 18 best single layer), then applied across layers 16/18/20 at strength 2.0. The full strength sweep (1.5-4.0) is documented in the writeup.
4. **Evaluation:** matched 170-prompt harmful split (71 en / 71 hi / 28 bn), deterministic decoding (`do_sample=false`, `max_new_tokens=128`), phrase-based refusal classifier (`scripts/classify_v2.py`), plus manual review of every flagged item.

## Results

| Config | Layers | Strength | Hard refusals (of 170) | Fixes | Confirmed regressions |
|---|---|---|---|---|---|
| baseline | - | - | 92 (54%) | - | - |
| s1.5 | 18 | 1.5 | 96 | 4 | 0 |
| s2.0 | 18 | 2.0 | 99 | 8 | 0 |
| s3.0 | 18 | 3.0 | 101 | 10 | 0 |
| s4.0 | 18 | 4.0 | 101 | 11 | 0 |
| s2.0_ml | 16, 18, 20 | 2.0 | 110 (65%) | 20 | 0 |

Every flagged item was read in full; all flags were phrase changes the classifier missed, not real refusals lost. Fluency: mean output tokens 100 -> 102; length-limited outputs 97 -> 94 of 170 (no measurable degradation).

## Independent collaboration comparison (Mrunmayee Daware's side, separate run)

Separate compute (Kaggle), calibration and datasets; not merged with the runs above and does not establish equivalence with our configuration. SafeTune 0.1.19, seed 0, greedy decoding, strength 2.0 across layers [16,17,18]; calibration on 32 BeaverTails harmful + 32 Alpaca harmless prompts; safety measured on 170 English AdvBench goals with phrase detectors.

- **Capability:** ARC-Easy (en, n=200) 88% -> 90%; XNLI (hi, n=100) 65% -> 70%; Belebele (en/hi/bn, n=60 each) all lower (86.67 -> 83.33, 70 -> 68.33, 70 -> 66.67).
- **Refusal:** already at the phrase-detector ceiling before steering (170/170 harmful refused in both runs).
- **Limitations:** no significance testing, first-n slices with custom scoring, revisions not pinned, no exported direction/calibration texts. These results do not support a "no regressions" or improved semantic-safety claim.

## Judging criteria coverage

- **Before-and-after metrics on a small open-weights model:** baseline 92/170 -> steered 110/170 hard refusals (table above).
- **Dense open-weights Cohere Labs model + the open-source stack:** `tiny-aya-fire` + SafeTune, with a CircuitKIT smoke check.
- **A real pipeline, end to end:** direction extraction -> layer validation -> steering -> matched-split evaluation -> manual review.
- **Track 2a problem statements:** red-teaming and robustness evaluation; multilingual and code-mixed safety.
- **Findings stay open:** code and writeup in this repo; eval data, per-config generations, manifests and the extracted refusal direction in the linked Hugging Face collections. All runs on free compute.

## Contents

- `Lexsi_hum_baseline_preflight.ipynb` - Colab notebook (free-tier T4) used for the runs.
- `scripts/03_baseline_colab.py` - baseline generation over the matched split.
- `scripts/04_intervention_colab.py` - SafeTune direction extraction + steering sweep (`--strength`, `--layers`, `--skip-done`).
- `scripts/05_circuitkit_check_colab.py` - CircuitKIT load/discovery smoke check.
- `scripts/classify_v2.py` - phrase-based response classifier (flags regressions for manual review).
- `lexsi-track2-hum-writeup.pdf` - full writeup (method, sweep matrix, per-language breakdown, capability check, collaboration comparison, limitations).
- Model collection: https://huggingface.co/collections/ha55an-hg/lexsi-track-2-team-hum-model
- Dataset collection: https://huggingface.co/collections/ha55an-hg/lexsi-track-2-team-hum-dataset

## Reproduce

1. Open the notebook on a free Colab T4.
2. Run `03_baseline_colab.py`, then `04_intervention_colab.py` per configuration.
3. Direction extraction caches to `/content/direction_cache.pt`; each 170-prompt generation pass takes about 20 minutes on a T4.

## Honest limitations

- A 170-prompt matched split with a phrase-based classifier plus manual review of flagged items: a feasibility result, not a semantic-safety audit.
- The independent collaboration run found mixed capability effects (all three Belebele slices lower) and does not support a broader no-regressions or improved semantic-safety claim.
