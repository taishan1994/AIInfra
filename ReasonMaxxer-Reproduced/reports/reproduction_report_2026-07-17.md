# ReasonMaxxer Reproduction Report

Date: 2026-07-17

## Target

- Paper: *Rethinking RL for LLM Reasoning: It's Sparse Policy Selection, Not Capability Learning*
- Repo: `ReasonMaxxer`

## Environment

- Python: `3.12.10`
- CUDA / GPU: CUDA `12.9`, 4 x `NVIDIA A800-SXM4-80GB`
- Core libraries:
  - `torch==2.7.0+cu126`
  - `transformers==4.52.3`
  - `peft==0.15.1`
  - `datasets==4.2.0`
  - `huggingface_hub==0.35.3`
  - `modelscope==1.38.1`
  - `vllm==0.9.2`

## Repository adjustments made for reproducibility

- Added explicit `huggingface_hub` dependency to `requirements.txt`.
- Added explicit `modelscope` dependency to `requirements.txt`.
- Added `scripts/fetch_model.py` for model download through:
  - Hugging Face with `HF_ENDPOINT=https://hf-mirror.com`
  - ModelScope when explicitly requested
- Extended `scripts/download_simplerl_data.py` with `--hf_endpoint`.
- Added automated smoke tests in `tests/test_pipeline_smoke.py`.
- Updated `README.md` with mirror/model download and smoke-test instructions.

## Validation performed

- `python -m compileall reasonmaxxer scripts`
- `python -m unittest discover -s tests -p 'test_*.py' -v`
- Tiny one-step LoRA training smoke test with a locally materialized random GPT-2 checkpoint

## Results

### Static checks

- Status: passed
- Result: all modules under `reasonmaxxer/`, `scripts/`, and `tests/` compiled successfully.

### Functional smoke tests

- Status: passed
- Command: `python -m unittest discover -s tests -p 'test_*.py' -v`
- Coverage:
  - answer extraction and answer verification helpers
  - synthetic SimpleRL parquet sampling
  - mid-pool rollout selection
  - training-data preparation
- Result: `Ran 2 tests ... OK`

### Tiny training smoke test

- Model:
  - base path: `models/tiny-gpt2-random`
  - construction: random weights instantiated from the cached `tiny-gpt2` config and tokenizer
- Command:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_reasonmaxxer.py \
  --training_data outputs/smoke_train/processed.json \
  --base_model models/tiny-gpt2-random \
  --output_dir outputs/smoke_train/run \
  --target_modules c_attn,c_proj \
  --lora_rank 4 \
  --lora_alpha 8 \
  --lora_dropout 0.0 \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum_steps 1 \
  --learning_rate 1e-4 \
  --warmup_steps 0 \
  --weight_decay 0.0 \
  --max_grad_norm 1.0 \
  --kl_weight 0.1 \
  --adv_clip 2.5 \
  --decision_objective adv_ce \
  --beta_neg 1.0 \
  --max_seq_len 128 \
  --truncate_side right \
  --dtype float32 \
  --hf_attn_implementation eager \
  --seed 42 \
  --logging_steps 1 \
  --save_every_fractional_epoch 0 \
  --no_save_every_epoch \
  --max_optimizer_steps 1
```
- Outcome:
  - training completed `1` optimizer step successfully
  - final adapter saved to `outputs/smoke_train/run/final`
  - logs written to:
    - `outputs/smoke_train/run/training_log.json`
    - `outputs/smoke_train/run/checkpoint_metrics.csv`
  - final logged metrics:
    - `loss = 10.822072`
    - `decision_tokens = 2`
    - `decision_frac_generated = 0.333333`

## Full paper reproduction plan

1. Download base model with `scripts/fetch_model.py`.
2. Download SimpleRL data with `scripts/download_simplerl_data.py`.
3. Run the example pipeline under `examples/qwen25_1p5b/`.
4. Use `05_eval_holdout60.sh` to choose checkpoint.
5. Use `06_eval_fullsuite.sh` for benchmark evaluation.

## Gaps relative to full paper tables

- Full benchmark reproduction has not yet been completed in this report.
- Exact paper-scale runtime and selected checkpoint metrics still need long-running evaluation.
- External model download through `hf-mirror` remains partially unstable in this environment because some repos still route large files through Hugging Face's `xet` CAS backend.
- The repository-side mitigation is in place:
  - `scripts/fetch_model.py` now sets `HF_ENDPOINT=https://hf-mirror.com`
  - `scripts/fetch_model.py` now defaults `HF_HUB_DISABLE_XET=1`
  - `scripts/fetch_model.py` supports explicit `--source modelscope`
