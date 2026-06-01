# Training with IPA Phonemes

This setup trains on pre-phonemized IPA text. Do not pass graphemes or raw language text to the training filelists.

Use `phoneme_cleaners` so Matcha keeps the IPA string as-is:

```yaml
cleaners: [phoneme_cleaners]
```

## Filelists

Each row is:

```text
/absolute/or/relative/path.wav|ipa phonemes.
```

Example:

```text
data/synthetic-multilingual-speech/wav/011439-chatterbox-speaker0.wav|sipuʁˈim ʁabˈim nikʃeʁˈu lamakˈom hazˈe.
```

Keep punctuation in the IPA text. For isolated word tests, a leading boundary like `. word.` can help pronunciation.

Resample audio before training if needed. For the configs above, use mono wavs at `22050` Hz. Keep audio samples in the normal audio range `[-1, 1]`.

## Configs

English single speaker:

```text
configs/data/synthetic_en_single_speaker.yaml
configs/experiment/synthetic_en_single_speaker.yaml
```

Mixed-language + 10% English fine-tune example:

```text
configs/data/synthetic_he_en_single_speaker.yaml
configs/experiment/synthetic_he_en_single_speaker.yaml
```

Keep the mel settings and stats close to the checkpoint you fine-tune from. For the LJSpeech/Matcha checkpoint path we used, keep the LJSpeech-style mel config/statistics unless intentionally retraining from scratch.

## Train

English fine-tune:

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 uv run --with-editable . python matcha/train.py \
  experiment=synthetic_en_single_speaker \
  extras.print_config=false
```

Mixed-language fine-tune from the latest English checkpoint:

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 uv run --with-editable . python matcha/train.py \
  experiment=synthetic_he_en_single_speaker \
  pretrained_ckpt_path=logs/train/synthetic_en_single_speaker/runs/2026-06-01_17-45-09/checkpoints/last.ckpt \
  extras.print_config=false
```

Resume an interrupted run:

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 uv run --with-editable . python matcha/train.py \
  experiment=synthetic_he_en_single_speaker \
  ckpt_path=logs/train/synthetic_he_en_single_speaker/runs/2026-06-01_18-17-45/checkpoints/last.ckpt \
  extras.print_config=false
```

## Outputs

Checkpoints are saved under:

```text
logs/train/<run_name>/runs/<date>/checkpoints/
```

Use:

```text
last.ckpt
```

for continuing training or quick listening, and the best validation checkpoint for final export.

## TensorBoard

```bash
uv run tensorboard --logdir logs/train
```

The training hook logs generated validation audio when enabled in the Lightning module.

## When It Sounds Fine

For same-speaker fine-tuning, the voice can sound recognizable after the first validation/epoch. Pronunciation and alignment usually need several epochs. Unseen isolated words are harder than sentence context; test both single words and full sentences.
