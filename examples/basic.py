from pathlib import Path
from types import SimpleNamespace

import soundfile as sf
import torch

from matcha.cli import (
    assert_required_models_available,
    load_vocoder,
    process_text,
    to_waveform,
    validate_args,
)
from matcha.models.matcha_tts import MatchaTTS

TEXT = "Hello from Matcha TTS."
OUTPUT = Path("outputs/basic.wav")


@torch.inference_mode()
def main():
    args = validate_args(
        SimpleNamespace(
            text=TEXT,
            file=None,
            model="matcha_ljspeech",
            checkpoint_path=None,
            vocoder=None,
            spk=None,
            temperature=0.667,
            speaking_rate=None,
            steps=10,
            batched=False,
            batch_size=1,
            denoiser_strength=0.00025,
        )
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = assert_required_models_available(args)

    model = MatchaTTS.load_from_checkpoint(paths["matcha"], map_location=device, weights_only=False).eval()
    vocoder, denoiser = load_vocoder(args.vocoder, paths["vocoder"], device)
    text = process_text(1, TEXT, device)

    output = model.synthesise(
        text["x"],
        text["x_lengths"],
        n_timesteps=args.steps,
        temperature=args.temperature,
        spks=None,
        length_scale=args.speaking_rate,
    )
    waveform = to_waveform(output["mel"], vocoder, denoiser, args.denoiser_strength)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    sf.write(OUTPUT, waveform, 22050, "PCM_24")
    print(f"Wrote {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
