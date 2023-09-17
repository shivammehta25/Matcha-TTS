import tempfile
from argparse import Namespace
from pathlib import Path

import gradio as gr
import soundfile as sf
import torch

from matcha.cli import (
    MATCHA_URLS,
    VOCODER_URL,
    assert_model_downloaded,
    get_device,
    load_matcha,
    load_vocoder,
    process_text,
    to_waveform,
)
from matcha.utils.utils import get_user_data_dir, plot_tensor

LOCATION = Path(get_user_data_dir())

args = Namespace(
    cpu=False,
    model="matcha_ljspeech",
    vocoder="hifigan_T2_v1",
    spk=None,
)

MATCHA_TTS_LOC = LOCATION / f"{args.model}.ckpt"
VOCODER_LOC = LOCATION / f"{args.vocoder}"
LOGO_URL = "https://shivammehta25.github.io/Matcha-TTS/images/logo.png"
assert_model_downloaded(MATCHA_TTS_LOC, MATCHA_URLS[args.model])
assert_model_downloaded(VOCODER_LOC, VOCODER_URL[args.vocoder])
device = get_device(args)

model = load_matcha(args.model, MATCHA_TTS_LOC, device)
vocoder, denoiser = load_vocoder(args.vocoder, VOCODER_LOC, device)


@torch.inference_mode()
def process_text_gradio(text):
    output = process_text(1, text, device)
    return output["x_phones"][1::2], output["x"], output["x_lengths"]


@torch.inference_mode()
def synthesise_mel(text, text_length, n_timesteps, temperature, length_scale):
    output = model.synthesise(
        text,
        text_length,
        n_timesteps=n_timesteps,
        temperature=temperature,
        spks=args.spk,
        length_scale=length_scale,
    )
    output["waveform"] = to_waveform(output["mel"], vocoder, denoiser)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        sf.write(fp.name, output["waveform"], 22050, "PCM_24")

    return fp.name, plot_tensor(output["mel"].squeeze().cpu().numpy())


def run_full_synthesis(text, n_timesteps, mel_temp, length_scale):
    phones, text, text_lengths = process_text_gradio(text)
    audio, mel_spectrogram = synthesise_mel(text, text_lengths, n_timesteps, mel_temp, length_scale)
    return phones, audio, mel_spectrogram


def main():
    description = """# 🍵 Matcha-TTS: A fast TTS architecture with conditional flow matching
    ### [Shivam Mehta](https://www.kth.se/profile/smehta), [Ruibo Tu](https://www.kth.se/profile/ruibo), [Jonas Beskow](https://www.kth.se/profile/beskow), [Éva Székely](https://www.kth.se/profile/szekely), and [Gustav Eje Henter](https://people.kth.se/~ghe/)
    We propose 🍵 Matcha-TTS, a new approach to non-autoregressive neural TTS, that uses conditional flow matching (similar to rectified flows) to speed up ODE-based speech synthesis. Our method:


    * Is probabilistic
    * Has compact memory footprint
    * Sounds highly natural
    * Is very fast to synthesise from


    Check out our [demo page](https://shivammehta25.github.io/Matcha-TTS). Read our [arXiv preprint for more details](https://arxiv.org/abs/2309.03199).
    Code is available in our [GitHub repository](https://github.com/shivammehta25/Matcha-TTS), along with pre-trained models.

    Cached examples are available at the bottom of the page.
    """

    with gr.Blocks(title="🍵 Matcha-TTS: A fast TTS architecture with conditional flow matching") as demo:
        processed_text = gr.State(value=None)
        processed_text_len = gr.State(value=None)

        with gr.Box():
            with gr.Row():
                gr.Markdown(description, scale=3)
                gr.Image(LOGO_URL, label="Matcha-TTS logo", height=150, width=150, scale=1, show_label=False)

        with gr.Box():
            with gr.Row():
                gr.Markdown("# Text Input")
            with gr.Row():
                text = gr.Textbox(value="", lines=2, label="Text to synthesise")

            with gr.Row():
                gr.Markdown("### Hyper parameters")
            with gr.Row():
                n_timesteps = gr.Slider(
                    label="Number of ODE steps",
                    minimum=0,
                    maximum=100,
                    step=1,
                    value=10,
                    interactive=True,
                )
                length_scale = gr.Slider(
                    label="Length scale (Speaking rate)",
                    minimum=0.5,
                    maximum=1.5,
                    step=0.05,
                    value=1.0,
                    interactive=True,
                )
                mel_temp = gr.Slider(
                    label="Sampling temperature",
                    minimum=0.00,
                    maximum=2.001,
                    step=0.16675,
                    value=0.667,
                    interactive=True,
                )

                synth_btn = gr.Button("Synthesise")

        with gr.Box():
            with gr.Row():
                gr.Markdown("### Phonetised text")
                phonetised_text = gr.Textbox(interactive=False, scale=10, label="Phonetised text")

        with gr.Box():
            with gr.Row():
                mel_spectrogram = gr.Image(interactive=False, label="mel spectrogram")

                # with gr.Row():
                audio = gr.Audio(interactive=False, label="Audio")

        with gr.Row():
            examples = gr.Examples(  # pylint: disable=unused-variable
                examples=[
                    [
                        "We propose Matcha-TTS, a new approach to non-autoregressive neural TTS, that uses conditional flow matching (similar to rectified flows) to speed up O D E-based speech synthesis.",
                        50,
                        0.677,
                        1.0,
                    ],
                    [
                        "The Secret Service believed that it was very doubtful that any President would ride regularly in a vehicle with a fixed top, even though transparent.",
                        2,
                        0.677,
                        1.0,
                    ],
                    [
                        "The Secret Service believed that it was very doubtful that any President would ride regularly in a vehicle with a fixed top, even though transparent.",
                        4,
                        0.677,
                        1.0,
                    ],
                    [
                        "The Secret Service believed that it was very doubtful that any President would ride regularly in a vehicle with a fixed top, even though transparent.",
                        10,
                        0.677,
                        1.0,
                    ],
                    [
                        "The Secret Service believed that it was very doubtful that any President would ride regularly in a vehicle with a fixed top, even though transparent.",
                        50,
                        0.677,
                        1.0,
                    ],
                    [
                        "The narrative of these events is based largely on the recollections of the participants.",
                        10,
                        0.677,
                        1.0,
                    ],
                    [
                        "The jury did not believe him, and the verdict was for the defendants.",
                        10,
                        0.677,
                        1.0,
                    ],
                ],
                fn=run_full_synthesis,
                inputs=[text, n_timesteps, mel_temp, length_scale],
                outputs=[phonetised_text, audio, mel_spectrogram],
                cache_examples=True,
            )

        synth_btn.click(
            fn=process_text_gradio,
            inputs=[
                text,
            ],
            outputs=[phonetised_text, processed_text, processed_text_len],
            api_name="matcha_tts",
            queue=True,
        ).then(
            fn=synthesise_mel,
            inputs=[processed_text, processed_text_len, n_timesteps, mel_temp, length_scale],
            outputs=[audio, mel_spectrogram],
        )

        demo.queue(concurrency_count=5).launch(share=True)


if __name__ == "__main__":
    main()
