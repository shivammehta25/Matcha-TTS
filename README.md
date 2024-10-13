<div align="center">

# 🍵 Matcha-TTS: 条件付きフローマッチングによる高速TTSアーキテクチャ

### [Shivam Mehta](https://www.kth.se/profile/smehta), [Ruibo Tu](https://www.kth.se/profile/ruibo), [Jonas Beskow](https://www.kth.se/profile/beskow), [Éva Székely](https://www.kth.se/profile/szekely), and [Gustav Eje Henter](https://people.kth.se/~ghe/)

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

<p style="text-align: center;">
  <img src="https://shivammehta25.github.io/Matcha-TTS/images/logo.png" height="128"/>
</p>

</div>

> これはMatcha-TTSの非公式日本語特化型コードです。
> なお、本家とはAttention機構が違います

私たちは、ODEに基づく音声合成を高速化するために、[条件付きフローマッチング](https://arxiv.org/abs/2210.02747) ([整流フロー](https://arxiv.org/abs/2209.03003) に類似)を使用する、非自己回帰的ニューラルTTSの新しいアプローチである🍵抹茶TTSを提案しました。
以下が利点です。

- 確率的である
- コンパクトなメモリフットプリント
- 非常に自然に聞こえる
- 合成速度が速い

詳細は[デモページ](https://shivammehta25.github.io/Matcha-TTS)と[ICASSP 2024論文](https://arxiv.org/abs/2309.03199)をご覧ください。

[訓練済みモデル](https://drive.google.com/drive/folders/17C_gYgEHOxI5ZypcfE_k1piKCtyR0isJ?usp=sharing)はCLIまたはgradioインターフェイスで自動的にダウンロードされます。

また、[HuggingFace 🤗 spaces](https://huggingface.co/spaces/shivammehta25/Matcha-TTS)でブラウザ上で🍵Matcha-TTSを試すこともできます。

## 解説動画

[![Watch the video](https://img.youtube.com/vi/xmvJkz3bqw0/hqdefault.jpg)](https://youtu.be/xmvJkz3bqw0)

## インストール

1. 環境を作る(オプション)

```
conda create -n matcha-tts python=3.10 -y
conda activate matcha-tts
```

2. Matcha TTSをpipまたはソースからインストール

```bash
pip install matcha-tts
```

ソースから

```bash
pip install git+https://github.com/tuna2134/Matcha-TTS-JP.git
cd Matcha-TTS-JP
pip install -e .
```

3. CLIを実行 / gradio app / jupyter notebook

```bash
# 必要なモデルをダウンロードします。
matcha-tts --text "<INPUT TEXT>"
```

```bash
matcha-tts-app
```

もしくはjupyter notebookで`synthesis.ipynb`を開きます。

### CLI引数

- テキストを与えての音声生成は以下の通りに実行してください。

```bash
matcha-tts --text "<INPUT TEXT>"
```

- ファイルから音声生成したい場合は以下の通りに実行してください。

```bash
matcha-tts --file <PATH TO FILE>
```

- バッチを利用してのファイルからの音声生成したい場合は以下の通りに実行してください。

```bash
matcha-tts --file <PATH TO FILE> --batched
```

追加の引数

- Speaking rate

```bash
matcha-tts --text "<INPUT TEXT>" --speaking_rate 1.0
```

- Sampling temperature

```bash
matcha-tts --text "<INPUT TEXT>" --temperature 0.667
```

- Euler ODE solver steps

```bash
matcha-tts --text "<INPUT TEXT>" --steps 10
```

## 自分のデータセットを使ってトレーニングする

JSUTデータセットを利用して、トレーニングしましょう！

1. まずJSUTをダウンロードして、data/jsutに配置してください。頑張って`train.txt`と`val.txt`に分けてください。
※wavファイルのサンプリングレートは20040hzにすることをおすすめします。

2. Matcha-TTSをクローンして、移動する。

```bash
git clone https://github.com/tuna2134/Matcha-TTS-JP.git
cd Matcha-TTS-JP
```

3. ソースからパッケージをインストールする

```bash
pip install -e .
```

4. `configs/data/hi-fi_jsut.yaml`を編集する。

```yaml
train_filelist_path: data/train.txt
valid_filelist_path: data/val.txt
```

5. データセット設定のyamlファイルで正規化統計を生成する。

```bash
matcha-data-stats -i jsut.yaml
# Output:
#{'mel_mean': -5.53662231756592, 'mel_std': 2.1161014277038574}
```

これらの値を `configs/data/hi-fi_jsut.yaml` の `data_statistics` キーで更新する。

```bash
data_statistics:  # Computed for ljspeech dataset
  mel_mean: -5.536622
  mel_std: 2.116101
```

6. トレーニングスクリプトを実行してください。

```bash
python matcha/train.py experiment=jsut
```

7. カスタムされたトレーニングモデルで音声を生成する。

```bash
matcha-tts --text "<INPUT TEXT>" --checkpoint_path <PATH TO CHECKPOINT>
```

## ONNXのサポート

> ONNXエクスポートと推論サポートを実装してくれた[@mush42](https://github.com/mush42)に感謝します。

抹茶のチェックポイントを[ONNX](https://onnx.ai/)にエクスポートし、エクスポートされたONNXグラフに対して推論を実行することができます。

### ONNXへ変換

チェックポイントをONNXに変換する前に以下の通りにONNXをインストールしてください。

```bash
pip install onnx
```

その後に以下の通りに実行してください

```bash
python3 -m matcha.onnx.export matcha.ckpt model.onnx --n-timesteps 5
```

(オプション) ONNX変換器は**vocoder-name**と**vocoder-checkpoint**引数を受け付けています。これにより、エクスポートしたグラフにボコーダーを組み込み、1回の実行で波形を生成することができます（エンドツーエンドのTTSシステムと同様）。

**Note** `n_timesteps`はモデル入力ではなくハイパーパラメータとして扱われます。つまり、(推論時ではなく)エクスポート時に指定する必要があります。指定しない場合`n_timesteps`は**5**に設定されます。

**Important**: 古いバージョンでは `scaled_product_attention` 演算子がエクスポートできないため、今のところエクスポートには torch>=2.1.0 が必要です。最終バージョンがリリースされるまでは、モデルをエクスポートしたい人はプレリリースとしてtorch>=2.1.0を手動でインストールする必要があります。

### ONNX推論

エキスポートされたモデルを推論する前に`onnxruntime`以下の通りにインストールしてください。

```bash
pip install onnxruntime
pip install onnxruntime-gpu  # GPU推論する場合
```

その後に以下の通りに実行して推論してください。

```bash
python3 -m matcha.onnx.infer model.onnx --text "hey" --output-dir ./outputs
```

音声合成のパラメーターもコントロールできます。

```bash
python3 -m matcha.onnx.infer model.onnx --text "hey" --output-dir ./outputs --temperature 0.4 --speaking_rate 0.9 --spk 0
```

**GPU**上で推論を実行するには、必ず**onnxruntime-gpu**パッケージをインストールして、推論コマンドに--gpu`を渡してください：

```bash
python3 -m matcha.onnx.infer model.onnx --text "hey" --output-dir ./outputs --gpu
```

MatchaだけをONNXにエクスポートした場合は、mel-spectrogramをグラフと`numpy`配列として出力ディレクトリに書き出します。
エクスポートしたグラフにボコーダーを埋め込んだ場合、`.wav`オーディオファイルを出力ディレクトリに書き出します。

MatchaだけをONNXにエクスポートし、完全なTTSパイプラインを実行したい場合は、`ONNX`フォーマットのボコーダーモデルへのパスを渡すことができます:

```bash
python3 -m matcha.onnx.infer model.onnx --text "hey" --output-dir ./outputs --vocoder hifigan.small.onnx
```

outputディレクトリにwavファイルが書き込まれます。

## Extract phoneme alignments from Matcha-TTS

If the dataset is structured as

```bash
data/
└── LJSpeech-1.1
    ├── metadata.csv
    ├── README
    ├── test.txt
    ├── train.txt
    ├── val.txt
    └── wavs
```
Then you can extract the phoneme level alignments from a Trained Matcha-TTS model using:
```bash
python  matcha/utils/get_durations_from_trained_model.py -i dataset_yaml -c <checkpoint>
```
Example:
```bash
python  matcha/utils/get_durations_from_trained_model.py -i ljspeech.yaml -c matcha_ljspeech.ckpt
```
or simply:
```bash
matcha-tts-get-durations -i ljspeech.yaml -c matcha_ljspeech.ckpt
```
---
## Train using extracted alignments

In the datasetconfig turn on load duration.
Example: `ljspeech.yaml`
```
load_durations: True
```
or see an examples in configs/experiment/ljspeech_from_durations.yaml


## 引用元

私たちのコードを使用する場合、あるいはこの研究が役に立つと思われる場合は、私たちの論文を引用してください。

```text
@inproceedings{mehta2024matcha,
  title={Matcha-{TTS}: A fast {TTS} architecture with conditional flow matching},
  author={Mehta, Shivam and Tu, Ruibo and Beskow, Jonas and Sz{\'e}kely, {\'E}va and Henter, Gustav Eje},
  booktitle={Proc. ICASSP},
  year={2024}
}
```

## 謝辞

Since this code uses [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template), you have all the powers that come with it.

Other source code we would like to acknowledge:

- [Coqui-TTS](https://github.com/coqui-ai/TTS/tree/dev): For helping me figure out how to make cython binaries pip installable and encouragement
- [Hugging Face Diffusers](https://huggingface.co/): For their awesome diffusers library and its components
- [Grad-TTS](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS): For the monotonic alignment search source code
- [torchdyn](https://github.com/DiffEqML/torchdyn): Useful for trying other ODE solvers during research and development
- [labml.ai](https://nn.labml.ai/transformers/rope/index.html): For the RoPE implementation
