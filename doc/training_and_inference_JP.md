# Diff-SVC(train/inference by yourself)
## 0. 環境設定
>お知らせ: 要件ファイルが更新され、3つのバージョンから選択できるようになりました。
 
1. requirements.txtには、開発・テスト時の全環境が含まれています。Torch1.12.1+cu113が含まれており、pipで直接インストールするか、中のPyTorch関連のパッケージ（torch/torchvision）を削除してからpipでインストールし、独自のtorch環境を使用することが可能です。
    ```
    pip install -r requirements.txt
    ```
2. **(推奨)**: `requirements_short.txt` は上のものを手動で整理したものですが、トーチ本体は含まれていません。また、以下のコードを実行するだけでもOKです。
    ```
    pip install -r requirements_short.txt
    ```
3. プロジェクトのルートディレクトリの下に @三千がコンパイルした要件リスト (requirements.png) があり、これは某ブランドのクラウドサーバーでテストされたものです。しかし、そのtorchバージョンはもう最新のコードとは互換性がありませんが、他の要件のバージョンは参考として使うことができます。

## 1. 推論
>プロジェクトのルートディレクトリにある `inference.ipynb` を使うか、@IceKyrin が書いた `infer.py` を筆者が推論用にアレンジしたものを使うことができます。
最初のブロックにある以下のパラメータを編集してください。
```
config_path= 'location of config.yaml in the checkpoints archive'
# E.g.: './checkpoints/nyaru/config.yaml'
# コンフィグとチェックポイントは一対一で対応しています。他のコンフィグファイルは使用しないでください。

project_name='name of the current project'
# E.g.: 'nyaru'

model_path='full path to the ckpt file'
# E.g.: './checkpoints/nyaru/model_ckpt_steps_112000.ckpt'

hubert_gpu=True
# 推論時にHuBERT（モデル内のモジュール）にGPUを使用するかどうか。モデルの他の部分には影響しない。 
# 現在のバージョンでは、HuBERTモジュールの推論を行う際のGPU使用率を大幅に削減しています。1060 6GのGPUで完全な推論が可能なため、オフにする必要はありません。
# また、長い音声の自動スライスがサポートされました（inference.ipynbとinfer.pyの両方がサポートしています）。30秒以上の音声は、@IceKyrinのコードのおかげで、無音部分で自動的にスライスされるようになりました。
```
### 調整可能なパラメータ
```
wav_fn='xxx.wav'  
# 入力オーディオへのパス。デフォルトのパスは、プロジェクトのルートディレクトリにあります。

use_crepe=True  
# CREPEはF0抽出アルゴリズムである。性能は良いが、遅い。これをFalseに変更すると、若干劣るがはるかに高速なParselmouthアルゴリズムが使用される。

thre=0.05  
# CREPEのノイズフィルタリングの閾値。入力音声がクリーンな場合は増加させることができますが、入力音声にノイズがある場合は、この値を維持するか、または減少させます。このパラメータは、前のパラメータがFalseに設定されている場合、何の効果もありません。

pndm_speedup=20  
# 推論加速度の倍率。デフォルトの拡散ステップ数は1000なので、この値を10に変更すると、100ステップで合成することになる。デフォルトの20は適度な値である。この値は50倍（20ステップで合成）までなら明らかな品質低下はありませんが、それ以上では大幅な品質低下を招く可能性があります。注意：下のuse_gt_melが有効な場合、この値がadd_noise_stepより小さいことを確認してください。また、この値は、拡散ステップ数で割り切れる値である必要があります。

key=0
# Transpose パラメーターです。デフォルト値は0です（1ではありません！）。入力された音声のピッチを{key}半音ずつずらし、合成します。例えば、男性の声を女性の声に変えるには、この値を8や12などに設定します（12は1オクターブ上へシフトします）。

use_pe=True
# Melスペクトログラムからオーディオを合成するためのF0抽出アルゴリズム。Falseに変更すると、入力オーディオのF0が使用されます。
# TrueとFalseを使用した場合、結果に若干の違いがあります。通常はTrueにした方が良いですが、必ずそうなるわけではありません。合成速度にはほとんど影響しません。
# (キーパラメータの値が何であるかにかかわらず、この値は常に変更可能であり、影響を及ぼさない)
# 44.1kHzの機種ではこの機能はサポートされていないため、自動的にOFFになります。オンにしておいても同様にエラーは発生しません。

use_gt_mel=False
# このオプションは、AIペイントの画像間機能に類似しています。Trueに設定すると、出力音声は入力話者とターゲット話者の音声の混合となり、その混合比は以下のパラメータで決定されます。
# 注意!!!このパラメータがtrueに設定されている場合、移調はサポートされていないため、keyパラメータが0に設定されていることを確認してください。

add_noise_step=500
# 前のパラメータと関連し、入力音声とターゲット音声の比率を制御します。1を指定すると完全に入力音声となり、1000を指定すると完全にターゲット音声となります。300程度にすると、両者がほぼ均等に混ざった状態になります。(この値は線形ではないので、このパラメータが非常に低い値に設定されている場合は、pndm_speedupを下げて合成の質を高めることができます)


wav_gen='yyy.wav'
# 出力する音声へのパス。デフォルトはプロジェクトのルートディレクトリにあります。ここでファイル拡張子を変更することで、ファイルタイプを変更することができます。
```

infer.pyを使用する場合、パラメータの変更方法は同様です。 `__name__=='__main__'` 内の値を変更し、プロジェクトのルートディレクトリで `python infer.py` を実行します。この方法では、入力音声をraw/の下に、出力音声をresults/の下に置く必要があります。

## 2. データ作成・トレーニング
### 2.1 データ作成
>Currently, both WAV and Ogg format audio are supported. The sampling rate is better to be higher than 24kHz. The program will automatically handle issues with sampling rates and the number of channels. The sampling rate should not be lower than 16kHz (which usually will not). \
The audio is better to be sliced into segments of 5-15 seconds. While there is no specific requirement for the audio length, it is best for them not to be too long or too short. The audio needs to be the target speaker's dry vocals without background music or other voices, preferably without excessive reverb, etc. If the audio is processed through vocal extraction, please try to keep the audio quality as high as possible. \
Currently, only single-speaker training is supported. The total audio duration should be 3 hours or above. No additional labeling is required. Just place the audio files under raw_data_dir described below. The structure of this directory does not matter; the program will locate the files by itself.

### 2.2 ハイパーパラメータの編集
>First, make a backup copy of config.yaml (this file is for the 24kHz vocoder; use config_nsf.yaml for the 44.1kHz vocoder), then edit it: \
The parameters below might be used (using project name `nyaru` as an example):
```
K_step: 1000
# The total number of diffusion steps. Changing this is not recommended.

binary_data_dir: data/binary/nyaru
# The path to the pre-processed data: the last part needs to be changed to the current project name.

config_path: training/config.yaml
# The path to this config.yaml itself that you are using. Since data will be written into this file during the pre-processing process, this must be the full path to where the yaml file will be stored.

choose_test_manually: false
# Manually selecting a test set. It is disabled by default, and the program will automatically randomly select 5 audio files as the test set.
# If set to true, enter the prefixes of the filenames of the test files in test_prefixes. The program will use the files starting with the corresponding prefix(es) as the test set.
# This is a list and can contain multiple prefixes, e.g.
test_prefixes:
- test
- aaaa
- 5012
- speaker1024
# IMPORTANT: the test set CAN NOT be empty. To avoid unintended effects, it is recommended to avoid manually selecting the test set.

endless_ds:False
# If your dataset is too small, each epoch will pass very fast. Setting this to True will treat 1000 epochs as a single one.

hubert_path: checkpoints/hubert/hubert.pt
# The path to the HuBERT model, make sure this path is correct. In most cases, the decompressed checkpoints.zip archive would put the model under the right path, so no edits are needed. The torch version is now used for inference.

hubert_gpu:True
# Whether or not to use GPU for HuBERT (a module in the model) during pre-processing. If set to False, CPU will be used, and the processing time will increase significantly. Note that whether GPU is used during inference is controlled separately in inference and not affected by this. Since HuBERT changed to the torch version, it is possible to run pre-processing and inference audio under 1 minute without exceeding VRAM limits on a 1060 6G GPU now, so it is usually not necessary to set it to False.

lr: 0.0008
# Initial learning rate: this value corresponds to a batch size of 88; if the batch size is smaller, you can lower this value a bit.

decay_steps: 20000
# For every 20,000 steps, the learning rate will decay to half the original. If the batch size is small, please increase this value.

# For a batch size of about 30-40, the recommended values are lr=0.0004，decay_steps=40000

max_frames: 42000
max_input_tokens: 6000
max_sentences: 88
max_tokens: 128000
# The batch size is calculated dynamically based on these parameters. If unsure about their exact meaning, you can change the max_sentences parameter only, which sets the maximum limit for the batch size to avoid exceeding VRAM limits.

pe_ckpt: checkpoints/0102_xiaoma_pe/model_ckpt_steps_60000.ckpt
# Path to the pe model. Make sure this file exists. Refer to the inference section for its purpose.

raw_data_dir: data/raw/nyaru
# Path to the directory of the raw data before pre-processing. Please put the raw audio files under this directory. The structure inside does not matter, as the program will automatically parse it.

residual_channels: 384
residual_layers: 20
# A group of parameters that control the core network size. The higher the values, the more parameters the network has and the slower it trains, but this does not necessarily lead to better results. For larger datasets, you can change the first parameter to 512. You can experiment with them on your own. However, it is best to leave them as they are if you are not sure what you are doing. 

speaker_id: nyaru
# The name of the target speaker. Currently, only single-speaker is supported. (This parameter is for reference only and has no functional impact)

use_crepe: true
# Use CREPE to extract F0 for pre-processing. Enable it for better results, or disable it for faster processing.

val_check_interval: 2000
# Inference on the test set and save checkpoints every 2000 steps.

vocoder_ckpt:checkpoints/0109_hifigan_bigpopcs_hop128
# For 24kHz models, this should be the path to the directory of the corresponding vocoder. For 44.1kHz models, this should be the path to the corresponding vocoder file itself. Be careful, do not put the wrong one.

work_dir: checkpoints/nyaru
# Change the last part to the project name. (Or it can also be deleted or left completely empty to generate this directory automatically, but do not put some random names)

no_fs2: true
# Simplification of the network encoder. It can reduce the model size and speed up training. No direct evidence of damage to the network performance has been found so far. Enabled by default.

```
> Do not edit the other parameters if you do not know that they do, even if you think you may know by judging from their names. 

### 2.3 データ前処理
Run the following commands under the diff-svc directory: \
#windows
```
set PYTHONPATH=.
set CUDA_VISIBLE_DEVICES=0
python preprocessing/binarize.py --config training/config.yaml
```
#linux
```
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python preprocessing/binarize.py --config training/config.yaml
```
For pre-processing, @IceKyrin has prepared a code for processing HuBERT and other features separately. If your VRAM is insufficient to do it normally, you can run `python ./network/hubert/hubert_model.py` first and then run the pre-processing commands, which can recognize the pre-processed HuBERT features.

### 2.4 トレーニング
#windows
```
set CUDA_VISIBLE_DEVICES=0
python run.py --config training/config.yaml --exp_name nyaru --reset 
```
#linux
```
CUDA_VISIBLE_DEVICES=0 python run.py --config training/config.yaml --exp_name nyaru --reset
```
>You need to change `exp_name` to your project name and edit the config path. Please make sure that the config file used for training is the same as the one used for pre-processing.\
*Important*: After finishing training (on the cloud), if you did not pre-process the data locally, you will need to download the corresponding ckpt file AND the config file for inference. Do not use the one on your local machine since pre-processing writes data into the config file. Make sure the config file used for inference is the same as the one from pre-processing. 

### 2.5 想定される問題

>**2.5.1 'Upsample' オブジェクトに 'recompute_scale_factor' 属性がない。**\
This issue was found in the torch version corresponding to cuda 11.3. If this issue occurs, please locate the `torch.nn.modules.upsampling.py` file in your python package (for example, in a conda environment, it is located under conda_dir\envs\environment_dir\Lib\site-packages\torch\nn\modules\upsampling.py), edit line 153-154 from
```
return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners,recompute_scale_factor=self.recompute_scale_factor)
```
>to
```
return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)
# recompute_scale_factor=self.recompute_scale_factor)
```

>**2.5.2 'utils'という名前のモジュールがありません**\
Please set up in your runtime environment (such as colab notebooks) as follows:
```
import os
os.environ['PYTHONPATH']='.'
!CUDA_VISIBLE_DEVICES=0 python preprocessing/binarize.py --config training/config.yaml
```
Note that this must be done in the project's root directory.

>**2.5.3 ライブラリ 'libsndfile.so' をロードすることができません。**\
This is an error that may occur in a Linux environment. Please run the following command:
```
apt-get install libsndfile1 -y
```
>**2.5.4 import 'consume_prefix_in_state_dict_if_present' を読み込むことができません。**\
The current torch version is too old. Please upgrade to a higher version of torch.

>**2.5.5 データの前処理が遅い**\
Check if `use_crepe` is enabled in config. Turning it off can significantly increase speed.\
Check if `hubert_gpu` is enabled in config.

その他、ご質問等ございましたら、QQチャンネルやDiscordサーバーにご参加いただき、お気軽にお尋ねください。
