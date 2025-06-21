# microWakeWord — 自定义唤醒词快速训练套件

> 针对 macOS (Apple Silicon) 与 Linux (x86_64/NVIDIA GPU) 自用整理  

## 目录结构约定

```
.
├── README.md
├── prepare_data.py # 第一次运行：准备数据
├── train_wakeword.py # 训练/量化/导出
├── microWakeWord/ # 克隆后本地可编辑安装
├── piper-sample-generator/ # 同上
└── 其余下载数据/模型/训练结果 ...
```

整个项目只需保存这一文件夹即可；以后 **无需联网也不用再装依赖**——  
Python 虚拟环境只要不删除，就不会重复拉包。

## 快速开始

### 1. 创建并激活虚拟环境  

```bash
conda create -n microwakeword python=3.10
conda activate microwakeword
```

### 2. 一次性安装依赖（离线可复用）

**macOS:**

```bash
pip install 'git+https://github.com/puddly/pymicro-features@puddly/minimum-cpp-version'
```

**Linux:**

```bash
pip install 'git+https://github.com/kahrendt/pymicro-features'
```

**通用包（两端都要装）：**

```bash
pip install \
  'git+https://github.com/whatsnowplaying/audio-metadata@d4ebb238e6a401bb1a5aaaac60c9e2b3cb30929f' \
  torch torchaudio piper-phonemize-cross==1.2.1 \
  datasets soundfile tqdm scipy mmap-ninja
```

### 3. 克隆并本地可编辑安装（只需一次）

```bash
git clone https://github.com/kahrendt/microWakeWord
pip install -e ./microWakeWord

# macOS 用支持 MPS 分支；Linux 用主仓库
if [[ "$(uname)" == "Darwin" ]]; then
  git clone -b mps-support https://github.com/kahrendt/piper-sample-generator
else
  git clone https://github.com/rhasspy/piper-sample-generator
fi
pip install -e ./piper-sample-generator
```

### 4. 准备数据

TODO: 中文暂不支持 需要修改代码
```bash
python prepare_data.py \
  --target_word "小陈啊" \
  --language zh_CN          # 可改 en_US / fr_FR / de_DE ...
```

首次执行会下载 & 生成：

*   Piper TTS 模型并合成 1 000 条唤醒词；
*   MIT RIR、AudioSet 子集、FMA 等背景噪声并转 16 kHz；
*   增广后生成训练 / 验证 / 测试 spectrogram Ragged Mmap；
*   负样本数据集 (speech / dinner_party / no_speech …)；
*   自动写出 training_parameters.yaml。

### 5. 训练

```bash
python train_wakeword.py
```

典型耗时（M1 Pro，batch 128）≈ 13 min。

最终模型位于：

```bash
trained_models/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite
```

可直接嵌入移动端或微控制器。