#!/usr/bin/env python3
"""
读取 training_parameters.yaml，训练并导出量化流式模型。
可多次运行；若检测到 checkpoint 会自动断点续训。
"""

import subprocess, yaml, os, sys, time, pathlib

CONFIG = "training_parameters.yaml"

def main():
    if not pathlib.Path(CONFIG).exists():
        sys.exit("❌ 找不到 training_parameters.yaml，请先执行 prepare_data.py")

    cmd = (
        "python -m microwakeword.model_train_eval "
        f"--training_config='{CONFIG}' "
        "--train 1 "
        "--restore_checkpoint 1 "
        "--test_tf_nonstreaming 0 "
        "--test_tflite_nonstreaming 0 "
        "--test_tflite_nonstreaming_quantized 0 "
        "--test_tflite_streaming 0 "
        "--test_tflite_streaming_quantized 1 "
        "--use_weights best_weights "
        "mixednet "
        "--pointwise_filters 64,64,64,64 "
        "--repeat_in_block 1,1,1,1 "
        "--mixconv_kernel_sizes '[5],[7,11],[9,15],[23]' "
        "--residual_connection 0,0,0,0 "
        "--first_conv_filters 32 "
        "--first_conv_kernel_size 5 "
        "--stride 3"
    )
    print("🏃 开始训练…（如有 checkpoint 会自动续训）")
    subprocess.check_call(cmd, shell=True)
    print("✅ 训练完成。模型位置：\n"
          "trained_models/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite")

if __name__ == "__main__":
    main()
