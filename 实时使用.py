# pip3 install pymicro-wakeword
# 将 trained_models 文件夹放在根目录

import sounddevice as sd
from pymicro_wakeword import MicroWakeWord          # PyPI 版包装器

mww = MicroWakeWord("trained_models/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite",
                    probability_threshold=0.5,      # 可按验证集调高
                    sliding_window_samples=5)       # 与训练时一致

# 麦克风必须是 16 kHz / 16-bit / 单声道
stream = sd.InputStream(samplerate=16_000, channels=1, dtype='int16', blocksize=160)  # 10 ms
with stream:
    for block in stream:
        if mww.process_streaming(block.tobytes()):
            print("✨ 唤醒成功！")
