# pip3 install pymicro-wakeword
# 将 trained_models 文件夹放在根目录

import sounddevice as sd
from pymicro_wakeword import MicroWakeWord          # PyPI 版包装器
from datetime import datetime

mww = MicroWakeWord(wake_word="hey_plaud",
                    tflite_model="trained_models/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite",
                    probability_cutoff=0.8,      # 可按验证集调高 # TODO 默认 0.5 似乎太敏感，0.7 效果会好一些， 建议生成类似语音去测试得到针对每一个词的阈值
                    sliding_window_size=5,       # 与训练时一致
                    refractory_seconds=0.5)      
# 麦克风必须是 16 kHz / 16-bit / 单声道
stream = sd.InputStream(samplerate=16_000, channels=1, dtype='int16', blocksize=160)  # 10 ms
stream.start()
try:
    print("开始监听...")
    while True:
        block, overflowed = stream.read(160)  # Read 160 samples (10ms at 16kHz)
        if mww.process_streaming(block.tobytes()):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            print(f"✨ 唤醒成功！[{timestamp}]")
except KeyboardInterrupt:
    print("停止监听...")
finally:
    stream.stop()
    stream.close()
