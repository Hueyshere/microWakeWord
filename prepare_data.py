#!/usr/bin/env python3
"""
准备数据、增广并生成 spectrogram 特征。运行一次即可；可通过 --force 触发重新下载/生成。
"""

import argparse, sys, shutil, subprocess, yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_CONF = {
    "en_US": ("en_US-libritts_r-medium.pt", "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt")
}
sys.path.append("piper-sample-generator/")

def sh(cmd, **kw):
    print("⚙️", cmd)
    subprocess.check_call(cmd, shell=True, **kw)

def download_tts_model(lang):
    model_file, url = MODEL_CONF[lang]
    model_path = ROOT / "piper-sample-generator" / "models" / model_file
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        sh(f"wget -O {model_path} {url}")
    return model_path

def gen_wakeword_samples(word, lang, max_samples, batch):
    out_dir = ROOT / "generated_samples"
    out_dir.mkdir(exist_ok=True)
    sh(f'python piper-sample-generator/generate_samples.py "{word}" --model {download_tts_model(lang)} --max-samples {max_samples} --batch-size {batch} --output-dir {out_dir}')

def ensure_dataset_dirs():
    import datasets, soundfile as sf, numpy as np, tqdm, tarfile, zipfile
    # MIT RIR
    rir_dir = ROOT / "mit_rirs"
    if not rir_dir.exists():
        rir_dir.mkdir()
        print("⬇️  MIT RIR…")
        ds = datasets.load_dataset("davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True)
        for row in tqdm.tqdm(ds, desc="RIR"):
            sf.write(rir_dir / Path(row["audio"]["path"]).name, (row["audio"]["array"] * 32767).astype(np.int16), 16000, subtype="PCM_16")
    # AudioSet
    audioset16 = ROOT / "audioset_16k"
    if not audioset16.exists():
        audioset16.mkdir()
        part = ROOT / "audioset" / "bal_train09.tar"
        if not part.exists():
            (ROOT / "audioset").mkdir()
            sh(f"wget -O {part} https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/bal_train09.tar")
            with tarfile.open(part) as t: t.extractall(ROOT / "audioset")
        audio_files = list((ROOT / "audioset" / "audio").rglob("*.flac"))
        ds = datasets.Dataset.from_dict({"audio": [str(p) for p in audio_files]}).cast_column("audio", datasets.Audio(sampling_rate=16000))
        for row in tqdm.tqdm(ds, desc="AudioSet 16 k"):
            sf.write(audioset16 / Path(row["audio"]["path"]).with_suffix(".wav").name, (row["audio"]["array"] * 32767).astype(np.int16), 16000, subtype="PCM_16")
    # FMA
    fma16 = ROOT / "fma_16k"
    if not fma16.exists():
        fma16.mkdir()
        zip_path = ROOT / "fma" / "fma_xs.zip"
        if not zip_path.exists():
            (ROOT / "fma").mkdir()
            sh(f"wget -O {zip_path} https://huggingface.co/datasets/mchl914/fma_xsmall/resolve/main/fma_xs.zip")
            with zipfile.ZipFile(zip_path) as z: z.extractall(ROOT / "fma")
        mp3s = list((ROOT / "fma" / "fma_small").rglob("*.mp3"))
        ds = datasets.Dataset.from_dict({"audio": [str(p) for p in mp3s]}).cast_column("audio", datasets.Audio(sampling_rate=16000))
        for row in tqdm.tqdm(ds, desc="FMA 16 k"):
            sf.write(fma16 / Path(row["audio"]["path"]).with_suffix(".wav").name, (row["audio"]["array"] * 32767).astype(np.int16), 16000, subtype="PCM_16")

def download_negative_sets():
    neg_dir = ROOT / "negative_datasets"
    if neg_dir.exists(): return
    neg_dir.mkdir()
    base = "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/"
    for fname in ["dinner_party.zip", "dinner_party_eval.zip", "no_speech.zip", "speech.zip"]:
        path = neg_dir / fname
        sh(f"wget -O {path} {base}{fname}")
        sh(f"unzip -q {path} -d {neg_dir}")

def gen_augmented_features():
    from microwakeword.audio.augmentation import Augmentation
    from microwakeword.audio.clips import Clips
    from microwakeword.audio.spectrograms import SpectrogramGeneration
    from mmap_ninja.ragged import RaggedMmap
    clips = Clips(input_directory="generated_samples", file_pattern="*.wav", random_split_seed=10, split_count=0.1)
    augmenter = Augmentation(
        augmentation_duration_s=3.2,
        augmentation_probabilities={
            "SevenBandParametricEQ": 0.1, "TanhDistortion": 0.1, "PitchShift": 0.1, "BandStopFilter": 0.1,
            "AddColorNoise": 0.1, "AddBackgroundNoise": 0.75, "Gain": 1.0, "RIR": 0.5,
        },
        impulse_paths=[], background_paths=["fma_16k", "audioset_16k"],
        background_min_snr_db=-5, background_max_snr_db=10,
        min_jitter_s=0.195, max_jitter_s=0.205,
    )
    out_root = Path("generated_augmented_features"); out_root.mkdir(exist_ok=True)
    for split, rep, slide in [("train", 2, 10), ("validation", 1, 10), ("test", 1, 1)]:
        gen = SpectrogramGeneration(clips=clips, augmenter=augmenter, slide_frames=slide, step_ms=10)
        RaggedMmap.from_generator(
            out_dir=out_root / split / "wakeword_mmap",
            sample_generator=gen.spectrogram_generator(split=("training" if split == "train" else split), repeat=rep),
            batch_size=100, verbose=True)

def write_training_yaml():
    cfg = {
        "window_step_ms": 10, "train_dir": "trained_models/wakeword",
        "features": [
            {"features_dir": "generated_augmented_features", "sampling_weight": 2.0, "penalty_weight": 1.0, "truth": True, "truncation_strategy": "truncate_start", "type": "mmap"},
            {"features_dir": "negative_datasets/speech", "sampling_weight": 10.0, "penalty_weight": 1.0, "truth": False, "truncation_strategy": "random", "type": "mmap"},
            {"features_dir": "negative_datasets/dinner_party", "sampling_weight": 10.0, "penalty_weight": 1.0, "truth": False, "truncation_strategy": "random", "type": "mmap"},
            {"features_dir": "negative_datasets/no_speech", "sampling_weight": 5.0, "penalty_weight": 1.0, "truth": False, "truncation_strategy": "random", "type": "mmap"},
            {"features_dir": "negative_datasets/dinner_party_eval", "sampling_weight": 0.0, "penalty_weight": 1.0, "truth": False, "truncation_strategy": "split", "type": "mmap"},
        ],
        "training_steps": [10_000], "positive_class_weight": [1], "negative_class_weight": [20], "learning_rates": [1e-3],
        "batch_size": 128, "time_mask_max_size": [0], "time_mask_count": [0], "freq_mask_max_size": [0], "freq_mask_count": [0],
        "eval_step_interval": 500, "clip_duration_ms": 1_500, "target_minimization": 0.9, "minimization_metric": None, "maximization_metric": "average_viable_recall",
    }
    with open("training_parameters.yaml", "w") as fp: yaml.safe_dump(cfg, fp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_word", default="hey_ploude")
    ap.add_argument("--language", default="en_US", choices=list(MODEL_CONF))
    ap.add_argument("--force", action="store_true", help="重新生成/下载")
    args = ap.parse_args()
    if args.force: shutil.rmtree("generated_samples", ignore_errors=True)
    gen_wakeword_samples(args.target_word, args.language, 1000, 100)
    ensure_dataset_dirs()
    download_negative_sets()
    gen_augmented_features()
    write_training_yaml()
    print("✅ 数据准备完成。下一步：python train_wakeword.py")

if __name__ == "__main__":
    main()
