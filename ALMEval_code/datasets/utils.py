from pathlib import Path
import soundfile as sf
from tqdm import tqdm

def binaural2single(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    wav_paths = list(input_path.rglob("*.wav"))
    for wav_path in tqdm(wav_paths):
        out_left_path = output_path / wav_path.relative_to(input_path).with_name(wav_path.stem + "_left.wav")
        out_right_path = output_path / wav_path.relative_to(input_path).with_name(wav_path.stem + "_right.wav")
        out_left_path.parent.mkdir(parents=True, exist_ok=True)
        audio, fs = sf.read(wav_path)
        sf.write(out_left_path, audio[:, 0], fs)
        sf.write(out_right_path, audio[:, 1], fs)
    