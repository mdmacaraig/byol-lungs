"""BYOL for Audio: Audio file converter.

This converts the original audio files found in the source folder recursively,
then store under the destination folder with the same relative path structure.

Converts followings:
    - Stereo to mono
    - Resample to the sampling rate in your config.yaml

Usage:
    python -m utils.convert_wav /path/to/fsd50k work/16k/fsd50k
"""

from byol_a.common import (sys, Path, torch, torchaudio, AT, load_yaml_config)
from multiprocessing import Pool
import fire
from tqdm import tqdm


def _converter_worker(args):
    subpathname, from_dir, to_dir, sample_rate, verbose = args
    from_dir, to_dir = Path(from_dir), Path(to_dir)
    to_name = to_dir/subpathname
    if verbose:
        print(from_dir, '->', to_name)

    # replace 'subpathname' with 'from_dir / subpathname' if downstream task is not icbhi!
    print(f'Attempting to load file: {subpathname}')

    # Load wav
    try:
        wav, org_sr = torchaudio.load(subpathname)
    except Exception as e:
        print(f"Error loading {subpathname}: {e}")
        return None


    # stereo to mono (compatible with librosa)
    # ref: https://librosa.org/doc/main/generated/librosa.to_mono.html#librosa.to_mono
    wav = wav.mean(0, keepdims=True)

    # resample
    wav = AT.Resample(org_sr, sample_rate)(wav)

    # to int16
    wav = (wav * 32767.0).to(torch.int16)

    # save wav
    to_name.parent.mkdir(exist_ok=True, parents=True)
    print(subpathname, ' finished')
    torchaudio.save(to_name, wav, sample_rate)

    return to_name.name


def convert_wav(from_dir, to_dir, config_path='config.yaml', verbose=True) -> None:
    cfg = load_yaml_config(config_path)
    files = [str(f).replace(from_dir, '') for f in Path(from_dir).glob('**/*.wav')]
    files = [f[1:] if f[0] == '/' else f for f in files]
    print(f'Processing {len(files)} files...')
    assert len(files) > 0

    with Pool() as p:
        args = [[f, from_dir, to_dir, cfg.sample_rate, verbose] for f in files]
        list(tqdm(p.imap(_converter_worker, args), total=len(args)))

    print('finished.')


if __name__ == "__main__":
    fire.Fire(convert_wav)