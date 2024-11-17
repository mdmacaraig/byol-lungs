"""BYOL for Audio: Dataset class definition."""
import pywt
from .common import (random, np, torch, F, torchaudio, AF, AT, Dataset)
import librosa
import scipy.signal


class MelSpectrogramLibrosa:
    """Mel spectrogram using librosa."""
    def __init__(self, fs=16000, n_fft=1024, shift=160, n_mels=64, fmin=60, fmax=7800):
        self.fs, self.n_fft, self.shift, self.n_mels, self.fmin, self.fmax = fs, n_fft, shift, n_mels, fmin, fmax
        self.mfb = librosa.filters.mel(sr=fs, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

    def __call__(self, audio):
        X = librosa.stft(np.array(audio), n_fft=self.n_fft, hop_length=self.shift)
        return torch.tensor(np.matmul(self.mfb, np.abs(X)**2 + np.finfo(float).eps))
    
class WaveletScalogram:
    """Wavelet scalogram using PyWavelets with optimizations."""
    def __init__(self, wavelet='cmor1.5-1.0', scales=None, sampling_rate=16000, target_shape=(64, 101)):
        """
        Args:
            wavelet: Type of mother wavelet (e.g., 'morl', 'cmor').
            scales: Array of scales to use for the CWT.
            sampling_rate: Sampling rate of the input audio.
            target_shape: (height, width) to resize the scalogram for consistency.
        """
        self.wavelet = wavelet
        self.sampling_rate = sampling_rate
        self.target_shape = target_shape

        # Default scales optimized for respiratory sounds
        if scales is None:
            self.scales = np.arange(1, 64)  # Reduced number of scales
        else:
            self.scales = scales

    def __call__(self, audio):
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()

        target_rate = 8000  # Reduce temporal resolution
        if self.sampling_rate > target_rate:
            audio = scipy.signal.resample(audio, int(len(audio) * target_rate / self.sampling_rate))

        # Compute CWT
        coefficients, _ = pywt.cwt(audio, scales=self.scales, wavelet=self.wavelet, sampling_period=1 / target_rate)
        scalogram = np.abs(coefficients)  # Magnitude of coefficients

        # Resize scalogram to target_shape
        scalogram_resized = scipy.signal.resample(scalogram, self.target_shape[0], axis=0)  # Reduce frequency resolution
        scalogram_resized = scipy.signal.resample(scalogram_resized, self.target_shape[1], axis=1)  # Reduce time resolution

        return torch.tensor(scalogram_resized, dtype=torch.float32)

class WaveInLMSOutDataset(Dataset):
    """Wave in, log-mel spectrogram out, dataset class.

    Choosing librosa or torchaudio:
        librosa: Stable but slower.
        torchaudio: Faster but cannot reproduce the exact performance of pretrained weight,
            which might be caused by the difference with librosa. Librosa was used in the pretraining.

    Args:
        cfg: Configuration settings.
        audio_files: List of audio file pathnames.
        labels: List of labels corresponding to the audio files.
        tfms: Transforms (augmentations), callable.
        use_librosa: True if using librosa for converting audio to log-mel spectrogram (LMS).
    """

    def __init__(self, cfg, audio_files, labels, tfms, use_librosa=False):
        # argment check
        assert (labels is None) or (len(audio_files) == len(labels)), 'The number of audio files and labels has to be the same.'
        super().__init__()

        # initializations
        self.cfg = cfg
        self.files = audio_files
        self.labels = labels
        self.tfms = tfms
        self.unit_length = int(cfg.unit_sec * cfg.sample_rate)
        self.to_melspecgram = MelSpectrogramLibrosa(
            fs=cfg.sample_rate,
            n_fft=cfg.n_fft,
            shift=cfg.hop_length,
            n_mels=cfg.n_mels,
            fmin=cfg.f_min,
            fmax=cfg.f_max,
        ) if use_librosa else AT.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            power=2,
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # load single channel .wav audio
        wav, sr = torchaudio.load(self.files[idx])
        assert sr == self.cfg.sample_rate, f'Convert .wav files to {self.cfg.sample_rate} Hz. {self.files[idx]} has {sr} Hz.'
        assert wav.shape[0] == 1, f'Convert .wav files to single channel audio, {self.files[idx]} has {wav.shape[0]} channels.'
        wav = wav[0] # (1, length) -> (length,)

        # zero padding to both ends
        length_adj = self.unit_length - len(wav)
        if length_adj > 0:
            half_adj = length_adj // 2
            wav = F.pad(wav, (half_adj, length_adj - half_adj))

        # random crop unit length wave
        length_adj = len(wav) - self.unit_length
        start = random.randint(0, length_adj) if length_adj > 0 else 0
        wav = wav[start:start + self.unit_length]

        # to log mel spectrogram -> (1, n_mels, time)
        lms = (self.to_melspecgram(wav) + torch.finfo().eps).log().unsqueeze(0)

        # transform (augment)
        if self.tfms:
            lms = self.tfms(lms)

        if self.labels is not None:
            return lms, torch.tensor(self.labels[idx])
        return lms

class WaveInOutDataset(Dataset):
    """Wave in, spectrogram/scalogram out, dataset class."""

    def __init__(self, cfg, audio_files, labels, tfms):
        assert (labels is None) or (len(audio_files) == len(labels)), 'The number of audio files and labels has to be the same.'
        super().__init__()

        self.cfg = cfg
        self.files = audio_files
        self.labels = labels
        self.tfms = tfms
        self.unit_length = int(cfg.unit_sec * cfg.sample_rate)

        # Initialize the transformation based on `to_scalogram`
        if getattr(cfg, 'to_scalogram', False):  # Check for `to_scalogram` in cfg
            self.transform = WaveletScalogram(
                wavelet='cmor1.5-1.0',
                sampling_rate=cfg.sample_rate,
            )
        else:
            self.transform = MelSpectrogramLibrosa(
                fs=cfg.sample_rate,
                n_fft=cfg.n_fft,
                shift=cfg.hop_length,
                n_mels=cfg.n_mels,
                fmin=cfg.f_min,
                fmax=cfg.f_max,
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # load single-channel .wav audio
        wav, sr = torchaudio.load(self.files[idx])
        assert sr == self.cfg.sample_rate, f'Convert .wav files to {self.cfg.sample_rate} Hz. {self.files[idx]} has {sr} Hz.'
        assert wav.shape[0] == 1, f'Convert .wav files to single channel audio, {self.files[idx]} has {wav.shape[0]} channels.'
        wav = wav[0]  # (1, length) -> (length,)

        # zero padding to both ends
        length_adj = self.unit_length - len(wav)
        if length_adj > 0:
            half_adj = length_adj // 2
            wav = torch.nn.functional.pad(wav, (half_adj, length_adj - half_adj))

        # random crop unit length wave
        length_adj = len(wav) - self.unit_length
        start = random.randint(0, length_adj) if length_adj > 0 else 0
        wav = wav[start:start + self.unit_length]

        # transform to scalogram or mel spectrogram
        output = self.transform(wav).unsqueeze(0)  # (1, scales or n_mels, time)

        # apply augmentations
        if self.tfms:
            output = self.tfms(output)

        if self.labels is not None:
            return output, torch.tensor(self.labels[idx])
        return output
