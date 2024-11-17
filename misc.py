# this is just for testing tensor shape

from byol_a.dataset import MelSpectrogramLibrosa, WaveletScalogram
import numpy as np

mel_transform = MelSpectrogramLibrosa(fs=16000)
wavelet_transform = WaveletScalogram(target_shape=(128, 256))

# random audio
audio = np.random.randn(16000)  # 1 second of audio at 16 kHz

mel_output = mel_transform(audio)
wavelet_output = wavelet_transform(audio)

print(mel_output.shape)
print(wavelet_output.shape)

assert mel_output.shape == wavelet_output.shape, "Tensor shapes do not match!"
print("Shapes are consistent:", mel_output.shape)