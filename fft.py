import librosa
import numpy as np

def standard(x):
    x = (x - np.mean(x)) / np.std(x)
    return x

def minmax(x, axis=None):
    x_min = x.min(axis=axis, keepdims=True)
    x_max = x.max(axis=axis, keepdims=True)
    return (x - x_min) / (x_max - x_min)


def cqt(
        y,
        sr=22050,
        n_bins=12 * 3 * 7,
        bins_per_octave=12 * 3,
        hop_length=512,
        fmin=32.7,
        window="hann",
        Qfactor=20.0,
        norm=minmax):
    mono = True if len(y.shape) == 1 else False

    filter_scale = (1 / bins_per_octave) * Qfactor

    if mono:
        S = np.abs(
            librosa.cqt(
            y,
            sr=sr,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            hop_length=hop_length,
            filter_scale=filter_scale,
            fmin=fmin,
            scale=True,
            window=window)).astype("float32").T

    else:
        S_lr = np.abs(
            librosa.cqt(
            (y[0] * 0.5 + y[1] * 0.5),
            sr=sr,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            hop_length=hop_length,
            filter_scale=filter_scale,
            fmin=fmin,
            scale=True,
            window=window)).astype("float32")

        S_lrm = np.abs(
            librosa.cqt(
            (y[0] - y[1]),
            sr=sr,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            hop_length=hop_length,
            filter_scale=filter_scale,
            fmin=fmin,
            scale=True,
            window=window)).astype("float32")

        S = np.array((S_lr.T, S_lrm.T))

    if norm is not None:
        S = norm(S)

    return S
