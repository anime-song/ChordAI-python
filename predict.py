from dlchordx import Chord
import numpy as np
import librosa
import json
import fft
import logging
import tensorflow as tf
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)


def allocate_gpu_memory(gpu_number=0):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')

    if len(physical_devices) > 0:
        try:
            #print("Found {} GPU(s)".format(len(physical_devices)))
            tf.config.experimental.set_memory_growth(
                physical_devices[gpu_number], True)
            # print("#{} GPU memory is allocated".format(gpu_number))
        except RuntimeError as e:
            print(e)
    else:
        print("Not enough GPU hardware devices available")


# GPUメモリ制限
# allocate_gpu_memory()

# GPUを無効にする
tf.config.set_visible_devices([], 'GPU')

def lastone(iterable):
    """与えられたイテレータブルオブジェクトの
    最後の一つの要素の時にTrue、それ以外の時にFalseを返す
    """
    # イテレータを取得して最初の値を取得する
    it = iter(iterable)
    last = next(it)
    # 2番目の値から開始して反復子を使い果たすまで実行
    for val in it:
        # 一つ前の値を返す
        yield last, False
        last = val  # 値の更新
    # 最後の一つ
    yield last, True


def convert_time_key(pred, bins_per_seconds, minTime=0.1):
    key = pred[2][0][0]

    times = []
    tones = [
        "N",
        "C",
        "Db",
        "D",
        "Eb",
        "E",
        "F",
        "Gb",
        "G",
        "Ab",
        "A",
        "Bb",
        "B",
        "Am",
        "Bbm",
        "Bm",
        "Cm",
        "C#m",
        "Dm",
        "Ebm",
        "Em",
        "Fm",
        "F#m",
        "Gm",
        "G#m"]

    beforeKey = key[0]
    beforeTime = 0.0
    nframes = len(key)

    for i, isLast in lastone(range(1, nframes)):
        if beforeKey != key[i] or isLast:
            currentTime = i / bins_per_seconds

            if currentTime - beforeTime < minTime:
                beforeKey = key[i]
                continue

            currentKey = tones[beforeKey]

            times.append(
                [beforeTime,
                 currentTime,
                 currentKey]
            )

            beforeTime = currentTime
            beforeKey = key[i]

    return times


def convert_time(pred, bins_per_seconds, minTime=0.1):
    with open("./index.json", mode="r") as f:
        chord_index = json.load(f)

    chord = pred[1][0][0]
    bass = pred[0][0][0]

    times = []
    tones = [
        "C",
        "Db",
        "D",
        "Eb",
        "E",
        "F",
        "Gb",
        "G",
        "Ab",
        "A",
        "Bb",
        "B",
    ]

    beforeChord = [chord[0], bass[0]]
    beforeTime = 0.0
    nframes = len(chord)

    for i in range(1, nframes):
        if (beforeChord[0] != chord[i]) or (beforeChord[1] != bass[i]):
            currentTime = i / bins_per_seconds

            if currentTime - beforeTime < minTime:
                beforeChord = [chord[i], bass[i]]
                continue

            currentChord = beforeChord[0]
            currentBass = beforeChord[1]

            try:
                chord_text_ = chord_index[str(currentChord)]

                if chord_text_ == "N.C.":
                    raise ValueError()

                chordTemp = Chord(chord_text_).reconfigured()
                chordText = chordTemp.name

                if chordTemp.bass.get_interval() != (
                        currentBass - 1) and currentBass != 0:
                    chordText += "/" + tones[currentBass - 1]
                    chordText = Chord(chordText).reconfigured().name

            except ValueError as e:
                chordText = chord_index[str(currentChord)]

            times.append(
                [round(beforeTime, 3),
                 round(currentTime, 3),
                 chordText]
            )

            beforeTime = currentTime
            beforeChord = [chord[i], bass[i]]

    return times


def preprocess(path, sr=22050, mono=False):
    y, sr = librosa.load(path, sr=sr, mono=mono)
    hop_length = 512 + 32
    bins_per_second = sr / hop_length
    duration = librosa.get_duration(y)

    if len(y.shape) == 1:
        y = np.array([y, y])

    S = fft.cqt(
        y,
        sr=sr,
        n_bins=12 * 3 * 7,
        bins_per_octave=12 * 3,
        hop_length=hop_length,
        Qfactor=22.0)

    p = 8192 - (S.shape[1] % 8192)
    S_padding = np.zeros((S.shape[0], S.shape[1] + p, S.shape[2]))
    S_padding[:, :S.shape[1], :] = S
    S_padding = S_padding.transpose(1, 2, 0)
    S_padding = np.array([S_padding])

    return S_padding, bins_per_second, duration


def minor_key_to_major_key(key):
    keys = [
        "N",
        "C",
        "Db",
        "D",
        "Eb",
        "E",
        "F",
        "Gb",
        "G",
        "Ab",
        "A",
        "Bb",
        "B",
        "Am",
        "Bbm",
        "Bm",
        "Cm",
        "C#m",
        "Dm",
        "Ebm",
        "Em",
        "Fm",
        "F#m",
        "Gm",
        "G#m"]
    index = keys.index(key)
    if index >= 13:
        index = index - 12
    return keys[index]

# モデル読み込み
model = tf.keras.models.load_model("./model/chordestimation")

while True:
    file_path = input("楽曲:")
    # D&D時のダブルクオーテーション削除
    file_path = file_path.replace("\"", "")
    music_name = os.path.splitext(os.path.basename(file_path))[0]

    S, bins_per_second, duration = preprocess(file_path)
    pred = model.predict(S)
    # pred[0] ベース
    # pred[1] コード
    # pred[2] キー
    # [0][0]で最も確率の高いインデックスを取得可能
    # Example: pred[0][0][0], pred[1][0][0]
    # [1][0]でクラスごとの確率を取得可能 (ただ確信度が高いためあまり意味はないかも)

    times = convert_time(pred, bins_per_second)
    key_times = convert_time_key(pred, bins_per_second)

    accidental_modified_tiems = []
    for chord_time in times:
        chord_start_time = chord_time[0]
        modified_chord = chord_time[2]

        if modified_chord != "N.C.":
            for key_time in key_times:
                if key_time[2] == "N":
                    continue
                start_time = key_time[0]
                end_time = key_time[1]

                if chord_start_time >= start_time and chord_start_time <= end_time:
                    modified_chord = Chord(modified_chord).modified_accidentals(minor_key_to_major_key(key_time[2])).name
                    break
        accidental_modified_tiems.append([
            chord_time[0],
            chord_time[1],
            modified_chord
        ])


    with open("./label/{}.txt".format(music_name), mode="w") as f:
        for t in accidental_modified_tiems:
            f.write("{}	{}	{}\n".format(t[0], t[1], t[2]))

    with open("./keys/{}.txt".format(music_name), mode="w") as f:
        for t in key_times:
            f.write("{}	{}	{}\n".format(t[0], t[1], t[2]))

    print("complete {}".format(music_name))
