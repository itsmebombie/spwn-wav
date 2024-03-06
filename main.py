
MODIFY_PITCH = False
SAMPLE_LEN_MS = 20
SOUND_NAME = "elementary.wav"
USE_SONGS = True

import json
import os
import librosa
import numpy as np
from numba import jit
from pydub import AudioSegment

def load_sound_effects(input_folder):
    audio_data = [{} for _ in range(-12, 13)]


    sample_rate = int(44100)  # Assuming all files are 44100Hz
    sample_rate_div_1000 = sample_rate / 1000
    num_samples = int(sample_rate_div_1000 * SAMPLE_LEN_MS)

    i = 0
    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename[0] != "s" or filename[-4:] != ".ogg": continue
        i+=1
        # if i == 100: break
        print(filename)

        file_path = os.path.join(input_folder, filename)

        # Load the audio file using pydub
        try: audio = AudioSegment.from_ogg(file_path)
        except: continue
        audio = audio.set_sample_width(1).set_channels(1) # 8 bit, mono

        audio_len = len(audio)
        
        # skip if the audio is shorter than n ms, the root cause of some errors
        if audio_len < SAMPLE_LEN_MS: continue

        sound_id = int(filename[1:-4])

        # Extract all samples in intervals of n ms
        samples = np.array(audio.get_array_of_samples(), dtype=np.int8)
        r = range(-12, 13) if MODIFY_PITCH else [0] # wtf

        # this is terrible af i cant release this
        for semitones in r:
            pitch_audio = audio_data[semitones+12]
            pitch_audio[sound_id] = []

            # convert each sfx to float to shift the pitch
            # then convert them back to int8
            arr = samples
            if MODIFY_PITCH and semitones != 0:
                arr = (librosa.effects.pitch_shift(
                    librosa.util.buf_to_float(samples), # removed .copy() from `samples` in `.buf_to_float(samples)`, hope nothing breaks
                    sr=44100,
                    n_steps=semitones,
                ) * 127).astype(np.int8)

            for ms in range(0, audio_len-SAMPLE_LEN_MS, SAMPLE_LEN_MS):
                sample_idx = int(ms * sample_rate_div_1000)
                pitch_audio[sound_id].append(arr[sample_idx : sample_idx + num_samples])

            pitch_audio[sound_id] = np.array(pitch_audio[sound_id], dtype=np.int8)

    to_return = [
        (list(i.values()), list(i.keys())) 
        if len(i) > 0 else ([np.array([[-1]], dtype=np.int8)],[-1])
        for i in audio_data
    ]

    return to_return


@jit(nopython=True)
def euclidean_distance(arr1: np.ndarray, arr2: np.ndarray) -> np.float64:
    return np.sum((arr1 - arr2)**2)**0.5 # np.sum returns int


@jit(nopython=True)
def find_closest(array: np.ndarray, arr_pitches: np.ndarray) -> tuple:
    min_distance = 9e9
    closest_array = None

    for arr_l1_idx in range(len(arr_pitches)):
        arr_l1 = arr_pitches[arr_l1_idx]
        for arr_l2_idx in range(len(arr_l1)):
            arr_l2 = arr_l1[arr_l2_idx]

            distance = euclidean_distance(array, arr_l2)

            if distance < min_distance:
                min_distance = distance
                closest_array = (arr_l1_idx, arr_l2_idx, distance)
                # print(distance, closest_array)

    # if closest_array == None: raise Exception("Wtf?????/") # just code good bro :rofl:
    
    return closest_array


def min_distance(ls):
    return min(ls, key=lambda x: x[3])


def match_sound(array_of_sounds, path):
    audio = AudioSegment.from_file(path)

    sample_rate = int(44100)
    num_samples = int(sample_rate / 1000 * SAMPLE_LEN_MS)
    # print(weight_pattern)

    cumdist = 0

    audio = audio.set_frame_rate(sample_rate).set_sample_width(1).set_channels(1)
    audio_len = len(audio)
    print(audio.frame_rate)
    print(audio_len)
    
    indexes = [] # [(soundname, timestamp, pitch), ...]
    samples = tuple(audio.get_array_of_samples())
    for ms in range(0, audio_len-SAMPLE_LEN_MS, SAMPLE_LEN_MS):
        sample_idx = int(ms/1000 * sample_rate)
        original_array = np.array(samples[sample_idx : sample_idx + num_samples], dtype=np.int8)

        results = []
        for semitones in (range(len(array_of_sounds)) if MODIFY_PITCH else [12]):
            arr_pitches = array_of_sounds[semitones][0]
            sfx_ids = array_of_sounds[semitones][1]

            arr_l1_idx, timestamp, dist = find_closest(original_array, arr_pitches)
            pitch = (semitones-12)*MODIFY_PITCH

            results.append((sfx_ids[arr_l1_idx], timestamp, pitch, dist, arr_l1_idx))
        sfx_id, timestamp, pitch, dist, arr_l1_idx = min_distance(results)
        
        print(f"time: {ms/1000:.3f}s  dist: {dist:.2f}  {(sfx_id, timestamp, pitch)}")
        cumdist += dist

        indexes.append((sfx_id, timestamp*SAMPLE_LEN_MS, pitch, arr_l1_idx))
    
    print(f"\navg dist: {cumdist / ((audio_len-SAMPLE_LEN_MS) / SAMPLE_LEN_MS):.3f}\ntime elapsed")

    with open("./output.json", 'w') as f:
        f.write(str(indexes).replace("(", "[").replace(")", "]").replace(" ", ""))
    print(indexes[0])
    print("playing sound")
    audio_output = AudioSegment(
        data=bytes(np.array([
            array_of_sounds[i[2]+12][0][i[3]][i[1]//SAMPLE_LEN_MS]
            for i in indexes
        ], dtype=np.int8)),
        sample_width=1,
        frame_rate=sample_rate,
        channels=1
    )
    audio_output.export("out.wav", format="wav")


if __name__ == "__main__":
    localappdata = os.getenv('LOCALAPPDATA')
    if localappdata is None:
        print("LocalAppdata is none!")
        raise Exception("LocalAppData is None!")
    input_folder_path = os.path.join(localappdata, "GeometryDash")

    array = load_sound_effects(input_folder_path)
    match_sound(array, SOUND_NAME)
