# https://www.youtube.com/watch?v=SW0YGA9d8y8
# https://github.com/microsoft/pylance-release/blob/main/TROUBLESHOOTING.md#unresolved-import-warnings
# source ./soundvenv/bin/activate
# to commit -> move pain.py to rep folder

#region Imports
import os

import IPython.display as ipd
import pandas as pd
import numpy as np
from numpy import pad

import librosa
import librosa.display
import essentia 
import essentia.standard as es


from sklearn.neighbors import KNeighborsClassifier
import sklearn as skl 
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm, sklearn.model_selection, sklearn.metrics 
import matplotlib.pyplot as plt
import seaborn as sns

#import tensorflow as tf
import utils
#endregion

# Data Frame Creation
data = pd.read_csv('data/fma_metadata/raw_tracks.csv')
# data.info()
# print(data.isnull().sum())



# Data Cleaning and Feature Engineering
def preprocess_data(df):
    # remove irrelevant columns
    df.drop(columns=['album_id', 'album_title', 'album_url', 'artist_id', 'artist_name', 'artist_url', 'artist_website', 'license_image_file', 'license_image_file_large', 'license_parent_id', 'license_title', 'license_url', 'tags', 'track_bit_rate', 'track_comments', 'track_composer', 'track_copyright_c', 'track_copyright_p', 'track_date_created', 'track_date_recorded', 'track_disc_number', 'track_duration', 'track_explicit', 'track_explicit_notes', 'track_favorites', 'track_image_file', 'track_information', 'track_instrumental', 'track_interest', 'track_language_code', 'track_listens', 'track_lyricist', 'track_number', 'track_publisher', 'track_title', 'track_url'], inplace=True)

    # remove empty genre columns
    df.dropna(subset=['track_genres'], inplace=True)
    
    
    return df

preprocess_data(data)
#print(data)
# data.info()
# print(data.isnull().sum())

# Feature Engineering (add essentia stuff)

#region track id to path mapping


def get_fma_small_ids(base_dir):
    valid_ids = set()
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.mp3'):
                try:
                    tid = int(file.replace('.mp3', ''))
                    valid_ids.add(tid)
                except ValueError:
                    continue
    return valid_ids
base_dir = 'data/fma_small'
valid_track_ids = get_fma_small_ids(base_dir)


# first, check which song ids we process
track_ids = data['track_id'].astype(int)
genres = data['track_genres']
#print(genres)

# Check if track is specific genre (e.g., Pop)
# Define genre ID mapping
genre_ids = {
    '10': 'Pop',
    '12': 'Rock',
    '31': 'Metal',
    '25': 'Punk'
}

# Set limits for each genre TODO: change this to 80 songs per genre
genre_limits = {
    'Pop': 1,
    'Rock': 1,
    'Metal': 1,
    'Punk': 1
}
genre_counts = {
    'Pop': 0,
    'Rock': 0,
    'Metal': 0,
    'Punk': 0
}

track_id_to_genre = {}

# Iterate through DataFrame to find tracks by genre
for idx, row in data.iterrows():
    tid = int(row['track_id'])
    if tid not in valid_track_ids:
        continue  # Skip if track ID is not in valid set
    genre_str = row['track_genres']
    for genre_id, genre_name in genre_ids.items():
        if f'"genre_id": "{genre_id}"' in genre_str or f"'genre_id': '{genre_id}'" in genre_str:
            if genre_counts[genre_name] < genre_limits[genre_name]:
                track_id_to_genre[int(row['track_id'])] = genre_name
                genre_counts[genre_name] += 1
            break  # Only count the track once

#Test print
#print(f"Collected {len(track_id_to_genre)} tracks:")
#print("track_id_to_genre:", track_id_to_genre)
#for tid, path in track_id_to_genre.items():
#    print(f"Track ID: {tid}, Genre: {track_id_to_genre[tid]}")


# Build path map from matched track_ids TODO: fix, i need all 5 songs for each genre, not just the first 5
track_id_to_path = {}

def collect_track_paths(base_dir, track_id_to_genre):
    track_id_to_path = {}
    
    # Convert track IDs to a set for fast lookup
    target_ids = set(track_id_to_genre.keys())
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.mp3'):
                try:
                    # Extract track ID from filename (e.g., '000123.mp3' -> 123)
                    track_id = int(file.replace('.mp3', ''))
                    
                    if track_id in target_ids:
                        full_path = os.path.join(root, file)
                        track_id_to_path[track_id] = full_path
                except ValueError:
                    # In case the filename is not a valid int
                    continue
    
   # print(f"Found paths for {len(track_id_to_path)} out of {len(track_id_to_genre)} tracks.")
    return track_id_to_path

track_id_to_path = collect_track_paths(base_dir, track_id_to_genre)
# Test print
print(f"Collected {len(track_id_to_path)} tracks:")
missing_ids = set(track_id_to_genre.keys()) - set(track_id_to_path.keys())
print("Missing track IDs (no .mp3 file found):", missing_ids)
#for tid, path in track_id_to_path.items():
  #  print(f"Track ID: {tid}, Genre: {track_id_to_genre[tid]}, Path: {path}")
#print(len(track_id_to_path))
#print(track_id_to_path)

#endregion

#region load audio files

#load mono audio files -> beat tracking, tempo estimation, onset detection, rhythmic analysis, uniform preprocessing
mono_loaded_audio = {}

for track_id, filepath in track_id_to_path.items():
    if not os.path.exists(filepath) or os.path.getsize(filepath) < 1024:
        print(f"Skipping invalid or empty file: {filepath}")
        continue
    try:
        audio = es.MonoLoader(filename=filepath)()
        mono_loaded_audio[track_id] = audio
    except Exception as e:
        print(f"Error loading {filepath} (track_id {track_id}): {e}")
        continue
#test print
#print(len(mono_loaded_audio))
#print(mono_loaded_audio)

#load eqloud audio files -> pitch estimation, music transcription, chord detection, melodic analysis
eqloud_loaded_audio = {}

for track_id, filepath in track_id_to_path.items():
    if not os.path.exists(filepath) or os.path.getsize(filepath) < 1024:
        print(f"Skipping invalid or empty file: {filepath}")
        continue
    try:
        audio = es.EqloudLoader(filename=filepath, sampleRate=44100)()
        eqloud_loaded_audio[track_id] = audio
    except Exception as e:
        print(f"Error loading {filepath} (track_id {track_id}): {e}")
        continue
#test print
#print(len(eqloud_loaded_audio))
#print(eqloud_loaded_audio)
#endregion

"""
=====================================
|        Melody Features          |
=====================================
"""
#region Yin Pitch Extraction -> Estimates the fundamental frequency given the frame of a monophonic music signal

def extract_pitch_yin(eqloud_loaded_audio, frame_size=2048, hop_size=1024, sample_rate=44100):

    yin_processed_data_pitch = {}

    window = es.Windowing(type='hann') #Using Hann window 
    yin = es.PitchYin(frameSize=frame_size)

    total_tracks_len = len(eqloud_loaded_audio)
    for idx, (track_id, audio) in enumerate(eqloud_loaded_audio.items(), start=1):
        pitches = []
        confidences = []
        times = []

        # Process audio in frames with feedback
        duration_sec = len(audio) / sample_rate
        num_frames = (len(audio) - frame_size) // hop_size
        print(f"[{idx}/{total_tracks_len}] [Yin] Processing track {track_id} ({duration_sec:.1f}s, ~{num_frames} frames)")

        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            if len(frame) < frame_size:
                break

            pitch, confidence = yin(window(frame))
            pitches.append(pitch)
            confidences.append(confidence)
            times.append(i / sample_rate)

            # Log progress every 10 seconds of audio TODO: this doesnt work
            if i % (sample_rate * 10) == 0 and i != 0:
                print(f"    ↳ Processed {i / sample_rate:.1f}s")

        yin_processed_data_pitch[track_id] = {
            'yin_pitch_values': pitches,
            'yin_pitch_confidence': confidences,
            'yin_pitch_times': times
        }

    return yin_processed_data_pitch
#Call the function to extract pitch using Yin
yin_processed_data_pitch = extract_pitch_yin(eqloud_loaded_audio)
#pitch test print
#print(yin_processed_data_pitch)

#endregion

#region Yin Pitch added to original data
# Add pitch mean as a new feature to the original data DataFrame
data['track_id'] = data['track_id'].astype(int)

# Prepare pitch feature dataframe
yin_pitch_features = []

for track_id, yin_pitch_data in yin_processed_data_pitch.items():
    yin_pitch_values = np.array(yin_pitch_data['yin_pitch_values'])
    yin_confidences = np.array(yin_pitch_data['yin_pitch_confidence'])
    
    # Filter out 0 Hz pitches (no pitch detected)
    valid_yin_pitches = yin_pitch_values[yin_pitch_values > 0]
#TODO: check yin algorithm output types
    
    
    if (len(valid_yin_pitches) > 0):

        yin_mean_confidence_threshold = np.mean(yin_confidences)

       # yin_raw_pitches = valid_yin_pitches
        yin_mean_pitch = np.mean(valid_yin_pitches)
        yin_median_pitch = np.median(valid_yin_pitches)
        yin_std_pitch = np.std(valid_yin_pitches)
    else:
        yin_mean_confidence_threshold = np.nan

        #yin_raw_pitches = np.nan
        yin_mean_pitch = np.nan
        yin_median_pitch = np.nan
        yin_std_pitch = np.nan

    yin_pitch_features.append({
        'track_id': track_id,
       # 'yin_raw_pitches': yin_raw_pitches,
        'yin_pitch_mean': yin_mean_pitch,
        'yin_pitch_median': yin_median_pitch,
        'yin_pitch_std': yin_std_pitch,
        'yin_confidence_threshold': yin_mean_confidence_threshold
    })

# Call the function to create DataFrame and merge pitch_yin with original data
pitch_df = pd.DataFrame(yin_pitch_features)
# Keep only the rows with track_ids in processed_data_pitch
data = data[data['track_id'].isin(yin_processed_data_pitch.keys())]
data = pd.merge(data, pitch_df, on='track_id', how='left')


# Test print
#cols = ['track_id','yin_pitch_median','yin_confidence_threshold']
#print(data[cols])
#print(data)
#print(len(data))
#print(data[['track_id', 'pitch_mean', 'pitch_median', 'pitch_std']].head())
#print("Pitch entries:", list(processed_data_pitch.keys())[:10])
#print("DataFrame track_ids:", data['track_id'].head(10).tolist())
#print(data.info())
#endregion

#region Melody / Predominant pitch Extraction -> Estimates the fundamental frequency of the predominant melody from polyphonic music signals using the MELODIA algorithm
def extract_pitch_melodia(eqloud_loaded_audio, frame_size=2048, hop_size=128, sample_rate=44100): #TODO: justify the hop size

    processed_data_pitch_melodia = {}


    melodia = es.PredominantPitchMelodia(frameSize=frame_size, hopSize=hop_size)
    

    
    for idx, (track_id, audio) in enumerate(eqloud_loaded_audio.items(), start=1):
        duration_sec = len(audio) / sample_rate
        num_frames = (len(audio) - frame_size) // hop_size
        total_tracks = len(eqloud_loaded_audio)
        print(f"[{idx}/{total_tracks}] [Melodia] Processing track {track_id} ({duration_sec:.1f}s, ~{num_frames} frames)")

        try:
            melodia_pitch_values, melodia_pitch_confidence = melodia(audio)
            melodia_pitch_times = np.arange(len(melodia_pitch_values)) * hop_size / sample_rate

            processed_data_pitch_melodia[track_id] = {
                'melodia_pitch_values': melodia_pitch_values,
                'melodia_pitch_confidence': melodia_pitch_confidence,
                'melodia_pitch_times': melodia_pitch_times
            }

        except Exception as e:
            print(f"Error processing track {track_id}: {e}")
            continue

    return processed_data_pitch_melodia

#Call the function to extract pitch using Melodia
processed_data_pitch_melodia = extract_pitch_melodia(eqloud_loaded_audio)

#Test print
#print(processed_data_pitch_melodia)
#endregion

#region Melodia Pitch added to original data #TODO: merge melodia results with original data


#endregion

#region MIDI Note Number (MNN) Statistics #TODO: check if this is correct
def extract_mnn_stats(processed_data_pitch_melodia):
    """
    Given a dict
      { track_id: {
          'melodia_pitch_values': np.array([...Hz...]),
          'melodia_pitch_confidence': np.array([...]),
          'melodia_pitch_times': np.array([...])
        }, ... }
    convert each non-zero Hz pitch into MIDI note numbers,
    round to the nearest semitone, and compute mean/median/std.
    Returns a dict:
      { track_id: {'mnn_mean':…, 'mnn_median':…, 'mnn_std':…}, … }
    """
    processed_data_mnn = {}

    for track_id, pitch_data in processed_data_pitch_melodia.items():
        freqs = np.array(pitch_data['melodia_pitch_values'])
        # Keep only frames with a detected pitch > 0 Hz
        freqs = freqs[freqs > 0]

        if freqs.size > 0:
            # Convert Hz -> MIDI note numbers (float)
            mnn = 69 + 12 * np.log2(freqs / 440.0)
            # Round to nearest integer semitone
            mnn_int = np.round(mnn).astype(int)

            processed_data_mnn[track_id] = {
                'mnn_mean':   np.mean(mnn_int),
                'mnn_median': np.median(mnn_int),
                'mnn_std':    np.std(mnn_int)
            }
        else:
            # No valid pitches detected
            processed_data_mnn[track_id] = {
                'mnn_mean':   np.nan,
                'mnn_median': np.nan,
                'mnn_std':    np.nan
            }

    return processed_data_mnn

# Call it on your Melodia output:
processed_data_mnn = extract_mnn_stats(processed_data_pitch_melodia)

# Example: turn into a DataFrame and merge back into `data`
mnn_features = [
    dict(track_id=tid, **stats)
    for tid, stats in processed_data_mnn.items()
]
mnn_df = pd.DataFrame(mnn_features)

# Keep only processed tracks
data = data[data['track_id'].isin(mnn_df['track_id'])]

# Merge
data = data.merge(mnn_df, on='track_id', how='left')

# Test print
#print(data[['track_id', 'mnn_mean', 'mnn_median', 'mnn_std']].head(10))
#print(data.info())
#print(data)
#endregion

"""
=====================================
|        Harmmony Features          |
=====================================
"""
#region Inharmonicity extraction
def extract_inharmonicity(mono_loaded_audio, frame_size=2048, hop_size=1024, sample_rate=44100):
    processed_data_inharmonicity = {}

    window = es.Windowing(type='hann')
    spectrum = es.Spectrum()
    spectral_peaks = es.SpectralPeaks(magnitudeThreshold=0.0001, maxPeaks=100)
    inharmonicity_extractor = es.Inharmonicity()

    total_tracks = len(mono_loaded_audio)

    for idx, (track_id, audio) in enumerate(mono_loaded_audio.items(), start=1):
        duration_sec = len(audio) / sample_rate
        num_frames = (len(audio) - frame_size) // hop_size
        print(f"[{idx}/{total_tracks}] [Inharmonicity] Processing track {track_id} ({duration_sec:.1f}s, ~{num_frames} frames)")

        inharmonicity_values = []

        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            if len(frame) < frame_size:
                break

            windowed_frame = window(frame)
            spec = spectrum(windowed_frame)
            frequencies, magnitudes = spectral_peaks(spec)
            # Skip frames with no valid peaks or 0 Hz fundamental
            if len(frequencies) == 0 or frequencies[0] <= 0:
                 continue
            inh = inharmonicity_extractor(frequencies, magnitudes)
            inharmonicity_values.append(inh)

        # Compute statistics
        if inharmonicity_values:
            mean_inharmonicity = np.mean(inharmonicity_values)
            median_inharmonicity = np.median(inharmonicity_values)
            std_inharmonicity = np.std(inharmonicity_values)
        else:
            mean_inharmonicity = np.nan
            median_inharmonicity = np.nan
            std_inharmonicity = np.nan

        processed_data_inharmonicity[track_id] = {
            'inharmonicity_values': inharmonicity_values,
            'inharmonicity_mean': mean_inharmonicity,
            'inharmonicity_median': median_inharmonicity,
            'inharmonicity_std': std_inharmonicity
        }

    return processed_data_inharmonicity
# Call the function to extract inharmonicity
processed_data_inharmonicity = extract_inharmonicity(mono_loaded_audio)
# Test print
#print(processed_data_inharmonicity)
#endregion

#region Chromagram extraction #TODO: check if this works/name variables better




def extract_chromagram(mono_loaded_audio,
                       frame_size=2048,
                       hop_size=1024,
                       sample_rate=44100):
    processed_data_chromogram = {}

    window   = es.Windowing(type='hann',   size=frame_size)
    spectrum = es.Spectrum()
    chroma   = es.Chromagram()  # expects CQT‐length spectrum

    total = len(mono_loaded_audio)
    for idx, (track_id, audio) in enumerate(mono_loaded_audio.items(), start=1):
        print(f"[{idx}/{total}] [Chroma] Processing track {track_id}")

        frames = []
        for i in range(0, len(audio)-frame_size, hop_size):
            frame = window(audio[i:i+frame_size])
            spec  = spectrum(frame)            # length = 1025
            # pad to 32768
            spec_padded = pad(spec, (0, 32768 - spec.shape[0]), mode='constant')
            c = chroma(spec_padded)            
            frames.append(c)

        processed_data_chromogram[track_id] = np.vstack(frames) if frames else np.empty((0,12))

    return processed_data_chromogram


# Call it
processed_data_chromogram = extract_chromagram(mono_loaded_audio)
# Test print
#print(processed_data_chromogram)

#endregion

#region HPCP Extraction

def extract_hpcp(mono_loaded_audio,
                 frame_size=2048,
                 hop_size=1024,
                 sample_rate=44100,
                 bins_per_octave=12,
                 min_freq=50,
                 max_freq=5000):
    """
    Returns { track_id: np.array(n_frames,12) } using HPCP (a chromagram).
    """
    processed_data_hpcp = {}

    window       = es.Windowing(type='hann', size=frame_size)
    spectrum     = es.Spectrum()
    spectral_peaks = es.SpectralPeaks(orderBy='magnitude', magnitudeThreshold=0.01)
    hpcp         = es.HPCP(size=bins_per_octave,
                           minFrequency=min_freq,
                           maxFrequency=max_freq,
                           referenceFrequency=440.0)

    total = len(mono_loaded_audio)
    for idx, (tid, audio) in enumerate(mono_loaded_audio.items(), start=1):
        print(f"[{idx}/{total}] [HPCP] Processing track {tid}")
        frames = []
        for i in range(0, len(audio)-frame_size, hop_size):
            frame = window(audio[i:i+frame_size])
            spec  = spectrum(frame)
            freqs, mags = spectral_peaks(spec)
            c = hpcp(freqs, mags)  # yields 12-bin chroma
            frames.append(c)

        processed_data_hpcp[tid] = np.vstack(frames) if frames else np.empty((0, bins_per_octave))

    return processed_data_hpcp

# Call it
processed_data_hpcp = extract_hpcp(mono_loaded_audio)
# Test print
#print(processed_data_hpcp)
#endregion

#region Key Extraction
def extract_key(eqloud_loaded_audio, sample_rate=44100):
    processed_data_key = {}
    
    key_extractor = es.KeyExtractor(sampleRate=sample_rate)

    total_tracks = len(eqloud_loaded_audio)

    for idx, (track_id, audio) in enumerate(eqloud_loaded_audio.items(), start=1):
        duration_sec = len(audio) / sample_rate
        print(f"[{idx}/{total_tracks}] [Key] Processing track {track_id} ({duration_sec:.1f}s)")

        try:
            key, scale, strength = key_extractor(audio)

            processed_data_key[track_id] = {
                'key': key,
                'scale': scale,
                'strength': strength
            }

        except Exception as e:
            print(f"Error processing track {track_id}: {e}")
            continue

    return processed_data_key

# Call the function to extract key
processed_data_key = extract_key(mono_loaded_audio)
# Test print
#print(processed_data_key)
#endregion
"""
=====================================
|        Rythm Features          |
=====================================
"""
#region BPM Extraction
def extract_bpm(mono_loaded_audio, sample_rate=44100):
    """
    Extract BPM using RhythmExtractor2013 from mono audio.
    Returns a dictionary with BPM per track.
    """
    processed_data_bpm = {}
    rhythm_extractor = es.RhythmExtractor2013(method="multifeature")

    total_tracks = len(mono_loaded_audio)

    for idx, (track_id, audio) in enumerate(mono_loaded_audio.items(), start=1):
        duration_sec = len(audio) / sample_rate
        print(f"[{idx}/{total_tracks}] [BPM] Processing track {track_id} ({duration_sec:.1f}s)")

        try:
            bpm, ticks, _, _, _ = rhythm_extractor(audio)
        except Exception as e:
            print(f"Error processing BPM for track {track_id}: {e}")
            bpm = None
            ticks = []

        processed_data_bpm[track_id] = {
            'bpm': bpm,
            'ticks': ticks
        }

    return processed_data_bpm
# Call the function to extract BPM
processed_data_bpm = extract_bpm(mono_loaded_audio)
# Test print
#print(processed_data_bpm)
#endregion

#region Onset Extraction
def extract_onset_rate(mono_loaded_audio, sample_rate=44100):
    """
    Extract onset rate (number of onsets per second) from mono audio tracks.
    Returns a dictionary with onset rate per track.
    """
    processed_data_onset_rate = {}

    onset_rate_algo = es.OnsetRate()

    total_tracks = len(mono_loaded_audio)

    for idx, (track_id, audio) in enumerate(mono_loaded_audio.items(), start=1):
        duration_sec = len(audio) / sample_rate
        print(f"[{idx}/{total_tracks}] [Onset Rate] Processing track {track_id} ({duration_sec:.1f}s)")

        try:
            onset_rate = onset_rate_algo(audio)
            processed_data_onset_rate[track_id] = {
                'onset_rate': onset_rate
            }
        except Exception as e:
            print(f"Error processing onset rate for track {track_id}: {e}")
            processed_data_onset_rate[track_id] = {
                'onset_rate': None
            }

    return processed_data_onset_rate
# Call the function to extract onset rate

processed_data_onset_rate = extract_onset_rate(mono_loaded_audio)
# Test print
#print(processed_data_onset_rate)
#endregion

#region Beat Histogram Extraction TODO: fucking fix this
def extract_beat_histogram(mono_loaded_audio, frame_size=1024, hop_size=512, sample_rate=44100):
    processed_data_beat_histogram = {}

    windowing = es.Windowing(type='hann')
    spectrum = es.Spectrum()
    cartesian_to_polar = es.CartesianToPolar()
    onset_detection = es.OnsetDetection(method='complex')
    bpm_histogram_algo = es.BpmHistogram()

    total_tracks = len(mono_loaded_audio)

    for idx, (track_id, audio) in enumerate(mono_loaded_audio.items(), start=1):
        duration_sec = len(audio) / sample_rate
        print(f"[{idx}/{total_tracks}] [Beat Histogram] Processing track {track_id} ({duration_sec:.1f}s)")

        novelty_curve = []

        try:
            for i in range(0, len(audio) - frame_size, hop_size):
                frame = audio[i:i + frame_size]
                windowed = windowing(frame)
                spec = spectrum(windowed)
                mag, phase = cartesian_to_polar(spec)
                onset_val = onset_detection(mag, phase)
                novelty_curve.append(float(onset_val))

            if len(novelty_curve) == 0:
                raise ValueError("Novelty curve (onset function) is empty.")

            # Convert to plain Python float list
            novelty_curve_floats = [float(x) for x in novelty_curve]

            histogram, bpm1, bpm2 = bpm_histogram_algo(novelty_curve_floats)

            processed_data_beat_histogram[track_id] = {
                'beat_histogram': histogram,
                'bpm_peak_1': bpm1,
                'bpm_peak_2': bpm2
            }

        except Exception as e:
            print(f"Error processing beat histogram for track {track_id}: {e}")
            processed_data_beat_histogram[track_id] = {
                'beat_histogram': None,
                'bpm_peak_1': None,
                'bpm_peak_2': None
            }

    return processed_data_beat_histogram
# Call the function to extract onset rate

#processed_data_beat_histogram = extract_beat_histogram(mono_loaded_audio)
#Test print
#print(processed_data_beat_histogram)
#endregion

"""
=====================================
|        Dynamic Features          |
=====================================
"""
#region Loudness Mean Extraction
def extract_loudness_mean(mono_loaded_audio, sample_rate=44100, frame_size=2048, hop_size=1024):
    """
    Extract mean loudness per track using Essentia Loudness algorithm.
    Returns dict {track_id: mean loudness}
    """
    loudness_algo = es.Loudness()

    processed_loudness = {}
    total_tracks = len(mono_loaded_audio)

    for idx, (track_id, audio) in enumerate(mono_loaded_audio.items(), start=1):
        print(f"[{idx}/{total_tracks}] [Loudness] Processing track {track_id}")

        try:
            loudness_values = []
            for i in range(0, len(audio) - frame_size, hop_size):
                frame = audio[i:i+frame_size]
                loudness_val = loudness_algo(frame)
                loudness_values.append(loudness_val)

            mean_loudness = np.mean(loudness_values) if loudness_values else None
            processed_loudness[track_id] = mean_loudness

        except Exception as e:
            print(f"Error processing loudness for track {track_id}: {e}")
            processed_loudness[track_id] = None

    return processed_loudness
# Call the function to extract loudness mean
processed_loudness_mean = extract_loudness_mean(mono_loaded_audio)
# Test print
#print(processed_loudness_mean)
#endregion

#region Dynamic Range Extraction
def extract_dynamic_range(mono_loaded_audio, frame_size=2048, hop_size=1024):
    """
    Extract dynamic range (max RMS - min RMS) per track.
    Returns dict {track_id: dynamic_range}
    """
    rms_algo = es.RMS()

    processed_dynamic_range = {}
    total_tracks = len(mono_loaded_audio)

    for idx, (track_id, audio) in enumerate(mono_loaded_audio.items(), start=1):
        print(f"[{idx}/{total_tracks}] [Dynamic Range] Processing track {track_id}")

        try:
            rms_values = []
            for i in range(0, len(audio) - frame_size, hop_size):
                frame = audio[i:i+frame_size]
                rms_val = rms_algo(frame)
                rms_values.append(rms_val)

            if rms_values:
                dynamic_range = max(rms_values) - min(rms_values)
            else:
                dynamic_range = None

            processed_dynamic_range[track_id] = dynamic_range

        except Exception as e:
            print(f"Error processing dynamic range for track {track_id}: {e}")
            processed_dynamic_range[track_id] = None

    return processed_dynamic_range
# Call the function to extract dynamic range
processed_dynamic_range = extract_dynamic_range(mono_loaded_audio)
# Test print
#print(processed_dynamic_range)
#endregion

#region RMS Energy STD Extraction
def extract_rms_energy_std(mono_loaded_audio, frame_size=2048, hop_size=1024):
    """
    Extract standard deviation of RMS energy per track.
    Returns dict {track_id: rms_std}
    """
    rms_algo = es.RMS()

    processed_rms_std = {}
    total_tracks = len(mono_loaded_audio)

    for idx, (track_id, audio) in enumerate(mono_loaded_audio.items(), start=1):
        print(f"[{idx}/{total_tracks}] [RMS Energy] Processing track {track_id}")

        try:
            rms_values = []
            for i in range(0, len(audio) - frame_size, hop_size):
                frame = audio[i:i+frame_size]
                rms_val = rms_algo(frame)
                rms_values.append(rms_val)

            rms_std = np.std(rms_values) if rms_values else None
            processed_rms_std[track_id] = rms_std

        except Exception as e:
            print(f"Error processing RMS std for track {track_id}: {e}")
            processed_rms_std[track_id] = None

    return processed_rms_std
# Call the function to extract RMS energy standard deviation
processed_rms_energy_std = extract_rms_energy_std(mono_loaded_audio)
# Test print
#print(processed_rms_energy_std)
#endregion


"""
=====================================
|        Tone Colour Features          |
=====================================
"""

"""
=====================================
|        Form Features          |
=====================================
"""


"""
=====================================
|        High-Level Features          |
=====================================
"""

# Create Features / Target Variables (Make flashcards)



# ML Processing



# Hyperparameter Tuning



# Predictions and Evaluation



# Plot

