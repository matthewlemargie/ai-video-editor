import ffmpeg
import numpy as np
import io
import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def extract_audio_np(video_path, target_sr=16000):
    out, _ = (
        ffmpeg.input(video_path)
        .output("pipe:", format="wav", ac=1, ar=str(target_sr))
        .run(capture_stdout=True, capture_stderr=True)
    )
    audio, sr = sf.read(io.BytesIO(out))
    assert sr == target_sr
    return audio, sr


def cosine_similarity(a, b):
    return 1 - cdist([a], [b], metric="cosine")[0][0]


def match_speakers_to_profiles(chunk_embeds, global_profiles, threshold=0.9):
    speaker_map = {}
    for i, embed in enumerate(chunk_embeds):
        best_match = None
        best_sim = -1
        for global_id, profile in global_profiles.items():
            sim = cosine_similarity(embed, profile)
            if sim > best_sim and sim > threshold:
                best_sim = sim
                best_match = global_id
        if best_match is not None:
            speaker_map[i] = best_match
        else:
            new_id = len(global_profiles)
            global_profiles[new_id] = embed
            speaker_map[i] = new_id
    return speaker_map


def split_audio(wav, sr, chunk_duration=60):
    chunk_len = chunk_duration * sr
    return [wav[i:i + chunk_len] for i in range(0, len(wav), chunk_len)]


def diarize(video_path, n_speakers=2):
    wav, sr = extract_audio_np(video_path)
    encoder = VoiceEncoder()

    chunks = split_audio(wav, sr, chunk_duration=60)
    all_embeds = None
    all_wav_splits = []
    offset = 0

    for chunk in chunks:
        _, cont_embeds, wav_splits = encoder.embed_utterance(
            chunk, return_partials=True, rate=8
            )

        if all_embeds is None:
            all_embeds = cont_embeds
        else:
            all_embeds = np.vstack([all_embeds, cont_embeds])

        all_wav_splits.extend([
            (slc.start / sr + offset, slc.stop / sr + offset) for slc in wav_splits
        ])

        offset += len(chunk) / sr

    labels = AgglomerativeClustering(n_clusters=n_speakers).fit_predict(all_embeds)

    # Get average embeddings per actual label
    speaker_embeds = {}
    for label in set(labels):
        idx = np.where(labels == label)[0]
        speaker_embeds[label] = np.mean(all_embeds[idx], axis=0)

    # Group into segments
    segments = []
    current_speaker = labels[0]
    start_time = all_wav_splits[0][0]
    for i in range(1, len(labels)):
        if labels[i] != current_speaker:
            end_time = all_wav_splits[i][0]
            segments.append((current_speaker, start_time, end_time))
            current_speaker = labels[i]
            start_time = all_wav_splits[i][0]

    segments.append((current_speaker, start_time, all_wav_splits[-1][1]))

    return segments


def diarize_chunk(wav_chunk, sr, encoder, n_speakers=2, rate=16):
    _, cont_embeds, wav_splits = encoder.embed_utterance(
        wav_chunk, return_partials=True, rate=rate
    )
    print(type(cont_embeds), ", ", type(wav_splits))
    if len(cont_embeds) < 2:
        return [], [], [], {}

    split_times = [(slc.start / sr, slc.stop / sr) for slc in wav_splits]
    labels = AgglomerativeClustering(n_clusters=n_speakers).fit_predict(cont_embeds)

    # Get average embeddings per actual label
    speaker_embeds = {}
    for label in set(labels):
        idx = np.where(labels == label)[0]
        speaker_embeds[label] = np.mean(cont_embeds[idx], axis=0)

    segments = []
    current_speaker = labels[0]
    start_time = split_times[0][0]
    for i in range(1, len(labels)):
        if labels[i] != current_speaker:
            end_time = split_times[i][0]
            segments.append((current_speaker, start_time, end_time))
            current_speaker = labels[i]
            start_time = split_times[i][0]
    segments.append((current_speaker, start_time, split_times[-1][1]))

    return segments, labels, [s for s, _ in split_times], speaker_embeds


def old_diarize(video_path, n_speakers=2):
    wav, sr = extract_audio_np(video_path)
    encoder = VoiceEncoder(window_secs=0.5, overlap=0.25)

    chunks = split_audio(wav, sr, chunk_duration=60)
    global_profiles = {}
    all_segments = []
    offset = 0

    for chunk in chunks:
        segments, labels, times, speaker_embeds = diarize_chunk(chunk, sr, encoder, n_speakers)
        # return segments
        if not segments:
            offset += len(chunk) / sr
            continue

        chunk_profile_map = match_speakers_to_profiles(
            list(speaker_embeds.values()), global_profiles
        )
        label_to_global_id = {
            local_label: chunk_profile_map[i]
            for i, local_label in enumerate(speaker_embeds.keys())
        }

        for (label, start, end) in segments:
            global_id = label_to_global_id.get(label, label)
            all_segments.append((global_id, start + offset, end + offset))

        offset += len(chunk) / sr

    return all_segments


def main(video_path, n_speakers=2):
    check_cuda()
    wav, sr = extract_audio_np(video_path)
    encoder = VoiceEncoder()

    chunks = split_audio(wav, sr, chunk_duration=60)
    global_profiles = {}
    all_segments = []
    offset = 0

    for chunk in chunks:
        segments, labels, times, speaker_embeds = diarize_chunk(chunk, sr, encoder, n_speakers)
        if not segments:
            offset += len(chunk) / sr
            continue

        chunk_profile_map = match_speakers_to_profiles(
            list(speaker_embeds.values()), global_profiles
        )
        label_to_global_id = {
            local_label: chunk_profile_map[i]
            for i, local_label in enumerate(speaker_embeds.keys())
        }

        for (label, start, end) in segments:
            global_id = label_to_global_id.get(label, label)
            all_segments.append((global_id, start + offset, end + offset))

        offset += len(chunk) / sr

    print("\n🔤️  Diarized Speaker Segments:")
    for speaker, start, end in all_segments:
        print(f"Speaker {speaker}: {start:.1f}s – {end:.1f}s")

    # Plot diarization timeline
    fig, ax = plt.subplots(figsize=(12, 2))
    for speaker, start, end in all_segments:
        ax.plot([start, end], [speaker, speaker], lw=6)
    ax.set_title("Speaker Diarization")
    ax.set_xlabel("Time (min:sec)")
    # ax.set_xticks(list(range(0, 180, 5)))
    ax.set_yticks(sorted(set([s for s, _, _ in all_segments])))
    ax.set_yticklabels([f"Speaker {s}" for s in sorted(set([s for s, _, _ in all_segments]))])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(60))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x//60)}:{int(x%60):02d}"))
    ax.grid(True, which='both', axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main("inputvids/rhettandlink_edit2.mp4", n_speakers=2)

# "inputvids/rhettandlink_edit3.mp4"
