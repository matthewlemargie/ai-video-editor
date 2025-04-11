import whisper
from pyannote.audio import Pipeline
import ffmpeg
import os
import subprocess

def compile_pdf(tex_file):
    subprocess.run(["pdflatex", tex_file], check=True)

# Whisper transcription
whisper_model = whisper.load_model("base")
video_path = "inputvids/rhettandlink_edit2.mp4"
whisper_result = whisper_model.transcribe(video_path, word_timestamps=True)

# Extract audio for PyAnnote
audio_path = "temp_audio.wav"
ffmpeg.input(video_path).output(audio_path, ac=1, ar=16000).run(quiet=True, overwrite_output=True)

# PyAnnote diarization
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
)
diarization = pipeline({'uri': 'audio', 'audio': audio_path})

# Merge speaker turns with Whisper segments
speaker_segments = []
for segment, _, speaker in diarization.itertracks(yield_label=True):
    speaker_text = ""
    for whisper_segment in whisper_result["segments"]:
        if (whisper_segment["start"] < segment.end and
                whisper_segment["end"] > segment.start):
            speaker_text += whisper_segment["text"].strip() + " "
    if speaker_text.strip():
        speaker_segments.append((speaker, speaker_text.strip()))

# Generate LaTeX document
tex_filename = "transcript.tex"
with open(tex_filename, "w") as f:
    f.write(r"""\documentclass[12pt]{article}
            \usepackage[margin=1in]{geometry}
            \usepackage{parskip}
            \usepackage{titlesec}
            \usepackage{lmodern}
            \titleformat{\section}{\normalfont\Large\bfseries}{}{0em}{}
            \renewcommand{\familydefault}{\sfdefault}
            \begin{document}
            \title{Video Transcript}
            \author{}
            \date{}
            \maketitle
            """)
    for speaker, paragraph in speaker_segments:
        speaker = f"Speaker {speaker.split('_')[-1]}"
        f.write(f"\\section*{{{speaker}}}\n")
        f.write(paragraph.replace("\n", " ") + "\n\n")
    f.write("\\end{document}")

print(f"LaTeX file saved as {tex_filename}")

compile_pdf("transcript.tex")

os.remove("transcript.tex")
