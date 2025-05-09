# AI TikTok/Instagram Reel Editor

A video editing script for content creators that automatically edits
widescreen videos of podcasts/youtube videos/etc. into a format 
appropriate for TikTok/Instagram Reels/etc.

The program uses a mix of speaker diarization, face embeddings,
shot change tracking, and speech-to-text to edit videos.

[Example usage and output](https://youtu.be/zcGg5_8qJ4c)

## How to Use

Create conda environment with:

`conda env create -f environment.yml`

Then run main.py on a video specifying number of speakers:

`python main.py --video-path path/to/video.mp4 --n-speakers 2`

You may also set the face embedding threshold with the `--threshold` arg, e.g. `--threshold=0.4` 

Add tag `--add-subtitles` to create .srt file for video

Default subtitles are sentence subtitles. Run with `--word-timestamps` for each word subtitled individually.

Open Blender and run blender.py from Text Editor in Video Editing mode to import editing data for last video processed with main.py

You can now continue editing the video or export.

