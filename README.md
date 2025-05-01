# AI TikTok/Instagram Reel Editor

A video editing script for content creators that automatically edits
widescreen videos of podcasts/youtube videos/etc. into a format 
appropriate for TikTok/Instagram Reels/etc.

The program uses a mix of speaker diarization, face embeddings,
shot change tracking, and speech-to-text to edit videos.

## How to Use

Create conda environment with:

`conda env create -f environment.yml`

Then run main.py on a video specifying number of speakers and faces:

`python main.py --video-path path/to/video.mp4 --n-speakers 2 --max-num-faces 2`

Add tag `--add-subtitles` to output edited video with subtitles

The output video can be found in the output directory

### Example

[Example output with audio](https://www.youtube.com/watch?v=JIgFepXPoT0)

![Input Example](examples/input.gif)

![Output Example](examples/output.gif)

(Credit: Good Mythical Morning)
