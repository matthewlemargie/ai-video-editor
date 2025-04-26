# AI TikTok/Instagram Reel Editor

A video editing script for content creators that automatically edits
widescreen videos of podcasts/youtube videos/etc. into a format 
appropriate for TikTok/Instagram Reels/etc.

The program uses a mix of speaker diarization, face embeddings,
shot change tracking, and speech-to-text to edit videos.

How to use:

Create conda environment with:

`conda env create -f environment.yml`

Then run main.py on a video specifying number of speakers and faces:

`python main.py --video-path path/to/video.mp4 --n-speakers 3 --max-num-faces 3`

Here is an example of a video edited by the script:

![Input Example](examples/input.gif)

![Output Example](examples/output.gif)

(Credit: Good Mythical Morning)
