[![Build Status](https://travis-ci.org/jcheong0428/facesync.svg?branch=master)](https://travis-ci.org/jcheong0428/facesync)
[![Coverage Status](https://coveralls.io/repos/github/jcheong0428/facesync/badge.svg?branch=master)](https://coveralls.io/github/jcheong0428/facesync?branch=master)

# FaceSync: Open source framework for recording facial expressions with head-mounted cameras

The FaceSync toolbox provides 3D blueprints for building the head-mounted camera setup described in our [paper](https://psyarxiv.com/p5293/). The toolbox also provides functions to automatically synchronize videos based on audio, manually align audio, plot facial landmark movements, and inspect synchronized videos to graph data.   


## Installation

To install (for osx or linux) open Terminal and type

`pip install facesync`

or

`git clone https://github.com/jcheong0428/facesync.git`
then in the repository folder type
`python setup.py install`


## Dependencies
For full functionality, FACESYNC requires [ffmpeg](https://ffmpeg.org/) and the [libav](https://libav.org/) library.

Linux
`sudo apt-get install libav-tools`

OS X
`brew install ffmpeg`
`brew install libav`

also requires following packages:
- numpy
- scipy
You may also install these via `pip install -r requirements.txt`

## Recommended Processing Steps
1. Extract Audio from Target Video
2. Find offset with Extracted Audio
3. Trim Video using Offset.
*If you need to resize your video, do so before trimming.
Otherwise timing can be off.

```
from facesync.facesync import facesync
# change file name to include the full
video_files = ['path/to/sample1.MP4']
target_audio = 'path/to/cosan_synctune.wav'
# Intialize facesync class
fs = facesync(video_files=video_files,target_audio=target_audio)
# Extracts audio from sample1.MP4
fs.extract_audio()
# Find offset by correlation
fs.find_offset_corr(search_start=14,search_end=16)
print(fs.offsets)
# Find offset by fast fourier transform
fs.find_offset_fft()
print(fs.offsets)
```

# FaceSync provides handy utilities for working with facial expression data.

## Manually align the audios with AudioAligner.
```
%matplotlib notebook
from facesync.utils import AudioAligner
file_original = 'path/to/audio.wav'
file_sample = 'path/to/sample.wav'
AudioAligner(original=file_original, sample=file_sample)
```
<img src="/screenshots/AudioAligner.png" align="center" />


## Plot facial landmarks and how they change as a result of Action Unit changes.
```
%matplotlib notebook
from facesync.utils import ChangeAU, plotface
changed_face = ChangeAU(aulist=['AU6','AU12','AU17'], au_weight = 1.0)
ax = plotface(changed_face)
```
<img src="/screenshots/plotface.png" align="center" />


## Use the VideoViewer widget to play both video and data at the same time (only available on Python).
```
import facesync.utils as utils
%matplotlib notebook
utils.VideoViewer(path_to_video='path/to/video.mp4', data_df = fexDataFrame)
```
<img src="/screenshots/VideoViewer.png" align="center" />

# Citation
Please cite the following paper if you use our head-mounted camera setup or software.   
#### Cheong, J. H., Brooks, S., & Chang, L. J. (2017, November 1). FaceSync: Open source framework for recording facial expressions with head-mounted cameras. Retrieved from psyarxiv.com/p5293
