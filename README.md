[![Build Status](https://travis-ci.org/jcheong0428/facesync.svg?branch=master)](https://travis-ci.org/jcheong0428/facesync)
[![Coverage Status](https://coveralls.io/repos/github/jcheong0428/facesync/badge.svg?branch=master)](https://coveralls.io/github/jcheong0428/facesync?branch=master)

# FACESYNC - Python toolbox to sync videos by audio. 

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
- python 2.7.x
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
print(fs.offsets
```

##You can manually align the audios with AudioAligner. 
<img style="float:left;vertical-align:top;" src="/screenshots/AudioAligner.png" align="left" />

## With Python3, you can use the VideoViewer widget to play both video and data at the same time. 
<img style="float:left;vertical-align:top;" src="/screenshots/VideoViewer.png" align="left" />

