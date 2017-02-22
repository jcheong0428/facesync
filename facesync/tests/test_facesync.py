from facesync.facesync import facesync
import os, glob

def test_facesyc(tmpdir):
	fs = facesync()
	assert(fs.audio_files==None)
	assert(fs.video_files==None)
	assert(fs.offsets==None)
	assert(fs.target_audio==None)
	cwd = os.getcwd()
	video_files = [os.path.join(os.path.dirname(__file__), 'resources','sample1.MP4')]
	target_audio = os.path.join(os.path.dirname(__file__), 'resources','cosan_synctune.wav')
	fs = facesync(video_files=video_files,target_audio=target_audio)
	fs.extract_audio()
	print(glob.glob(os.path.join(os.path.dirname(__file__), 'resources','*.MP4')))
	print(fs.audio_files)
	assert(fs.audio_files == [os.path.join(os.path.dirname(__file__), 'resources','sample1.wav')])
	fs.find_offset_cross()
	assert(isinstance(fs.offsets,list))
	fs.find_offset_corr(search_start=14,search_end=16)
	fs.find_offset_dist(search_start=14,search_end=16)
	fs.resize_vids(resolution = 32,suffix = 'test')

	# add tests for video concat