from facesync.facesync import facesync
import os, glob
import numpy as np

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
	
	print('testing fft cross correlation')
	fs.find_offset_cross(length=3,search_start=15)
	assert(np.round(fs.offsets[0])==np.round(15.1612471655))

	assert(isinstance(fs.offsets,list))

	print('testing correlation method')
	fs.find_offset_corr(search_start=15,search_end=16,fps=441)
	assert(np.round(fs.offsets[0])==np.round(15.1612603317))

	print('testing sparse correlation method')
	fs.find_offset_corr_sparse(length = 1.8,search_start=15,search_end=16,sparse_ratio=.5,fps=441)
	assert(np.round(fs.offsets[0])==np.round(15.1612603317))

	print('testing distance method')
	fs.find_offset_dist(search_start=15,search_end=16,fps=441)

	print('testing trimming method')
	# fs.trim_vids(call = False)
	print('testing resizing method with Popen')
	# fs.resize_vids(resolution = 32,suffix = 'test',call = False)

	# add tests for video concat