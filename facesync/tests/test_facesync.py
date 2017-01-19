from facesync.facesync import facesync

def test_facesyc(tmpdir):
	fs = facesync()
	assert(len(fs.audio_files)==0)
	assert(len(fs.video_files)==0)
	assert(len(fs.offsets)==0)
	assert(fs.target_audio==None)
	assert(isinstance(fs.audio_files,list))
	assert(isinstance(fs.video_files,list))
	assert(isinstance(fs.offsets,list))

	video_files = ['../resources/sample1.MP4']
	target_audio = ['../resources/cosan_synctune.wav']
	fs = facesync(video_files=video_files,target_audio=target_audio)
	fs.extract_audio()

	