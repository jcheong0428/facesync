from facesync.facesync import facesync

def test_facesyc(tmpdir):
	fs = facesync()
	assert(len(fs.audio_files)==0)
	assert(len(fs.video_files)==0)
	assert(isinstance(fs.target_audio,None))
	assert(isinstance(fs.audio_files,list))
	assert(isinstance(fs.video_files,list))

