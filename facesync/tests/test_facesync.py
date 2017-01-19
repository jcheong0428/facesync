from facesync.facesync import facesync

def test_facesyc(tmpdir):
	fs = facesync()
	assert(len(fs.audio_files)==0)