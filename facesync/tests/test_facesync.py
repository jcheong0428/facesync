from facesync.facesync import facesync
import os 

def test_facesyc(tmpdir):
	fs = facesync()
	assert(len(fs.audio_files)==0)
	assert(len(fs.video_files)==0)
	assert(len(fs.offsets)==0)
	assert(fs.target_audio==None)
	assert(isinstance(fs.audio_files,list))
	assert(isinstance(fs.video_files,list))
	assert(isinstance(fs.offsets,list))

	video_files = [os.path.join(str(tmpdir.join('sample1.MP4')))]
	target_audio = os.path.join(str(tmpdir.join('cosan_synctune.wav')))
	fs = facesync(video_files=video_files,target_audio=target_audio)
	fs.extract_audio()
	assert(fs.audio_files == [os.path.join(str(tmpdir.join('sample1.wav')))])
	fs.find_offset_fft()
	assert(isinstance(fs.offsets,list))
	