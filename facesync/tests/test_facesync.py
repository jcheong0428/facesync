from facesync.facesync import facesync
import os, glob

def test_facesyc(tmpdir):
	fs = facesync()
	assert(len(fs.audio_files)==0)
	assert(len(fs.video_files)==0)
	assert(len(fs.offsets)==0)
	assert(fs.target_audio==None)
	assert(isinstance(fs.audio_files,list))
	assert(isinstance(fs.video_files,list))
	assert(isinstance(fs.offsets,list))
	cwd = os.getcwd()
	video_files = [os.path.join(cwd,'facesync/resources/sample1.MP4')]
	target_audio = os.path.join(cwd,'facesync/resources/cosan_synctune.wav')
	fs = facesync(video_files=video_files,target_audio=target_audio)
	fs.extract_audio()
	print(glob.glob(os.path.join(str(cwd.join('*')))))
	print(glob.glob(os.path.join(str(cwd.join('facesync/*')))))
	print(glob.glob(os.path.join(str(cwd.join('facesync/resources/*')))))
	assert(fs.audio_files == [os.path.join(cwd,'facesync/resources/sample1.wav')])
	fs.find_offset_fft()
	assert(isinstance(fs.offsets,list))
	