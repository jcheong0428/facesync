from __future__ import division

'''
    FaceSync Class
    ==========================================
    Class to sync videos by audio matching.

'''
__all__ = ['facesync']
__author__ = ["Jin Hyun Cheong"]
__license__ = "MIT"

import os
import numpy as np
import subprocess
import scipy.io.wavfile as wav

def _get_vid_resolution(vidFile):
    """ Gets video resolution for a given file using ffprobe.
    """
    cmd = [
         'ffprobe','-v','error','-of','flat=s=_','-select_streams','v:0','-show_entries','stream=height,width', vidFile
    ]
    proc = subprocess.Popen(cmd,stdout=subprocess.PIPE)
    out = proc.communicate()[0]
    out = out.split('\n')[:2]
    return tuple([int(elem.split('=')[-1]) for elem in out])

def write_offset_to_file(afile, offset, header='offset'):
    '''
    Helper function to write offset output to file.
    '''
    (path2fname, fname) = os.path.split(afile)
    fname = os.path.join(path2fname,fname.split(".")[0] + '.txt')
    f = open(fname, 'a+')
    f.write(header+'\n')
    f.write(str(offset)+'\n')
    f.close()

def processInput(rate0,data0,afile,fps,length,search_start,search_end,verbose):
    '''
    Helper function for multiprocessing
    '''
    if verbose:
        print(afile)
    rate1,data1 = wav.read(afile)
    assert(rate0==rate1), "Audio sampling rate is not the same for target and sample" # Check if they have same rate
    searchtime = search_end-search_start # seconds to search alignment
    if np.ndim(data0)>1:
        data0 = data0[:,0]
    if np.ndim(data1)>1:
        data1 = data1[:,0]
    to_compare = data0[0:rate0*length]
    try:
        assert(data1.shape[0] - (searchtime+length)*rate0 >= 0)
    except:
        print("Original length need to be shorter or reduce searchtime to allow buffer at end.")
    rs = []
    ts = []
    # for i in np.linspace(0,searchtime,fps*searchtime):
    inputs = list(np.linspace(search_start,search_end,fps*searchtime))

    ts = inputs
    rs.append(rs)
    # offset_r = ts[np.argmax(rs)] + search_start
    offset_r = ts[np.argmax(rs)]
    self.offsets.append(offset_r)
    write_offset_to_file(afile, offset_r,header='corr_multi')
    return rs,offset_r

def calc_rs(i, to_compare, sample):
    try:
        assert(to_compare.shape[0]==sample.shape[0])
        r=np.corrcoef(to_compare,sample)[0][1]
    except:
        print("Shape mismatch at %s" %str(i))
    return r, i

class facesync(object):
    """
    facesync is a class to represents multiple videos
    so that one can align them based on audio.

    Args:
        data: list of video files
        Y: Pandas DataFrame of training labels
        X: Pandas DataFrame Design Matrix for running univariate models
        mask: binary nifiti file to mask brain data
        output_file: Name to write out to nifti file
        **kwargs: Additional keyword arguments to pass to the prediction algorithm

    """
    def __init__(self, video_files=None, audio_files=None, target_audio = None, offsets=None,**kwargs):
        '''
        Args:
            video_files: list of video filenames to process
            audio_files: list of video filenames to process
            target_audio: audio to which videos will be aligned
            offsets: list of offsets to trim the video_files
        '''
        # Initialize attributes
        self.video_files = video_files
        self.audio_files = audio_files
        self.target_audio = target_audio
        self.offsets = offsets

        if self.video_files is not None:
            assert(isinstance(self.video_files,list)),'Place path to files in a list'
        if self.audio_files is not None:
            assert(isinstance(self.audio_files,list)),'Place path to files in a list'
        if (self.video_files is not None) & (self.offsets is not None):
            assert(len(self.video_files)==len(self.offsets)),'Number of videos and number of offsets should match'

    def extract_audio(self,rate=44100,call=True,verbose=True):
        '''
        This method extracts audio from video files in self.video_files and saves audio files in self.audio_files

        Input
        ------------
        rate: rate of audio stream frequency to be extracted, default 44100
        call: boolean, whether to wait for each process to finish or open multiple threads
        verbose: if True, prints the currently processing audio filename
        '''
        assert(len(self.video_files)!=0),'No video files to process'
        self.audio_files = []
        for i, vidfile in enumerate(self.video_files):
            if verbose:
                print(vidfile)
            (path2fname, vname) = os.path.split(vidfile)
            aname = vname.split(".")[0] + ".wav"
            infile = os.path.join(path2fname,vname)
            outfile = os.path.join(path2fname,aname)
            self.audio_files.append(outfile)
            # cmd = ' '.join(["avconv", "-i", infile, "-y", "-vn", "-ac", "1","-ar",str(rate),"-f", "wav", outfile])
            command = "ffmpeg -y -i " + infile + " -ab 128k -ac 2 -ar " +str(rate) +" -vn " + outfile
            if call:
                subprocess.call(command, shell=True)
            else:
                subprocess.Popen(command, shell=True)

    def find_offset_cross(self,length = 10,search_start=0,verbose=True):
        '''
        Find offset using Fourier Transform cross correlation.

        Input
        ------------
        length: seconds to use for the cross correlation matching, default is 10 seconds
        verbose: if True, prints the currently processing audio filename

        Output
        ------------
        allrs : list of cross correlation results using fftconvolve. to retrieve the offset time need to zero index and subtract argmax.
        '''
        import numpy as np
        from scipy.signal import fftconvolve
        assert(self.target_audio is not None), 'Target audio not specified'
        assert(self.audio_files is not None), 'Audio files not specified'
        self.offsets = []
        rate0,data0 = wav.read(self.target_audio)
        allrs = []
        for i, afile in enumerate(self.audio_files):
            if verbose:
                print(afile)
            rate1,data1 = wav.read(afile)
            assert(rate0==rate1), "Audio sampling rate is not the same for target and sample" # Check if they have same rate
            # Take first audio channel
            if np.ndim(data0)>1:
                data0 = data0[:,0]
            if np.ndim(data1)>1:
                data1 = data1[:,0]
            x = data0[:rate0*length] # target audio
            y = data1[int(search_start*rate0):int(search_start*rate0)+rate0*length] # change sample audio location
            # Pad target audio with zeros if not same length.
            if len(x) < len(y):
                xnew = np.zeros_like(y)
                xnew[:len(x)] = x
                x = xnew
            assert(len(x)==len(y)), "Length of two samples must be the same"
            crosscorr = fftconvolve(x,y[::-1],'full')
            zero_index = int(len(crosscorr) / 2 ) -1
            offset_x = search_start+(zero_index - np.argmax(crosscorr))/float(rate0)
            # assert(len(crosscorr)==len(x))
            self.offsets.append(offset_x)
            write_offset_to_file(afile, offset_x,header='xcorr_len'+str(length))
            allrs.append(crosscorr)
        return allrs

    def find_offset_corr(self,length=5,search_start=0,search_end=20,fps=44100,verbose=True):
        '''
        Find offset based on correlation of two audio.

        Input
        ------------
        self.target_audio : Original audio to which other files will be aligned to
        self.audio_files : List of audio files that needs to be trimmed
        length : length of original sample to compare
        search_start, search_end: start and end times to search for alignment in seconds
        fps: level of temporal precision, default 44100
        verbose: if True, prints the currently processing audio filename

        Output
        ------------
        rs: correlation values
        '''
        assert(self.target_audio is not None), 'Target audio not specified'
        assert(self.audio_files is not None), 'Audio files not specified'
        self.offsets = []
        allrs = []
        rate0,data0 = wav.read(self.target_audio)
        for i, afile in enumerate(self.audio_files):
            if verbose:
                print(afile)
            rate1,data1 = wav.read(afile)
            assert(rate0==rate1), "Audio sampling rate is not the same for target and sample" # Check if they have same rate
            searchtime = search_end-search_start # seconds to search alignment
            if np.ndim(data0)>1:
                data0 = data0[:,0]
            if np.ndim(data1)>1:
                data1 = data1[:,0]
            to_compare = data0[0:rate0*length]
            try:
                assert(data1.shape[0] - (searchtime+length)*rate0 >= 0)
            except:
                print("Original length need to be shorter or reduce searchtime to allow buffer at end.")
            rs = []
            ts = []
            # for i in np.linspace(0,searchtime,fps*searchtime):
            for i in np.linspace(search_start,search_end,fps*searchtime):
                sample = data1[int(rate0*i):int(rate0*(i+length))][0:to_compare.shape[0]]
                try:
                    assert(to_compare.shape[0]==sample.shape[0])
                except:
                    print("Shape mismatch at %s" %str(i))
                try:
                    rs.append(np.corrcoef(to_compare,sample)[0][1])
                    ts.append(i)
                except:
                    pass
            allrs.append(rs)
            # offset_r = ts[np.argmax(rs)] + search_start
            offset_r = ts[np.argmax(rs)]
            self.offsets.append(offset_r)
            write_offset_to_file(afile, offset_r,header='corr_fps'+str(fps)+'_len'+str(length)+'_start'+str(search_start)+'_end'+str(search_end))
        return allrs

    def find_offset_corr_sparse(self,length=5,search_start=0,search_end=20,fps=44100,sparse_ratio=.5,verbose=True):
        '''
        Finds offset by correlation with sparse sampling.

        Input
        ------------
        self.target_audio : Original audio to which other files will be aligned to
        self.audio_files : List of audio files that needs to be trimmed
        length : length of original sample to compare
        search_start, search_end: start and end times to search for alignment in seconds
        fps: level of temporal precision, default 44100
        sparse_ratio = Determines the sparse sampling of the target audio to match (default is .5)
        verbose: if True, prints the currently processing audio filename

        Output
        ------------
        offset_r : time to trim based on correlation
        offset_d : time to trim based on distance
        rs: correlation values
        ds: difference values
        '''
        assert(self.target_audio is not None), 'Target audio not specified'
        assert(self.audio_files is not None), 'Audio files not specified'
        self.offsets = []
        allrs = []
        rate0,data0 = wav.read(self.target_audio)
        for i, afile in enumerate(self.audio_files):
            if verbose:
                print(afile)
            rate1,data1 = wav.read(afile)
            assert(rate0==rate1), "Audio sampling rate is not the same for target and sample" # Check if they have same rate
            searchtime = search_end-search_start # seconds to search alignment
            if np.ndim(data0)>1:
                data0 = data0[:,0]
            if np.ndim(data1)>1:
                data1 = data1[:,0]
            # to_compare = data0[0:rate0*length]
            sampleix = list(range(0,int(rate0*length)-1))
            np.random.shuffle(sampleix)
            sampleix = np.sort(sampleix[0:int(rate0*length*sparse_ratio)])
            to_compare = data0[sampleix]

            try:
                assert(data1.shape[0] - (searchtime+length)*rate0 >= 0)
            except:
                print("Original length need to be shorter or reduce searchtime to allow buffer at end.")
            rs = []
            ts = []
            # for i in np.linspace(0,searchtime,fps*searchtime):
            for i in np.linspace(search_start,search_end,fps*searchtime):
                # sample = data1[int(rate0*i):int(rate0*(i+length))][0:to_compare.shape[0]]
                sample = data1[int(rate0*i):int(rate0*(i+length))][sampleix]
                try:
                    assert(to_compare.shape[0]==sample.shape[0])
                except:
                    print("Shape mismatch at %s" %str(i))
                try:
                    rs.append(np.corrcoef(to_compare,sample)[0][1])
                    ts.append(i)
                except:
                    pass
            allrs.append(rs)
            # offset_r = ts[np.argmax(rs)] + search_start
            offset_r = ts[np.argmax(rs)]
            self.offsets.append(offset_r)
            write_offset_to_file(afile, offset_r, header='corr_sparse_fps'+str(fps)+'_len'+str(length)+'_start'+str(search_start)+'_end'+str(search_end))
        return allrs

    def find_offset_corr_multi(self,length=5,search_start=0,search_end=20,fps=44100,verbose=True):
        '''
        Find offset based on correlation with multiprocessing.
        Requires joblib package.

        Input
        ------------
        self.target_audio : Original audio to which other files will be aligned to
        self.audio_files : List of audio files that needs to be trimmed
        length : length of original sample to compare
        search_start, search_end: start and end times to search for alignment in seconds
        fps: level of temporal precision, default 44100
        verbose: if True, prints the currently processing audio filename

        Output
        ------------
        self.offsets: max offsets
        rs: correlation values
        '''
        from joblib import Parallel, delayed
        import multiprocessing
        num_cores = multiprocessing.cpu_count()-1 # don't use all cores

        assert(self.target_audio is not None), 'Target audio not specified'
        assert(self.audio_files is not None), 'Audio files not specified'
        self.offsets = []
        allrs = []
        rate0,data0 = wav.read(self.target_audio)
        for i, afile in enumerate(self.audio_files):
            if verbose:
                print(afile)
            rate1,data1 = wav.read(afile)
            assert(rate0==rate1), "Audio sampling rate is not the same for target and sample" # Check if they have same rate
            searchtime = search_end-search_start # seconds to search alignment
            if np.ndim(data0)>1:
                data0 = data0[:,0]
            if np.ndim(data1)>1:
                data1 = data1[:,0]
            to_compare = data0[0:rate0*length]
            try:
                assert(data1.shape[0] - (searchtime+length)*rate0 >= 0)
            except:
                print("Original length need to be shorter or reduce searchtime to allow buffer at end.")
            rs = []
            ts = []
            out = Parallel(n_jobs=num_cores,backend='threading')(delayed(calc_rs)(i,to_compare,data1[int(rate0*i):int(rate0*(i+length))][0:to_compare.shape[0]]) for i in np.linspace(search_start,search_end,fps*searchtime))
            rs,ts= zip(*out)
            allrs.append(rs)
            offset_r = ts[np.argmax(rs)]
            self.offsets.append(offset_r)
            write_offset_to_file(afile, offset_r,header='corr_fps'+str(fps)+'_len'+str(length)+'_start'+str(search_start)+'_end'+str(search_end))
        return allrs

    def find_offset_dist(self,length=5,search_start=0,search_end=20,fps=44100,verbose=True):
        '''
        Find offset based on squared distance of audio wave.

        Input
        ------------
        self.target_audio : Original audio to which other files will be aligned to
        self.audio_files : List of audio files that needs to be trimmed
        length : length of original sample to compare
        search_start, search_end: start and end times to search for alignment in seconds
        fps: level of temporal precision, default 44100
        verbose: if True, prints the currently processing audio filename

        Output
        ------------
        offset_d : time to trim based on distance
        rs: correlation values
        ds: difference values
        '''
        assert(self.target_audio is not None), 'Target audio not specified'
        assert(self.audio_files is not None), 'Audio files not specified'
        self.offsets = []
        allds = []
        rate0,data0 = wav.read(self.target_audio)
        for i, afile in enumerate(self.audio_files):
            if verbose:
                print(afile)
            rate1,data1 = wav.read(afile)
            assert(rate0==rate1), "Audio sampling rate is not the same for target and sample" # Check if they have same rate
            searchtime = search_end-search_start # seconds to search alignment
            if np.ndim(data0)>1:
                data0 = data0[:,0]
            if np.ndim(data1)>1:
                data1 = data1[:,0]
            to_compare = data0[0:rate0*length]
            try:
                assert(data1.shape[0] - (searchtime+length)*rate0 >= 0)
            except:
                print("Original length need to be shorter or reduce searchtime to allow buffer at end.")
            ds = []
            ts = []
            # for i in np.linspace(0,searchtime,fps*searchtime):
            for i in np.linspace(search_start,search_end,fps*searchtime):
                sample = data1[int(rate0*i):int(rate0*(i+length))][0:to_compare.shape[0]]
                try:
                    assert(to_compare.shape[0]==sample.shape[0])
                except:
                    print("Shape mismatch at %s" %str(i))
                try:
                    ds.append(sum((to_compare-sample)**2))
                    ts.append(i)
                except:
                    pass
            allds.append(ds)
            # offset_d = ts[np.argmin(ds)] + search_start
            offset_d = ts[np.argmin(ds)]
            self.offsets.append(offset_d)
            write_offset_to_file(afile, offset_d,header='dist_fps'+str(fps)+'_len'+str(length)+'_start'+str(search_start)+'_end'+str(search_end))
        return allds

    def resize_vids(self, resolution = 64, suffix = None,call = True, force=False):
        '''
        Resize videos.

        Inputs
        ------------
        resolution: height of the video
        suffix: what to name the resized video. If not specified, will append video names with resolution
        call: boolean, whether to wait for each process to finish or open multiple threads,
        True: call, False: multithread, default is call
        force: whether to force creating new files some video files are already at the desired resolution; defaults to False
        '''
        if suffix == None:
            suffix = str(resolution)

        out = []
        for vidfile in self.video_files:
            (path2fname, vname) = os.path.split(vidfile)
            print("Resizing video: %s" % (vname))
            current_resolution = _get_vid_resolution(vidfile)
            if current_resolution[1] == resolution and not force:
                print("Native resolution already ok, skipping: %s" % (vname))
                final_vidname = os.path.join(path2fname,vname)
                out.append(final_vidname)
                continue
            else:
                final_vidname = os.path.join(path2fname,vname.split('.')[0]+'_'+suffix+'.'+vname.split('.')[-1])
                out.append(final_vidname)
                command = 'ffmpeg -y -i ' + vidfile + ' -vf scale=-1:'+str(resolution)+' '+final_vidname
                if not os.path.exists(final_vidname):
                    if call:
                        subprocess.call(command, shell=True)
                    else:
                        subprocess.Popen(command, shell=True)
        return out

    def concat_vids(self, final_vidname = None, resolution_fix=False, checkres=True):
        '''
        Concatenate list of videos to one video.

        Inputs
        ------------
        final_vidname = Filepath/filname of the concatenated video. If not specified will use the first video name appended with _all
        '''
        assert(len(self.video_files)!=0),'No video files to process'
        if (final_vidname != None):
            self.final_vidname = final_vidname
        if (len(self.video_files)!=0) and (final_vidname == None):
            (path2fname, vname) = os.path.split(self.video_files[0])
            self.final_vidname = os.path.join(path2fname,vname.split('.')[0]+'_all.'+vname.split('.')[-1])
        assert(type(self.final_vidname)==str),'final_vidname must be a string with full path'

        #Check that files are all of the same resolution
        if checkres:
            resolutions = [_get_vid_resolution(elem) for elem in self.video_files]
            if len(set(resolutions)) > 1:
                if resolution_fix:
                    min_resolution = min([elem[1] for elem in resolutions])
                    print("Videos mismatch in resolution, resizing to: %s..." % (min_resolution))
                    new_vids= self.resize_vids(resolution=min_resolution)
                    self.video_files = new_vids
                    resolutions = [_get_vid_resolution(elem) for elem in self.video_files]
                    assert(len(set(resolutions))<=1),"Videos still mismatched. Something went wrong with automatic resizing? Try resizing manually."
                    print("Resizing complete. Continuing.")
                else:
                    raise TypeError("Video files have different resolutions!")

        # Create intermediate video files
        tempfiles = str();
        for i, vidfile in enumerate(self.video_files):
            (path2fname, vname) = os.path.split(vidfile)
            print("Joining video: %s" % (vname))
            if len(tempfiles)!=0:
                tempfiles = tempfiles+"|"
            intermediatefile = os.path.join(path2fname,"intermediate"+str(i)+'.ts')
            if not os.path.exists(intermediatefile):
                command = "ffmpeg -i "+ vidfile +" -c copy -bsf:v h264_mp4toannexb -f mpegts " + intermediatefile
                subprocess.call(command, shell=True)
            tempfiles = tempfiles + intermediatefile

        # Concatenate videos
        command = 'ffmpeg -y -i "concat:' + tempfiles + '" -c copy -bsf:a aac_adtstoasc '+ self.final_vidname
        subprocess.call(command, shell=True)
        #remove intermediates
        for i, vidfile in enumerate(self.video_files):
            (path2fname, vname) = os.path.split(vidfile)
            intermediatefile = os.path.join(path2fname,"intermediate"+str(i)+'.ts')
            command = "rm -f " + intermediatefile
            subprocess.call(command, shell=True)

    def trim_vids(self,offsets = None, suffix = None,call=True):
        '''
        Trims video based on offset

        Inputs
        ------------
        offsets: list of offsets to trim the self.video_files with
        length of offsets should match length of self.video_files
        suffix: string to add to end of the trimmed video, default: 'trimmed'
        call: boolean, whether to wait for each process to finish or open multiple threads,
        True: call, False: multithread, default is call
        '''
        if suffix == None:
            suffix = 'trimmed'
        if offsets is not None:
            self.offsets= offsets
        assert(len(self.video_files)==len(self.offsets)),'Number of videos and number of offsets should match'
        for i,vidfile in enumerate(self.video_files):
            seconds = str(self.offsets[i])
            (path2fname, vname) = os.path.split(vidfile)
            print("Trimming video: %s" % (vname))
            final_vidname = os.path.join(path2fname,vname.split('.')[0]+'_'+suffix+'.'+vname.split('.')[-1])
            # command = 'ffmpeg -y -ss ' + str(seconds) + ' -i ' + vidfile + ' -c copy ' + final_vidname
            # command = 'ffmpeg -y -ss ' + seconds.split('.')[0] + ' -i ' + vidfile + ' -ss 00:00:00.' + seconds.split('.')[1] + ' -c copy ' + final_vidname
            command = 'ffmpeg -y -i ' + vidfile + ' -ss ' + str(seconds) + ' -crf 23 '  + final_vidname
            # command = 'ffmpeg -y -i ' + vidfile + ' -ss ' + str(seconds) + ' -vcodec libx264 -crf 23 -acodec copy '  + final_vidname
            if call:
                subprocess.call(command, shell=True)
            else:
                subprocess.Popen(command, shell=True)
