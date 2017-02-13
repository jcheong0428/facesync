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
import math

def write_offset_to_file(afile, offset_r, header):
    fname = afile.split(".")[0] + '.txt'
    f = open(fname, 'w')
    f.write(header+'\n')
    f.write(str(offset_r)) 
    f.close()

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
    def __init__(self, video_files=None, audio_files=None, target_audio = None, **kwargs):
        '''
        Args:
            data: list of video filenames to process
            target_audio: audio to which videos will be aligned 
        Attributes:
            .paths: paths to video or audio files
            .fnames: video file names
            .anames: audio file names 
        '''
        # Initialize attributes
        self.video_files = []
        self.audio_files = []
        self.target_audio = None
        self.offsets = []

        if video_files is not None:
            assert(isinstance(video_files,list)),'Place path to files in a list'
            self.video_files = video_files
        if audio_files is not None:
            assert(isinstance(audio_files,list)),'Place path to files in a list'
            self.audio_files = audio_files
        if target_audio is not None: 
            self.target_audio = target_audio
        # Separate filepath to path and filename for easy processing.

    def extract_audio(self):
        '''
        This method extracts audio from video files in 'vidfiles'
        and then saves audio files in audfiles

        Returns whether it was successful
        '''
        assert(len(self.video_files)!=0),'No video files to process'
        rate = 44100
        self.audio_files = [] 
        for i, vidfile in enumerate(self.video_files):
            (path2fname, vname) = os.path.split(vidfile)
            aname = vname.split(".")[0] + ".wav"
            infile = os.path.join(path2fname,vname)
            outfile = os.path.join(path2fname,aname)
            self.audio_files.append(outfile)
            # cmd = ' '.join(["avconv", "-i", infile, "-y", "-vn", "-ac", "1","-ar",str(rate),"-f", "wav", outfile])
            cmd = "ffmpeg -i " + infile + " -y -ab 128k -ac 2 -ar 44100 -vn " + outfile
            p = subprocess.Popen(cmd,shell=True)
        return p

    def find_offset_fft(self,fft_bin_size=1024,overlap=0,box_height=512,box_width=43,samples_per_box=7,seconds_to_search = 60):
        '''
        Use the Frequency Constellation Alignment method as proposed by 
        https://github.com/allisonnicoledeal/VideoSync . 
        Audio data is split according to fft_bin_Size and Fast Fourier Transform is applied to each bin.
        Peak frequencies are determined for each sample bin. 
        Peak frequencies are compared between the two audio files to determine average offset.

        Input
        ------------
        self.audfiles : list of audio files to find offsets.
        self.target_audio : Original audio to which other files will be aligned to
        fft_bin_size : length of original to do fft; shouldn't be larger than actual clip length
        
        Output
        ------------
        delays are saved to self.delays
        '''
        def make_horiz_bins(data, fft_bin_size, overlap, box_height):
            horiz_bins = {}
            # process first sample and set matrix height
            sample_data = data[0:fft_bin_size]  # get data for first sample
            if (len(sample_data) == fft_bin_size):  # if there are enough audio points left to create a full fft bin
                intensities = fourier(sample_data)  # intensities is list of fft results
                for i in range(len(intensities)):
                    box_y = int(i/box_height)
                    if horiz_bins.has_key(box_y):
                        horiz_bins[box_y].append((intensities[i], 0, i))  # (intensity, x, y)
                    else:
                        horiz_bins[box_y] = [(intensities[i], 0, i)]
            # process remainder of samples
            x_coord_counter = 1  # starting at second sample, with x index 1
            for j in range(int(fft_bin_size - overlap), len(data), int(fft_bin_size-overlap)):
                sample_data = data[j:j + fft_bin_size]
                if (len(sample_data) == fft_bin_size):
                    intensities = fourier(sample_data)
                    for k in range(len(intensities)):
                        box_y = int(k/box_height)
                        if horiz_bins.has_key(box_y):
                            horiz_bins[box_y].append((intensities[k], x_coord_counter, k))  # (intensity, x, y)
                        else:
                            horiz_bins[box_y] = [(intensities[k], x_coord_counter, k)]
                x_coord_counter += 1
            return horiz_bins


        # Compute the one-dimensional discrete Fourier Transform
        # INPUT: list with length of number of samples per second
        # OUTPUT: list of real values len of num samples per second
        def fourier(sample):  #, overlap):
            mag = []
            fft_data = np.fft.fft(sample)  # Returns real and complex value pairs
            for i in range(int(len(fft_data)/2)):
                r = fft_data[i].real**2
                j = fft_data[i].imag**2
                mag.append(round(math.sqrt(r+j),2))
        #         mag.append(round(np.sqrt(r+j),2))
            return mag


        def make_vert_bins(horiz_bins, box_width):
            boxes = {}
            for key in horiz_bins.keys():
                for i in range(len(horiz_bins[key])):
                    box_x = horiz_bins[key][i][1] / box_width
                    if boxes.has_key((box_x,key)):
                        boxes[(box_x,key)].append((horiz_bins[key][i]))
                    else:
                        boxes[(box_x,key)] = [(horiz_bins[key][i])]

            return boxes


        def find_bin_max(boxes, maxes_per_box):
            freqs_dict = {}
            for key in boxes.keys():
                max_intensities = [(1,2,3)]
                for i in range(len(boxes[key])):
                    if boxes[key][i][0] > min(max_intensities)[0]:
                        if len(max_intensities) < maxes_per_box:  # add if < number of points per box
                            max_intensities.append(boxes[key][i])
                        else:  # else add new number and remove min
                            max_intensities.append(boxes[key][i])
                            max_intensities.remove(min(max_intensities))
                for j in range(len(max_intensities)):
                    if freqs_dict.has_key(max_intensities[j][2]):
                        freqs_dict[max_intensities[j][2]].append(max_intensities[j][1])
                    else:
                        freqs_dict[max_intensities[j][2]] = [max_intensities[j][1]]

            return freqs_dict


        def find_freq_pairs(freqs_dict_orig, freqs_dict_sample):
            time_pairs = []
            for key in freqs_dict_sample.keys():  # iterate through freqs in sample
                if freqs_dict_orig.has_key(key):  # if same sample occurs in base
                    for i in range(len(freqs_dict_sample[key])):  # determine time offset
                        for j in range(len(freqs_dict_orig[key])):
                            time_pairs.append((freqs_dict_sample[key][i], freqs_dict_orig[key][j]))

            return time_pairs


        def find_delay(time_pairs):
            t_diffs = {}
            for i in range(len(time_pairs)):
                delta_t = time_pairs[i][0] - time_pairs[i][1]
                if t_diffs.has_key(delta_t):
                    t_diffs[delta_t] += 1
                else:
                    t_diffs[delta_t] = 1
            t_diffs_sorted = sorted(t_diffs.items(), key=lambda x: x[1])
        #     print t_diffs_sorted
            time_delay = t_diffs_sorted[-1][0]

            return time_delay

        # Main method section
        assert(self.target_audio is not None), 'Target audio not specified'
        assert(self.audio_files is not None), 'Audio files not specified'
        self.offsets = []
        print os.getcwd()
        rate0,data0 = wav.read(os.path.join(self.target_audio))
        for i, afile in enumerate(self.audio_files):
            rate1,data1 = wav.read(afile)
            assert(rate0==rate1), "Audio sampling rate is not the same for target and sample" # Check if they have same rate
            # get one side of the audio in case of stereo audio
            if np.ndim(data0)>1:
                data0 = data0[:,0]
            if np.ndim(data1)>1:
                data1 = data1[:,0]
            bins_dict0 = make_horiz_bins(data0, fft_bin_size, overlap, box_height)
            boxes0 = make_vert_bins(bins_dict0, box_width)
            ft_dict0 = find_bin_max(boxes0, samples_per_box)

            bins_dict1 = make_horiz_bins(data1, fft_bin_size, overlap, box_height)
            boxes1 = make_vert_bins(bins_dict1, box_width)
            ft_dict1 = find_bin_max(boxes1, samples_per_box)

            # Determie time delay
            pairs = find_freq_pairs(ft_dict0, ft_dict1)
            delay = find_delay(pairs)
            samples_per_sec = float(rate0) / float(fft_bin_size)
            seconds= round(float(delay) / float(samples_per_sec), 4)
            self.offsets.append(seconds)
            write_offset_to_file(afile, seconds,header='fft')

    def find_offset_corr(self,length=5,search_start=0,search_end=20,fps=120):
        '''
        Input
        ------------
        self.target_audio : Original audio to which other files will be aligned to
        self.audio_files : List of audio files that needs to be trimmed
        length : length of original sample to compare
        search_start, search_end: start and end times to search for alignment in seconds
        fps: level of temporal precision
        
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
                print "Original length need to be shorter or reduce searchtime to allow buffer at end."
            rs = []
            ts = []
            for i in np.linspace(0,searchtime,fps*searchtime):
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
            write_offset_to_file(afile, offset_r,header='corr')
        return allrs

    def find_offset_dist(self,length=5,search_start=0,search_end=20,fps=120):
        '''
        Input
        ------------
        self.target_audio : Original audio to which other files will be aligned to
        self.audio_files : List of audio files that needs to be trimmed
        length : length of original sample to compare
        search_start, search_end: start and end times to search for alignment in seconds
        fps: level of temporal precision
        
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
                print "Original length need to be shorter or reduce searchtime to allow buffer at end."
            ds = []
            ts = []
            for i in np.linspace(0,searchtime,fps*searchtime):
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
            write_offset_to_file(afile, offset_d,header='dist')
        return allds

    def concat_vids(self, final_vidname = None):
        '''
        This method concatenates multiple videos into one continuous video. 
        final_vidname = Name to call concatenated video. If not specified will use the first video name appended with _all
        '''
        assert(len(self.video_files)!=0),'No video files to process'
        if (len(self.video_files)!=0) and (final_vidname == None):
            (path2fname, vname) = os.path.split(self.video_files[0])
            self.final_vidname = os.path.join(path2fname,vname.split('.')[0]+'_all.'+vname.split('.')[-1])
        assert(type(final_vidname)==str),'final_vidname must be a string with full path'

        # Create intermediate video files
        tempfiles = str();
        for i, vidfile in enumerate(self.video_files):
            (path2fname, vname) = os.path.split(vidfile)
            if len(tempfiles)!=0:
                tempfiles = tempfiles+"|"
            intermediatefile = os.path.join(path2fname,"intermediate"+str(i)+'.ts')
            command = "ffmpeg -i "+ vidfile +" -c copy -bsf:v h264_mp4toannexb -f mpegts " + intermediatefile
            subprocess.Popen(command, shell=True)
            tempfiles = tempfiles + intermediatefile

        # Concatenate videos
        command = 'ffmpeg -i "concat:' + tempfiles + '" -c copy -bsf:a aac_adtstoasc '+ self.final_vidname
        subprocess.Popen(command, shell=True)
        #remove intermediates
        for i, vidfile in enumerate(self.video_files):
            (path2fname, vname) = os.path.split(vidfile)
            intermediatefile = os.path.join(path2fname,"intermediate"+str(i)+'.ts')
            command = "rm -f " + intermediatefile
            subprocess.Popen(command, shell=True)

    def resize_vids(self, resolution = 64, suffix = None):
        '''
        resolution: height of the video
        suffix: what to name the resized video. If not specified, will append video names with resolution
        '''
        if suffix == None: 
            suffix = str(resolution)

        for vidfile in self.video_files: 
            (path2fname, vname) = os.path.split(vidfile)
            final_vidname = os.path.join(path2fname,vname.split('.')[0]+'_'+suffix+'.'+vname.split('.')[-1])
            command = 'ffmpeg -i ' + vidfile + ' -vf scale=-1:'+str(resolution)+' '+final_vidname
            subprocess.Popen(command, shell=True)

    def trim_vids(self,offsets = None, suffix = None):
        '''
        Trims video based on offset
        '''
        if suffix == None: 
            suffix = 'trimmed'
        if offsets is None:
            offsets = self.offsets
        for i,vidfile in enumerate(self.video_files):
            seconds = offsets[i]
            (path2fname, vname) = os.path.split(vidfile)
            final_vidname = os.path.join(path2fname,vname.split('.')[0]+'_'+suffix+'.'+vname.split('.')[-1])
            command = 'ffmpeg -y -ss ' + str(seconds) + ' -i ' + vidfile + ' -c copy ' + final_vidname
            subprocess.Popen(command, shell=True)




