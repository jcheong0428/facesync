from __future__ import division

'''
    FaceSync Utils Class
    ==========================================
    VideoViewer: Watch video and plot data simultaneously.
    AudioAligner: Align two audios manually

'''
__all__ = ['VideoViewer','AudioAligner']
__author__ = ["Jin Hyun Cheong"]
__license__ = "MIT"

import os
import numpy as np
import matplotlib.pyplot as plt

def VideoViewer(path_to_video, data_df):
    """
    This function plays a video and plots the data underneath the video and moves a cursor as the video plays. 
    Plays videos using Jupyter_Video_Widget by https://github.com/Who8MyLunch/Jupyter_Video_Widget 
    Currently working on: Python 3
    For plot update to work properly plotting needs to be set to: %matplotlib notebook 
    
    Args: 
        path_to_video : file path or url to a video. tested with mov and mp4 formats.
        data_df : pandas dataframe with columns to be plotted. (plotting too many column can slowdown update)
        
    """
    from jpy_video import Video

    from IPython.display import display, HTML

    display(HTML(data="""
    <style>
        div#notebook-container    { width: 95%; }
        div#menubar-container     { width: 65%; }
        div#maintoolbar-container { width: 99%; }
    </style>
    """))
    
    f = os.path.abspath(path_to_video)
    wid = Video(f)
    wid.layout.width='640px'
    wid.display()
    
    fps = wid.timebase**-1 # time base is play rate hard coded at 30fps 
    
    fig,ax = plt.subplots(1,1,figsize=(9,3)) # hardcode figure size for now..
    t=wid.current_time
    ax.axvline(fps*t,color='k',linestyle='--') # cursor is always first of ax 
    # plot each column
    data_df.plot(ax=ax)
    ax.set_xlabel('Time')
    plt.tight_layout()
    
    def plot_dat(ax,t,fps=fps):
        if ax.lines:
            ax.lines[0].set_xdata([np.round(fps*t),np.round(fps*t)])
        fig.canvas.draw()

    def on_value_change(change,ax=ax,fps=fps):
        if change['name']=='_event':
            plot_dat(ax=ax, t=change['new']['currentTime'],fps=fps)
            
    #  call on_value_change that will call plotting function plot_dat whenever there is cursor update 
    wid.observe(on_value_change)


def AudioAligner(original, sample, search_start=0.0,search_end=15.0, xmax = 60,manual=False,reduce_orig_volume=1):
    """
    This function pull up an interactive console to find the offsets between two audios.
    
    Args:
        original: path to original audio file (e.g. '../audios/Aimee.wav')
        sample: path to the sample audio file (e.g. '../audios/Jin.wav')
        search_start(float): start range for slider to search for offset
        search_end(float): end range for slider to search for offset
        xmax(int): Range of audio to plot from beginning
        manual(bool): set to True to turn off auto-refresh
        reduce_orig_volume(int or float): Original wav sounds are often larger so divide the volume by this number. 
    """
    import scipy.io.wavfile as wav
    from IPython.display import Audio
    from IPython.display import display
    from ipywidgets import widgets
    
    orig_r,orig = wav.read(original)
    orig = orig/reduce_orig_volume
    if np.ndim(orig) >1:
        orig = orig[:,0]
    tomatch_r,tomatch = wav.read(sample)
    tomatch = tomatch[:,0]

    fs = 44100

    def audwidg(offset,play_start):
        allshift = play_start
        samplesize = 30
        tomatchcopy = tomatch[int((allshift+offset)*tomatch_r):int((allshift+offset)*tomatch_r)+fs*samplesize]
        shape = tomatchcopy.shape[0]
        origcopy = orig[int((allshift)*tomatch_r):int((allshift)*tomatch_r)+fs*samplesize]
        if origcopy.shape[0] < tomatchcopy.shape[0]:
            diff = tomatchcopy.shape[0] - origcopy.shape[0]
            origcopy = np.pad(origcopy, pad_width = (0,diff),mode='constant')
        toplay = origcopy/reduce_orig_volume+tomatchcopy
        display(Audio(data=toplay,rate=fs))

    def Plot_Audios(offset,x_min,x_max):
#         print('Precise offset : ' + str(offset))
        fig,ax = plt.subplots(figsize=(20,3))
        ax.plot(orig[int(fs*x_min):int(fs*x_max)],linewidth=.5,alpha=.8,color='r')
        ax.plot(tomatch[int(fs*x_min)+int(fs*offset) : int(fs*x_max)+int(fs*offset)],linewidth=.5,alpha=.8)
        ax.set_xticks([(tick-x_min)*fs for tick in range(int(x_min),int(x_max+1))])
        ax.set_xticklabels([tick for tick in range(int(x_min),int(x_max)+1)])
        ax.set_xlim([(x_min-x_min)*fs, (x_max-x_min)*fs] )
        ax.set_ylabel('Audio')
        ax.set_xlabel('Target Audio Time')
        audwidg(offset,x_min)
        plt.show()

    widgets.interact(Plot_Audios,
                     offset=widgets.FloatSlider(value = 0.5*(search_start+search_end), readout_format='.3f', min = float(search_start), max = float(search_end), step = 0.001, 
                                                description='Adjusted offset: ',layout=widgets.Layout(width='90%')),
                     x_min=widgets.FloatSlider(description='Min X on audio plot', value=0.0,min=0.0,max=xmax,step=0.5, layout=widgets.Layout(width='50%')),
                     x_max=widgets.FloatSlider(description='Max X on audio plot', value=xmax,min=0.0,max=xmax,step=0.5, layout=widgets.Layout(width='50%')),
                     __manual=manual
                    )
