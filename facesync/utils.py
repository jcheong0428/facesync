from __future__ import division

'''
    FaceSync Utils Class
    ==========================================
    VideoViewer: Watch video and plot data simultaneously.

'''
__all__ = ['VideoViewer']
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
    