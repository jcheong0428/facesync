from __future__ import division

'''
    FaceSync Utils Class
    ==========================================
    VideoViewer: Watch video and plot data simultaneously.
    AudioAligner: Align two audios manually
    neutralface: points that show a face 
    ChangeAU: change AUs and return new face
'''
__all__ = ['VideoViewer','AudioAligner','neutralface','audict','plotface','ChangeAU']
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
    
    fig,ax = plt.subplots(1,1,figsize=(6.5,3)) # hardcode figure size for now..
    t=wid.current_time
    ax.axvline(fps*t,color='k',linestyle='--') # cursor is always first of ax 
    # plot each column
    data_df.plot(ax=ax)
    ax.set_xlabel('Frames')
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
        original: path to original audio file (e.g. '../audios/original.wav')
        sample: path to the sample audio file (e.g. '../audios/sample.wav')
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
    # volume is often louder on original so you can reduce it
    orig = orig/reduce_orig_volume
    # take one channel of target audio. probably not optimal
    if np.ndim(orig) >1:
        orig = orig[:,0]
    # grab one channel of sample audio
    tomatch_r,tomatch = wav.read(sample)
    tomatch = tomatch[:,0]

    fs = 44100

    def audwidg(offset,play_start):
        allshift = play_start
        samplesize = 30
        tomatchcopy = tomatch[int((allshift+offset)*tomatch_r):int((allshift+offset)*tomatch_r)+fs*samplesize]
        shape = tomatchcopy.shape[0]
        origcopy = orig[int((allshift)*tomatch_r):int((allshift)*tomatch_r)+fs*samplesize]
        # when target audio is shorter, pad difference with zeros
        if origcopy.shape[0] < tomatchcopy.shape[0]:
            diff = tomatchcopy.shape[0] - origcopy.shape[0]
            origcopy = np.pad(origcopy, pad_width = (0,diff),mode='constant')
        toplay = origcopy + tomatchcopy
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
                     x_min=widgets.FloatSlider(description='Min X on audio plot', value=0.0,min=0.0,max=xmax,step=0.1, layout=widgets.Layout(width='50%')),
                     x_max=widgets.FloatSlider(description='Max X on audio plot', value=xmax,min=0.0,max=xmax,step=0.1, layout=widgets.Layout(width='50%')),
                     __manual=manual
                    )
neutralface = {-34: (212, 336),
 -33: (222, 342), -32: (237, 342), -30: (203, 335), -29: (222, 335),
 -28: (237, 328), -26: (227, 288), -25: (238, 292), -19: (201, 219),
 -18: (184, 220), -17: (169, 214), -16: (184, 204), -15: (201, 203),
 -14: (217, 215), -13: (225, 181), -12: (203, 172), -11: (180, 170),
 -10: (157, 174), -9: (142, 180), -8: (122, 222), -7: (126, 255),
 -6: (133, 286), -5: (139, 318), -4: (148, 349), -3: (165, 375),
 -2: (190, 397), -1: (219, 414),
 0: (252, 419),
 1: (285, 414), 2: (315, 398), 3: (341, 377), 4: (359, 351),
 5: (368, 319), 6: (371, 287), 7: (376, 254), 8: (378, 221),
 9: (354, 180), 10: (339, 173), 11: (316, 167), 12: (293, 171),
 13: (270, 180), 14: (281, 215), 15: (296, 203), 16: (314, 202),
 17: (328, 212), 18: (315, 219), 19: (297, 219), 20: (248, 207),
 21: (248, 227), 22: (248, 247), 23: (248, 268), 24: (248, 294),
 25: (260, 291), 26: (271, 287), 27: (248, 333), 28: (262, 328),
 29: (279, 335), 30: (296, 335), 31: (250, 340), 32: (264, 342),
 33: (280, 342), 34: (288, 335)}

audict = {'AU1' : {-11:(2,0),11:(-2,0),-12:(5,-8),12:(-5,-8),-13:(0,-20),13:(0,-20) },
# Brow Lowerer 
'AU4': {-10:(4,5),10:(-4,5),-11:(4,15),11:(-4,15),-12:(5,20),12:(-5,20),-13:(0,15),13:(0,15) },  

# Upper Lid Raiser 
'AU5': {-9:(2,-9),9:(2,-9), -10:(2,-10),10:(-2,-10),-11:(2,-15),11:(-2,-15),-12:(5,-12),12:(-5,-12),-13:(0,-10),13:(0,-10),
                  -16:(0,-10),-15:(0,-10),16:(0,-10),15:(0,-10),
                  -19:(0,10),-18:(0,10),19:(0,10),18:(0,10)}  ,
# cheek raiser
'AU6': {-8:(20,0),8:(-20,0), -7:(10,-5),7:(-10,-5), -6:(2,-8), 6:(-2,-8),
                   -9:(5,5),9:(-5,5),
                  17:(-5,5),18:(-3,-3),19:(-3,-3),
                  -17:(5,5),-18:(3,-3),-19:(3,-3)},
# nose wrinkler
'AU9': {-15:(2,4),15:(-2,4),-14:(2,3),14:(-2,3),
                    20:(0,5), 21:(0,-5), 22:(0,-7), 23:(0,-10), 
                   -26:(5,-15),-25:(0,-15),24:(0,-15),25:(0,-15),26:(-5,-15),
                  -10:(2,0),10:(-2,0),-11:(2,8),11:(-2,8),-12:(5,12),12:(-5,12),-13:(0,10),13:(0,10)
                  },
# Lip corner Puller
'AU12': { -30: (-10,-15), -34: (-5,-5), 
                    30:(10,-15), 34:(5,-5), 
                    -29:(0,0), 29:(0,0) },
# Chin raiser
'AU17': { -2:(5,0),-1:(5,-5),0:(0,-20),-1:(-5,-5),2:(-5,0) },
# Lip Puckerer
'AU18': {-30:(5,0), 30:(-5,0), -34:(5,0), 34:(-5,0),
                    -33:(5,0),33:(-5,0), -29:(5,0),29:(-5,0),30:(-5,0),
                    -28:(0,0),28:(0,0),27:(0,-8),31:(0,10),-32:(0,7),32:(0,7)} ,
# Lips Part
'AU25': {-28:(0,-3),28:(0,-3),27:(0,-5),31:(0,7),-32:(0,7),32:(0,7)} }

def plotface(face):
    """
    This function will take a dictionary of dots by (x,y) coordinates like the neutralface. 
    
    """
    f, ax = plt.subplots(1,1,figsize=(7,7))
    for key in face.keys():
        (x,y) = face[key]
        ax.scatter(x,y)
    ax.set_xlim([0,500])
    ax.set_ylim([0,500])
    ax.invert_yaxis()
    plt.show()
    return ax

def ChangeAU(aulist, au_weight = 1.0, audict = audict, face = neutralface):
    '''
    This function will return a new face with the acti on units of aulist moved based on au_weight. 
    
    Args:
        aulist: list of AUs that are activated currently supported ['AU1','AU4','AU5','AU6','AU9','AU12','AU17','AU18','AU25]
        au_weights = float between 0 and 1.0 to activate all action unit or a dictionary to modular change of action units. 
        audict = Dictionary of AU movements 
        face = neutral face dictionary. 
    '''
    au_weights = {}
    # if dict, apply weight to each au
    if type(au_weight)==dict: 
        au_weights = au_weight
    # if a float apply to all
    elif type(au_weight)==float: 
        for au in audict.keys():
            au_weights[au] = au_weight
    newface = face.copy()
    for au in aulist:
        for landmark in audict[au].keys():
            newface[landmark] = (face[landmark][0] + au_weights[au] * audict[au][landmark][0], face[landmark][1] + au_weights[au] *  audict[au][landmark][1])
    return newface


