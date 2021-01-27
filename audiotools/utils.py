import numpy as np
import os
import sys
import IPython.display as ipd
import soundfile as sf
import resampy


from . import processing



__all__ = [
    'play',
    'getAudio'
    'resample'
]



def play(x, fs = 44100):
    ''' Plays audio signal (similar to soundsc in Matlab)


    Parameters
    ----------
        x : np.array (nCh, nSample)
            Signal to be played, either 1 or 2 channels
        fs : int (default = 44100)
            Sampling Frequency
        normalize : bool (default = True)
            True If audio should be normalize before playback
        interactive : bool (default = True)
            True if interactive player is wanted. This only works in ipython notebooks


    '''

    # If interactive is true, but script is not in a .ipynb
    interactive = ('ipykernel' in sys.modules)

    if interactive:
        return ipd.Audio(x, rate=fs)


def isAudioFile(x):
    ''' returns true if input is a supported audiofile

    '''
    validFormats = ("wav", "ogg", "mp3", "flac")
    return (x.lower()).endswith(validFormats)

def getAudio(soundcollection='HINT', audioFolder='audio'):
    ''' Returns a random audio file from an audio collection

    Paramters
    ---------
    soundcollection : str('HINT' | 'EMIME' | 'ICRA')
            Type of sound is to be loaded, default is HINT

    rootfolder : str filepath
            path to root folder. Default takes file from this repo



    Returns
    ------
    x : np.array
            Audio file
    fs : int
            sampling frequency

    '''


    if isAudioFile(soundcollection):
        x, fs = sf.read(soundcollection)
        x = x.reshape(1, -1)
        return x,fs

    audioDir = None
    if soundcollection == 'HINT':
        if audioFolder == 'default':
            audioDir = '../audio/speech/HINT/TRAIN/'
        else:
            audioDir = audioFolder + "/speech/HINT/TRAIN"


    elif soundcollection == 'ICRA':
        if audioFolder == 'default':
            audioDir = '../audio/noise/ICRA/'
        else:
            audioDir = audioFolder + "/noise/ICRA/"

    else:
        audioDir = soundcollection

    files = [f for f in os.listdir(audioDir) if isAudioFile(f)]
    filename = np.random.choice(files)
    x, fs = sf.read(audioDir + "/" + filename)
    x = x.reshape(1,-1)

    return x, fs

def genSinusoids(freq=[100],amp=[1],T=1,fs=44100):
    N = int(T*fs)
    n = np.arange(N)

    freq = np.array(freq).ravel()
    amp = np.array(amp).ravel()

    if len(amp) != len(freq):
        amp = np.ones(len(freq))*amp[0]

    x = np.sum([a * np.sin(2 * np.pi * f * n / fs) for a, f in zip(amp, freq)], axis=0)
    return x.reshape(1,-1),fs


def rms(x):
    ''' Returns the rms value of x
    Paramters
    ---------
        x : np.array
            input signal

    '''
    if np.ndim(x) == 1:
        x = x.reshape(1,-1)

    return np.sqrt(np.mean(x**2,axis=1))

def to_dB(x,power=False):
    ''' magnitude to dB conversion '''
    return 10*np.log10(np.abs(x+1e-99)) if power else 20*np.log10(np.abs(x+1e-99))

def from_dB(x,power=False):
    ''' dB to mag conversion'''
    return 10**(x/10) if power else 10**(x/20)

def resample(x, fsFrom, fsTo):
    return resampy.resample(x, fsFrom, fsTo)


def sameLength(a, b):
    ''' Concatenates zero to either signal a or b to make them same length

    Parameters
    ----------
    a : np.array(nch,nsamples)
        input a
    b : np.array(nch, nsamples)
        input b

    Returns
    -------
    a,b : np.array(nch,nsampels)
        outputs of same length

    '''
    diff = a.shape[1] - b.shape[1]

    if diff > 0:
        b = np.c_[b, np.zeros((b.shape[0], diff))]
    else:
        a = np.c_[a, np.zeros((a.shape[0], np.abs(diff)))]

    return a, b



def adjustSNR(x, y, snrdB):
    ''' adjusts the gain of y to get desired snr

    Parameters
    ----------
        x : np.array(nch, nsamples)
            input signal
        y : np.array(mch,nsamples)
            input noise (to be adjusted)
        snrdB : float
            desired SNR

    Returns
    -------
        y_ : np.array(nch,nsamples)
            adjusted y
        gain : float
            the gain applied to y to obtain desired snr


    '''

    snr = calcSNR(x, y, dB=False)
    # print(snr)
    gain = np.sqrt(snr / 10 ** (snrdB / 10))
    y_ = (gain * y.T).T

    return y_, gain


def calcSNR(x, y, dB=True):
    '''
    Computes the SNR of x and y, using energy in signal

    Parameters
    ----------
        x : np.array(nch, nsamples)
            input signal
        y : np.array(mch,nsamples)
            input noise
        dB : bool
            true for output in dB

    Returns
    -------
        snr : float
            SNR of x and y

    '''
    energyX = (x ** 2).sum(axis=1 if x.ndim == 2 else 0)
    energyY = (y ** 2).sum(axis=1 if y.ndim == 2 else 0)

    if dB:
        return 10 * np.log10(energyX / (energyY+1e-99))
    else:
        return energyX / (energyY+1e-99)


def ssn(n=10, soundcollection='HINT', audioFolder='audio'):
    ''' Generates Speech Shaped Noise from n random audio files'''

    audioDir = None
    if soundcollection == 'HINT':
        if audioFolder == 'default':
            audioDir = '../audio/speech/HINT/TRAIN/'
        else:
            audioDir = audioFolder + "/speech/HINT/TRAIN"


    elif soundcollection == 'ICRA':
        if audioFolder == 'default':
            audioDir = '../audio/noise/ICRA/'
        else:
            audioDir = audioFolder + "/noise/ICRA/"
    else:
        audioDir = audioFolder

    if (type(audioDir) is not np.ndarray):
        files = [f for f in os.listdir(audioDir) if isAudioFile(f)]
        filenames = np.random.choice(files, n)

        data = []
        for fil in filenames:
            x, fs = sf.read(audioDir + "/" + fil)
            # x = x.reshape(1,-1)

            data.append(x)
    else:
        data = audioDir
        fs = 0

    xlong = np.concatenate(data)
    X = np.fft.fft(xlong)
    N = np.abs(X) * np.exp(1j * 2 * np.pi * np.random.randn(len(X)))
    n = np.fft.ifft(N).real

    return n, fs


def genNoise(T, fs, modDepth=0.5, modRate=5, noiseSource=None):
    if T > 60:
        N = T
    else:
        N = int(T * fs)
    n = np.arange(N)

    if noiseSource is not None:
        noiseSource = noiseSource[0:N].reshape(1, -1)

    else:
        noiseSource = np.random.randn(1, N)

    modFilter = processing.butter(f=modRate, fs=fs)
    modSource = np.random.randn(1, N + fs)
    modSource = modFilter.transform(modSource)[:, -N::]

    #modSource -= modSource.mean()
    modSource = (modSource-modSource.min()) / (modSource.max()-modSource.min())

    return noiseSource * ((1 - modDepth) + modDepth * modSource)


def xcorr(x, y, maxlags, normalize=True):
    Nx = len(x)
    r = np.correlate(x, y, mode=2)

    #Nx = Nx if normalize else 1
    r = r[Nx - 1 - maxlags:Nx + maxlags] / (Nx if normalize else 1)
    return r


def coh(x,y,maxlags=False):
    # Elements in x
    Nx = len(x)
    # Correlation, similar to xcorr(x,y) in matlab
    r = np.correlate(x,y,2)
    # Normalization
    normalization = np.sqrt(np.sum(x**2)*np.sum(y**2))
    # Absolute value
    r = np.abs(r/normalization)
    # Remove elements where |tau| >= maxlags
    if maxlags:
        r = r[Nx - 1 - maxlags:Nx + maxlags]
    return r



if __name__ == '__main__':
    print("Hello")
    x, fs = getAudio(audioFolder='../audio')


    print(rms(x))