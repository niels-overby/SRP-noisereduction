'''
Common processing tools

Gain check
Normalize check
Gate check
Limit check
Filter (IIR) check
Delay
Envelope check

'''

import numpy as np
from scipy import signal
from . import utils
import matplotlib.pyplot as plt

class audioProcess:
    ''' Prototype for audio process class'''

    def __init__(self):
        None

    def transform(self, X):
        return X

    def getInfo(self):
        return {"process": 0}


class pipe(audioProcess):
    ''' concatenate audioprocesses into pipelines

    '''

    def __init__(self,processes):
        self.processes = processes

        self.pipeContents = [p.getInfo()['process'] for p in self.processes]

    def transform(self, X):
        for process in self.processes:
            X = process.transform(X)
        return X

    def getInfo(self):
        return {i : self.pipeContents[i] for i in range(len(self.pipeContents))}

class gain(audioProcess):
    ''' Gain the input signal with fixed dB '''

    def __init__(self,dB=0):
        self.dB = dB
        self.factor = utils.from_dB(self.dB)

    def transform(self, X):

        return X * self.factor

    def getInfo(self):
        return {'process' : 'gain',
                'gain_dB' : self.dB,
                'gain_mag' : self.factor}


class normalize(audioProcess):
    ''' normalizes to a given target level

    Parameters
    ----------
    level : numeric
        Set desired level in dB RMS

    bLink : bool
        True

    '''

    def __init__(self,level=65,bLink=True,):
        self.level = level
        self.bLink = bLink
        self.factor = 1

    def transform(self, X):
        ''' return normalized output'''

        # Compute scaling factor in dB
        self.factor = utils.rms(X) * utils.from_dB(-self.level)
        # max scaling if bLink
        self.factor[:] = np.max(self.factor) if self.bLink else self.factor

        # Return normalized signal
        return (X.T/self.factor).T

    def getInfo(self):
        return {'process' : 'normalize',
                'level' : self.level,
                'bLink' : self.bLink,
                'factor' : self.factor}

    def apply(self,X):
        'use this to apply gain if factor is already computed'
        
        return (X.T/self.factor).T
        

class envelope(audioProcess):
    ''' Envelope extraction

    Parameters
    ----------
    atk : numeric
        attack time constant
    rel : numeric or list
        release time constant, if two rel are given then envelope is scene aware
    fs : int
        sampling frequency
    mask : np.array
        mask indicating where rel[0] and rel[1] is to be used

    '''
    def __init__(self, atk=1e-3, rel=10e-3, fs=44100, mask=False,output_in_dB = False):
        self.atk = np.array(atk)
        self.rel = np.array(rel)
        self.fs = fs

        self.mask = mask * 1

        self.a = 1 - np.exp((-1 / fs) / self.atk)
        self.r = 1 - np.exp((-1 / fs) / self.rel)

        self.output_in_dB = output_in_dB

    def transform(self, X):

        X_ = np.zeros(X.shape)

        if np.ndim(X) <= 2:
            # Time series, nCh x nS
            for ich, Xch in enumerate(X):
                prevSample = np.abs(Xch[0])
                temp = 0

                for isam, sample in enumerate(np.abs(Xch)):
                    r = self.r if (type(self.mask) is int) else self.r[int(self.mask[ich, isam])]
                    a = self.a if (type(self.mask) is int) else self.a[int(self.mask[ich, isam])]

                    if prevSample > sample:
                        temp = r * sample + (1 - r) * temp
                    else:
                        temp = a * sample + (1 - a) * temp

                    X_[ich, isam] = temp
                    prevSample = temp

        if np.ndim(X) == 3:
            # Filterbank, nCh, nF, nS
            for ich, Xch in enumerate(X):
                for ifil, Xfil in enumerate(Xch):
                    prevSample = np.abs(Xfil[0])
                    temp = 0

                    for isam, sample in enumerate(np.abs(Xfil)):
                        #print(self.r)
                        r = self.r if (type(self.mask) is int) else self.r[int(self.mask[ich, ifil, isam])]
                        a = self.a if (type(self.mask) is int) else self.a[int(self.mask[ich, ifil, isam])]

                        if prevSample > sample:
                            temp = r * sample + (1 - r) * temp
                        else:
                            temp = a * sample + (1 - a) * temp

                        X_[ich, ifil, isam] = temp
                        prevSample = temp

        X_ = utils.to_dB(X_+1e-99) if self.output_in_dB else X_
        return X_

    def getInfo(self):
        return {
            'process' : 'envelope',
            'atk' : self.atk,
            'rel' : self.rel,
            'fs' : self.fs,
            'output_in_dB' : self.output_in_dB
        }

class gateLim(audioProcess):
    ''' Combined Gate and Limiter Tool

     Parameters
     ----------
        lowerGate : np.array
            Anything below this value (in dB) gets cutoff
            one value per input channel
        upperLim : np.array
            Anything above this value is limited

        atk : numeric
            attack time constant for envelope detection
            if atk or rel is less or equal to 0, then envelope is not used
        rel : numeric
            release time constant for envelope detection

        fs : integer
            sampling frequency

     '''

    def __init__(self,lowerGate = 0, upperLim = 120, atk = 0, rel = 0,fs=44100):
        self.lowerGate = lowerGate
        self.upperLim = upperLim
        self.atk = atk
        self.rel = rel
        self.fs = fs

        # Create envelope object if envelope is used
        self.envIn = (self.atk+self.rel) != 0
        if self.envIn:
            self.env = envelope(self.atk,self.rel,self.fs,output_in_dB=True)

    def transform(self, X):

        # Compute envelope signal if envelope is used, else use sample peak value
        Xenv = self.env.transform(X) if (self.atk+self.rel != 0) else utils.to_dB(np.abs(X)+1e-99)

        # Gate
        if self.lowerGate is not None:
            X[Xenv < self.lowerGate] = 0
        # Lim
        if self.upperLim is not None:
            X[Xenv > self.upperLim] = np.sign(X[Xenv > self.upperLim]) * 10**(self.upperLim/20)

        return X

    def getInfo(self):
        return {'process' : 'gateLim',
                'lowerGate' : self.lowerGate,
                'upperLim': self.upperLim,
                'atk' : self.atk,
                'rel' : self.rel}



class butter(audioProcess):
    ''' Basic IIR filters, wrapper for scipy butterworth filter

    Parameters
    ----------
        f  : numeric | list
            Cutoff frequency for lp or hp, list with 2 cutoff freq for bp / notch

        ftype : string (default = 'lowpass')
            {'lowpass','highpass','bandpass','bandstop'}
            type of filter
        fs : integer (default = 44100)
            Sampling frequency
        order : int (default = 4)
            filter order

    '''

    def __init__(self,f = 1, ftype='lp',fs = 44100,order = 4):
        self.f = f
        self.ftype = ftype
        self.fs=fs
        self.order = order

        self.nyq = fs/2.0 # Nyqvist frequency

        # Compute coefficients
        self.b,self.a = signal.butter(self.order,np.array(f)/self.nyq,btype=self.ftype)

    def transform(self, X):
        return signal.lfilter(self.b,self.a,X)

    def getInfo(self):
        return {
            'process' : 'butter',
            'f' : self.f,
            'ftype' : self.ftype,
            'fs' : self.fs,
            'order' : self.order,
            'coeffs' : [self.b,self.a]
        }




    def plot(self):
        w,h = signal.freqz(self.b,self.a,fs=self.fs)
        fig, ax1 = plt.subplots()
        ax1.set_title('Digital filter frequency response')

        ax1.semilogx(w, 20 * np.log10(abs(h) + 1e-9), 'b')
        ax1.set_ylabel('Amplitude [dB]', color='b')
        ax1.set_xlabel('Frequency [rad/sample]')

        ax2 = ax1.twinx()
        angles = np.unwrap(np.angle(h))
        ax2.plot(w, angles, 'g')
        ax2.set_ylabel('Angle (radians)', color='g')
        ax2.grid()
        ax2.axis('tight')
        return fig, ax1,ax2



class customFilter(audioProcess):
    def __init__(self,a,b,fs=44100):
        self.a = a
        self.b = b
        self.fs = fs

    def transform(self, X):
        return signal.lfilter(self.b,self.a,X)

    def plot(self):

        w, h = signal.freqz(self.b, self.a, fs=self.fs)
        fig, ax1 = plt.subplots()
        ax1.set_title('Digital filter frequency response')

        ax1.semilogx(w, 20 * np.log10(abs(h) + 1e-9), 'b')
        ax1.set_ylabel('Amplitude [dB]', color='b')
        ax1.set_xlabel('Frequency [rad/sample]')

        ax2 = ax1.twinx()
        angles = np.unwrap(np.angle(h))
        ax2.plot(w, angles, 'g')
        ax2.set_ylabel('Angle (radians)', color='g')
        ax2.grid()
        ax2.axis('tight')
        return fig, ax1, ax2

class expsmooth():

    def __init__(self, t=0.1, fs=44100):
        self.alpha = np.exp(-1 / (t * fs))
        #self.alpha = np.exp()
        self.b = [1 - self.alpha]
        self.a = [1, -self.alpha]

    def transform(self, X):
        return signal.lfilter(self.b, self.a, X)



class hilbertEnv(audioProcess):
    def __init__(self, fs=44100, lowpass=32):
        self.fs = fs
        if lowpass != None:
            self.myFilter = butter(f=lowpass, fs=fs, order=4)
            self.lowpass = lowpass
            self.order = 4
        else:
            self.lowpass = -1
            self.order = -1

    def transform(self, X):
        Y = signal.hilbert(X)
        Y = np.abs(Y)

        if self.lowpass != -1:
            Y = self.myFilter.transform(Y)
            
        return Y

    def getInfo(self):
        return {
            'process' : 'hilbert envelope',
            'fs' : self.fs,
            'lowpass cutoff' : self.lowpass,
            'filter_order' : self.order
        }

class halfwaveEnv(audioProcess):
    def __init__(self, fs=44100, lowpass=32):
        self.fs = fs
        if lowpass != None:
            self.myFilter = butter(f=lowpass, fs=fs, order=4)
            self.lowpass = lowpass
            self.order = 4
        else:
            self.lowpass = -1
            self.order = -1

    def transform(self, X):
        Y = np.copy(X)
        Y[Y < 0] = 0

        if self.lowpass != -1:
            Y = self.myFilter.transform(Y)

        return Y

    def getInfo(self):
        return {
            'process' : 'halfwave envelope',
            'fs' : self.fs,
            'lowpass cutoff' : self.lowpass,
            'filter_order' : self.order
        }


class Trim(audioProcess):
    ''' Trims length of input signal according to a threshold value

    Paramters
    ---------
    thr : float
        Threshold in dB

    '''

    def __init__(self,thr=0):
        self.thr = thr
        self.amin = None
        self.amax = None

    def transform(self,X):
        # X : np.array(nch, nSamples)
        myEnv = envelope()
        xdB = myEnv.transform(X)
        xdB = utils.to_dB(np.abs(X) + 1e-99)
        try:
            self.amin = (xdB > self.thr).argmax()
        except:
            self.amin = 0

        try:
            self.amax = (xdB[:,::-1] > self.thr).argmax()
        except:
            self.amax = 1


        return X[:,self.amin:-self.amax]

    def trim(self,x):
        if self.amin is None:
            return self.transform(x)
        else:
            return x[:,self.amin:-self.amax]


class SampleDelay(audioProcess):
    ''' Shift an input with an amount of samples

    '''


    def __init__(self,amnt=1,direction='h',fs=None):

        if fs != None:
            self.amnt = int(1/fs * amnt)
        else:
            self.amnt = amnt

        self.direction = direction
        self.fs =fs

    def transform(self,X):
        A_ = 0

        if self.amnt==0:
            return X

        if np.ndim(X) == 3:
            A_ = np.roll(X, self.amnt, axis=2)
            if self.amnt >= 0:
                A_[:,:,0:self.amnt] = 0
            else:
                A_[:,:,self.amnt::] = 0

        elif np.ndim(X) == 2:
            A_ = np.roll(X, self.amnt, axis=1)
            if self.amnt >0:
                A_[:,0:self.amnt] = 0
            else:
                A_[:,self.amnt::] = 0

        return A_

    def getInfo(self):
        return {
            'process' : 'Sample Delay',
            'fs' : self.fs,
            'amount' : self.amnt,
            'direction' : self.direction
        }


class Frames(audioProcess):
    def __init__(self,frame_ms=100,frame_skip=0.5,fs=16e3,normalize=False):
        self.frame_ms = frame_ms*1e-3
        self.frame_size = int(fs*self.frame_ms)
        self.frame_skip = int(self.frame_size*frame_skip)
        self.fs = fs
        self.normalize = normalize
        
    def transform(self,x):
        
        # Dimensionality
        nSamples = x.shape[-1] # [... x nSamples]
        nCh = x.shape[0]
        
        # Compute number of frames
        
        nFrames = int(np.floor((nSamples-self.frame_size)/self.frame_skip)+1)
        
        # number of subbands
        if x.ndim == 3:
            nBands = x.shape[1]
        elif x.ndim == 2:
            # In case x = [nCh x nSamples], then reshape to [nCh x 1 x nSamples] 
            # after segmenting into frames reshape back to original
            nBands = 1
            x = x.reshape(nCh,nBands,-1)
            
        # Initialize output matrix
        xFrames = np.zeros((nCh,nBands,nFrames,self.frame_size))
        
        
        for i_ch in range(nCh):
            for i_b in range(nBands):
                for i_f in range(nFrames):
                    idx_start = self.frame_skip * i_f
                    idx_end =  idx_start + self.frame_size
                    #print(idx_start,idx_end)
                    frame = x[i_ch,i_b,idx_start:idx_end]

                    if self.normalize:
                        frame = (frame-frame.mean())/frame.std()

                    xFrames[i_ch,i_b,i_f] = frame
                    
        if nBands==1:
            xFrames = xFrames.reshape(nCh,nFrames,-1)
                    
        return xFrames
    
    def getInfo(self):
        return {
            'process' : 'Frames',
            'fs' : self.fs,
            'frame_ms' : self.frame_ms,
            'frame_size' : self.frame_size,
            'frame_skip' : self.frame_skip
            
        }



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    print("Hello from processing")
    x, fs = utils.getAudio(audioFolder='../audio/')
    x = np.r_[x,x/2]



    myPipe = pipe([normalize(),
                   gain(0),envelope(),
                   butter(f = [400,1200],ftype='bandstop')])


    x = myPipe.transform(x)


    plt.style.use('seaborn')
    myPipe.processes[3].plot()
    plt.xticks([50,100,250,500,1000,2500,5000,10000,15000],[50,100,250,500,1000,2500,5000,10000,15000])
    plt.show()