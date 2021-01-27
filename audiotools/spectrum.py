import numpy as np
from scipy import signal
from . import utils
from . import processing as pr
import matplotlib.pyplot as plt


class STFT(pr.audioProcess):
    ''' STFT transformer

    Parameters
    ----------
        nperseg : int (default = 256)
            amounts of samples per window

        noverlapratio : float (default = 0.75)
            overlap ratio

        nfft : int (default = 4)
            fft size, if nfft is 16 or below, then nfft = nearestLog2(nfft * nperseg)

        inv : bool (default = False)
            true if ISTFT, false if STFT

        fs : integer (default = 44100)
            sampling frequency

    Attributes
    ----------
        fs_stft : int
            "sampling frequency" in STFT domain, much lower than fs

    '''

    def __init__(self, nperseg=256, noverlapratio=0.75, nfft=4, inv=False, fs=44100):
        self.nperseg = nperseg
        self.noverlapratio = noverlapratio
        self.noverlap = int(noverlapratio * nperseg)

        if nfft < 16:
            self.nfft = nperseg * nfft
        else:
            self.nfft = nfft

        self.inv = inv

        self.fs = fs
        self.t_stft = (self.nperseg - self.noverlap) / fs
        self.fs_stft = 1 / self.t_stft


    def transform(self, X,inv=None):

        self.inv = inv if inv is not None else self.inv


        if self.inv:
            # Divide by 2
            # X /= 2

            self.t_out, self.x_out = signal.istft(X, self.fs, window='hann', nperseg=self.nperseg,
                                                  noverlap=self.noverlap, nfft=self.nfft)

        else:
            self.f, self.t, self.x_out = signal.stft(X, self.fs, window='hann', nperseg=self.nperseg,
                                                     noverlap=self.noverlap, nfft=self.nfft)
            # Multiply by 2, not sure how to compensate for DC
            # self.x_out *= 2

        return self.x_out


    def getInfo(self):
        return {
            'process' : 'STFT',
            'inv' : self.inv,
            'nperseg' : self.nperseg,
            'nfft' : self.nfft,
            'noverlapratio' : self.noverlapratio,
            'fs_stft' : self.fs_stft,
            'fs' : self.fs
        }

    def plot(self, X=False):

        if X is not False:
            self.x_out = X

        if self.inv:
            print("Hello")
            plt.plot(self.x_out.T)

        else:
            plt.pcolormesh(self.t, self.f, 20 * np.log10(1e-99 + np.abs(self.x_out[0])))



class welch(pr.audioProcess):
    ''' Welch method for spectrum analysis. Can be used as input to feature extraction methods or for visualization
    
    Parameters
    ----------
        nperseg : int (default = 256)
            amounts of samples per window

        noverlapratio : float (default = 0.75)
            overlap ratio

        nfft : int (default = 4)
            fft size, if nfft is 16 or below, then nfft = nearestLog2(nfft * nperseg)

        window : string (default = 'hann')
            window type

        fs : integer (default = 44100)
            sampling frequency
    
    
    '''

    def __init__(self,nperseg=256,noverlapratio=0.75,nfft=4,window='hann',fs=44100,scaling='spectrum'):
        self.nperseg = nperseg
        self.noverlapratio = noverlapratio
        self.noverlap = int(noverlapratio * nperseg)

        if nfft < 16:
            self.nfft = nperseg * nfft
        else:
            self.nfft = nfft

        self.window = window
        self.fs = fs
        self.scaling = scaling

    def transform(self, X):
        self.f, self.Pxx = signal.welch(X,self.fs,
                                        window = self.window,
                                        nperseg=self.nperseg,
                                        noverlap=self.noverlap,
                                        nfft=self.nfft,
                                        scaling=self.scaling)



        return self.Pxx


    def plot(self):

        plot = plt.semilogy(self.f,self.Pxx.T)
        plt.title("Welch Method")
        plt.ylabel("PSD [V**2]")
        plt.xlabel("Frequency [Hz]")


        return plot

class modulationSpectrum(pr.audioProcess):
    ''' Computes the modulation spectrum of an input signal
    
    Parameters
    ---------- 
        cf : list (default = [0.5,1,2,4,8,16,32])
            center frequencies
        fs : integer
            sampling frequency
            
        bNorm : bool (default = True)
          normalize  
        
    
    '''

    def __init__(self,cf = [0.5,1,2,4,8,16,32],fs = 44100,bNorm = True):
        self.cf = cf
        self.fs = fs
        self.bNorm = bNorm

    def transform(self, X):

        if np.ndim(X) == 3:
            nCh,nB,nS = X.shape
            X = np.reshape(X,(nB*nCh,nS))
            reshape = True
        else:
            reshape = False

        # Envelope Extraction
        x = np.abs(signal.hilbert(X))

        # Determine low-pass cut-off frequency
        cfHzLP = max(100, 2 * np.around(np.max(self.cf)))

        # Filter
        b, a = signal.butter(4, cfHzLP / (self.fs * 0.5))
        x = signal.lfilter(b, a, x)

        # Downsample
        self.fsEnv = max(1200, min(self.fs, 4 * np.around(np.max(self.cf))))
        x = utils.resample(x, self.fs, self.fsEnv)

        if np.ndim(x) == 1:
            x = x.reshape(1, -1)

        #### Modulation spectrum Analysis ####

        # Input size
        nSubbands, nSamples = x.shape


        # Number of modulation filters
        nFilters = len(self.cf)

        # Allocate memory
        mspecMod = np.zeros((nSubbands, nFilters))
        mspecDC = np.zeros((nSubbands, 1))

        # Determine FFT resolution
        self.nfft = int(2 ** np.ceil(np.log2(x.shape[1])))

        # Frequency axis
        self.freqHz = np.arange(1 + self.nfft / 2) / (self.nfft / 2) / 2 * self.fsEnv

        # Find lower and upper 3dB edge freqencies
        self.fModLowHz = np.array(self.cf) * 2 ** (-1 / 2);
        self.fModHighHz = np.array(self.cf) * 2 ** (1 / 2);

        # BW in hz
        self.bwHz = self.fModHighHz - self.fModLowHz

        # Q-factor
        self.qFactor = self.cf / self.bwHz


        for i in range(nSubbands):
            subband = x[i, :]

            # magnitude spectrum
            mspec = np.abs(np.fft.fft(subband, self.nfft)) / self.nfft
            mspec = mspec[0:1 + int(self.nfft / 2)]

            # Take positive frequencies time two
            # reflect energy of negative frequency
            if np.mod(self.nfft, 2) == 1:
                # If single-sided spectrum is odd, only dc is unique
                mspec[1::] = mspec[1::] * 2
            else:
                # if even, then do not double DC nor nyquist
                mspec[1:-1] = mspec[1:-1] * 2

            # Loop over modulation filters
            for mm in range(nFilters):
                # Find frequency indeces for mm-th modulation filter
                idxMod = (self.fModLowHz[mm] < self.freqHz) & (self.freqHz < self.fModHighHz[mm])
                mspecMod[i, mm] = np.sqrt(np.sum(mspec[idxMod] ** 2))

            mspecDC[i] = mspec[0]

        if self.bNorm:
            mSpec = mspecMod / (mspecDC)
        else:
            mSpec = mspecMod





        if reshape:
            mSpec = np.reshape(mSpec,(nCh,nB,-1))

        self.mSpec = mSpec
        return mSpec

    def plot(self):

        plot = plt.semilogx(self.cf, self.mSpec.T, '-o')
        plt.xticks(self.cf, self.cf)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude")
        plt.grid()
        return plot


class FFT():
    ''' Very simply FFT class'''
    def __init__(self, N=None, fs=44100):
        self.N = N
        self.fs = fs

    def transform(self, x, inv=False):
        if inv:
            return np.fft.ifft(x, n=self.N)
        else:
            self.X = np.fft.fft(x, n=self.N)
            self.L = len(self.X.T)
            self.f = self.fs * np.linspace(0, 1, self.L)

            return self.X

    def plot(self):
        return plt.plot(self.f, 20 * np.log10(2 * np.abs(self.X.T / self.L)))

class IC(pr.audioProcess):
    def __init__(self,t=0.1,fs=1000):
        self.smooth = pr.expsmooth(t=t,fs=fs)

    def transform(self,X):
        phiLL = X[0] * np.conj(X[0])
        phiRR = X[1] * np.conj(X[1])
        phiLR = X[0] * np.conj(X[1])

        phiLL = self.smooth.transform(phiLL).real
        phiRR = self.smooth.transform(phiRR).real
        phiLR = self.smooth.transform(phiLR).real

        self.C = np.abs(phiLR)/np.sqrt(phiLL*phiRR+1e-99)
        return self.C


if __name__ == '__main__':
    print("Hello from spectrum")
    normalizer = pr.normalize(65)
    x,fs = utils.getAudio(audioFolder = '../audio')
    x = normalizer.transform(x)
    x = np.r_[x,x]

    X = STFT().transform(x)
    y = STFT(inv=True).transform(X)

    print(y.shape)

    #modulation spectrum
    #modspec = modulationSpectrum()
    #xms = modspec.transform(x)
    #modspec.plot()
    #plt.show()


    print("Done")