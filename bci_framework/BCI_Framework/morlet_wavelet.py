
#sig = np.zeros(128)
#sig[64] = 1000
#wavelet = signal.morlet
#widths = np.arange(1, 30)
##cwtmatr = signal.cwt(sig, wavelet, widths)
#output = np.zeros([len(widths), len(sig)])
#for ind, width in enumerate(widths):
#    wavelet_data = wavelet(min(10 * width, len(sig)), width)
##    wavelet_data = wavelet(width)
#    output[ind, :] = signal.convolve(sig, wavelet_data, mode='same')
#
#import pylab
#for i in range(10):
#    pylab.plot(np.abs(output[i,:]))
#pylab.show()

#%    [TFR,T,F,WT]=TFRSCALO(X,T,WAVE,FMIN,FMAX,N,TRACE) computes 
#%    the scalogram (squared magnitude of a continuous wavelet
#%    transform). 
#%
#%    X : signal (in time) to be analyzed (Nx=length(X)). Its
#%        analytic version is used (z=hilbert(real(X))).  
#%    T : time instant(s) on which the TFR is evaluated 
#%                             (default : 1:Nx).
#%    WAVE : half length of the Morlet analyzing wavelet at coarsest 
#%         scale. If WAVE = 0, the Mexican hat is used. WAVE can also be
#%           a vector containing the time samples of any bandpass
#%           function, at any scale.            (default : sqrt(Nx)). 
#%    FMIN,FMAX : respectively lower and upper frequency bounds of 
#%        the analyzed signal. These parameters fix the equivalent
#%        frequency bandwidth (expressed in Hz). When unspecified, you
#%        have to enter them at the command line from the plot of the
#%        spectrum. FMIN and FMAX must be >0 and <=0.5.
#%    N : number of analyzed voices.
#%    TRACE : if nonzero, the progression of the algorithm is shown
#%                                         (default : 0).
#%    TFR : time-frequency matrix containing the coefficients of the
#%        decomposition (abscissa correspond to uniformly sampled time,
#%        and ordinates correspond to a geometrically sampled
#%        frequency). First row of TFR corresponds to the lowest 
#%        frequency. When called without output arguments, TFRSCALO
#%        runs TFRQVIEW.
#%    F : vector of normalized frequencies (geometrically sampled 
#%        from FMIN to FMAX).
#%    WT : Complex matrix containing the corresponding wavelet
#%        transform. The scalogram TFR is the square modulus of WT.


from scipy.signal import convolve
from scipy import signal
import numpy as np
import sys
import math

def morlet_wavelet_transform(X,time,wave,fmin,fmax,N):
#function [tfr,t,f,wt]=tfrscalo(X,time,wave,fmin,fmax,N,trace)
    
    xrow = X.shape
    N=xrow[0]
    
    tcol = time.shape[0]
    
    s = X - np.mean(X)  
    z = signal.hilbert(s)
    
    fmin_s = str(fmin) 
    fmax_s = str(fmax) 
    N_s = str(N)
    
    if fmin >= fmax:
        print('FMAX must be greater or equal to FMIN')
        sys.exit()
    elif fmin<=0.0 or fmin>0.5:
        print('FMIN must be > 0 and <= 0.5')
        sys.exit()
    elif fmax<=0.0 or fmax>0.5:
        print('FMAX must be > 0 and <= 0.5')
        sys.exit()
    
    f = np.logspace(np.log10(fmin), np.log10(fmax), N)
    a = np.logspace(np.log10(fmax/fmin), np.log10(1), N) 

    wt = np.zeros((N,tcol)) + 0J
    tfr = np.zeros((N,tcol))
    
    for ptr in range(N):
        nha = wave*a[ptr]
        tha = np.arange(-round(nha), round(nha)+1)
        ha  = np.exp(-(2*np.log(10)/(nha**2))*np.power(tha,2)) * np.exp(2J*math.pi*f[ptr]*tha) 
        detail = np.divide(convolve(z,ha),float(math.sqrt(a[ptr])))
        detail = detail[int(round(nha)):(len(detail)-int(round(nha)))] 
        wt[ptr,:]  = detail[time]
        tfr[ptr,:] = detail[time] * detail[time].conjugate()
     
    
    t = time
    f = f.T
    
#   Normalization
    SP = np.fft.fft(z) 
    indmin = round(fmin * (xrow[0] - 2))
    indmax = 1 + round(fmax * (xrow[0] - 2))
    SPana = SP[indmin:indmax]
    tfr = tfr * np.linalg.norm(SPana)**2 / integ2d(tfr,t,f)/N
    
    return (tfr,t,f,wt)
#    if (nargout==0),
#     tfrqview(tfr,hilbert(real(X)),t,'tfrscalo',wave,N,f)
#    end
def integ2d(mat,x,y):

    [M,N] = mat.shape
    
    xc = x.shape[0]
    yr = y.shape[0]
    
    
    mat = (sum(mat.T).T - mat[:,1]/2.0-mat[:,N-1]/2.0)*(x[2]-x[1]) 
    dmat = mat[0:M-1]+mat[1:M] 
    dy = (y[1:M]-y[0:M-1])/2.0 
    som = sum(dmat*dy)
    return som 

def margtfr(tfr):
    
    [M,N] = tfr.shape
    t = np.arange(N)
    f = np.arange(M)

    E     = (integ2d(tfr,t,f)/float(M)).real
    margt = (integ(tfr.T,f.T)/float(M)).real
    margf = (integ(tfr,t)/float(N)).real
    return (margt,margf,E)

def integ(y,x):
     
    [M,N] = y.shape


    Nx = x.shape[0]
#     Nx = x.shape
#     if (Mx!=1):
#         print('X must be a row-vector')
    if (N!=Nx):
#     if (N!=Nx):
        print('Y must have as many columns as X')
    elif (N==1 and M>1):
        print('Y must be a row-vector or a matrix')
 
    dy = y[:,0:N-1] + y[:,1:N]
    dx = (x[1:N]-x[0:N-1])/2.0
    som = np.dot(dy, dx)

    return som

if __name__ == "__main__":
    
    print range(10)
    print signal.hilbert([1,2,3,4])
    sig = np.zeros(128)
    sig[64] = 1
    (tfr,t,f,wt) = morlet_wavelet_transform(sig,np.arange(128),6,0.05,0.45,128)

    import pylab
    for i in range(128):
        pylab.plot(np.abs(tfr[i,:]))
    pylab.show()
