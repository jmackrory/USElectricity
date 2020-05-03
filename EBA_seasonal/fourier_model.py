#Now to do some simple Fourier Series fitting too.
import numpy as np
import pandas as pd

pi = np.pi

class fourier_model(object):
    """ Compute Fourier Series assuming annual, weekly, and daily
    oscillations.
    """
    def __init__(self,n_max):
        self.n_max=n_max
        self.coeff=[]

    def total_fourier_series(self,D):
        """fourier_series
        Fits pandas time series D, with Fourier series. 
        Uses Pandas DateTimeIndex for times.
        Produces fourier series with annual, daily and weekly oscillations to fourier series.
        Computes coefficients, and then series.  Returns both
        D - demand (values to be fitted)
        T - DatetimeIndex
        n_max - maximum number of coefficients

        Note:Misses Holidays.
        """
        T=D.index
        T_dayofyear = T.dayofyear.values
        T_dayofweek = T.dayofweek.values
        T_hour = T.hour.values
        #split time up by day of year, weekly hour, and daily hour
        Tfit = [T_dayofyear/365,
                (T_dayofweek*24+T_hour)/168,
                 T_hour/24]
        Nt = len(D)
        ftot = np.zeros(Nt)
        self.coeff=[[np.sum(D)/Nt,0]]    
        for i in range(3):
            if self.n_max[i]>0:
                 ci = self.fit_fourier_series(D,Tfit[i], self.n_max[i])
                 ftot += self.fourier_series(ci,Tfit[i])             
            else:
                ci=None
            self.coeff.append(ci)
        #add on constant    
        ftot+= self.coeff[0][0]
        ftot=pd.Series(ftot,index=T)         
        return ftot

    def fit_fourier_series(self,D,T,nmax):
        """Fits the Fourier series to data D, on times T,
        and returns parameters.
        """
        Nt = len(D)
        #initial zero coefficients
        coeff=[]
        for n in range(1,nmax+1):
            an= 2*np.sum(np.cos(2*pi*n*T)*D)/Nt
            bn= 2*np.sum(np.sin(2*pi*n*T)*D)/Nt
            coeff.append([an,bn])
        return coeff                       

    def fourier_series(self,coeff,T):
        """fourier_series
        Make simple Fourier series with specified coefficients, 
        over time T. 
        T assumed to be in range [0,1).
        """
        i=1
        nmax=len(coeff)
        f=np.zeros(T.shape)
        #need a +1 somewhere due to 0-indexing,
        #and not including constant
        for n in range(nmax):
            an, bn = coeff[n]
            f += an*np.cos(2*pi*(n+1)*T)+bn*np.sin(2*pi*(n+1)*T)
        return f
