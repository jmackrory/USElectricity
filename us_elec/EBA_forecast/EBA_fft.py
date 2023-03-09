#A Bunch of FFT filtering routines I wrote to help detrend the
# hourly electricity data from the EBA dataset, from the EIA.
# Goes with EIA_explore.ipynb.

import numpy as np

def remove_square_peak(Y,f,center,width):
    """remove_yearly
    Assumes there is a yearly trend.
    Subtracts off everything on a monthly or longer timescale. (around 1/30)
    Replaces that with the average of the neighbouring points.
    
    inputs:
    Y - initial centered Fourier transform
    f - list of frequencies Fourier transform is evaluated a
    shape - function to use to define the window.  Takes a position input, and width. 
    center - frequency to center filter at, to remove        
    width - width of the filter.

    return:
    detrended -transform after subtracting off this component.  
    trend     -the subtracted portion.
    """ 
    #find stuff within +/- 1 width
    trend_msk= abs(f-center)<width
    #find stuff within +/- 1.5 widths, and not inside 1 ith
    mean_msk = abs(f-center)<1.5*width
    mean_msk = mean_msk & ~trend_msk

    replace_avg = Y[mean_msk].mean()
    replace_std = Y[mean_msk].std()
    trend=np.zeros(len(f))+0j
    trend[trend_msk] = Y[trend_msk]-replace_avg
    detrend = Y-trend
    return trend, detrend

def remove_sinc_peak(Y,f,center,width):
    """remove_sinc_peak
    Assumes there is a peak described by a sinc (fro mthe truncated FFT)
    Tries to set the peak height based on the value of the FFT at the peaks
    Subtracts off a sinc function with that amplitude. 
    Replaces that with the average of the neighbouring points.
    
    inputs:
    Y - initial centered Fourier transform
    f - list of frequencies Fourier transform is evaluated a
    shape - function to use to define the window.  Takes a position input, and width. 
    center - frequency to center filter at, to remove        
    width - width of the filter.

    return:
    detrended -transform after subtracting off this component.  
    trend     -the subtracted portion.
    """ 
    #find stuff within +/- 1 width
    trend_msk= abs(f-center)<width
    #find stuff within +/- 1.5 widths, and not inside 1 ith
    replace_avg = Y[trend_msk].mean()
    trend = replace_avg*sinc((f-center)/width)
    detrend = Y-trend
    return trend, detrend

def sinc(x):
    """sinc(x)
    Computes sin(x)/x, with care to take correct limit at x=0
    """
    msk=(abs(x)>1E-16)
    s=np.zeros(len(x))
    s[~msk]=1
    s[msk]=np.sin(x[msk])/x[msk]
    return s

def fft_detrend(F,f,width,remove_func):
    """detrend(F,f,width,remove_func)
    
    Removes mean, annual, daily and weekly trends in data
    by filtering the FFT.

    inputs:
    F - Fourier transformed function
    f - frequency list (assumed to be scaled so 1 = 1/day)
    width - frequency width to apply on filter
    remove_func - functional form of the filter.

    return:
    F_trend_tot - total trend removed
    F_detrend   - detrended function.
    """ 

    F_detrend=F
    F_trend_tot=np.zeros(len(F))+0j
    #remove mean/annual oscillations
    F_trend,F_detrend=remove_func(F,f,0,width)
    F_trend_tot+=F_trend
    #remove daily oscillations
    for k in [1,2]:
        #positive peak
        F_trend,F_detrend=remove_func(F_detrend,f,k,width)
        F_trend_tot+=F_trend
        #negative peak
        F_trend,F_detrend=remove_func(F_detrend,f,-k,width)
        F_trend_tot+=F_trend

    #remove weekly oscillations
    for i in range(1,6):
        f0=i/7
        F_trend,F_detrend=remove_func(F_detrend,f,f0,width)
        F_trend_tot+=F_trend
        F_trend,F_detrend=remove_func(F_detrend,f,-f0,width)
        F_trend_tot+=F_trend

    return F_trend_tot,F_detrend

#Try smoothing the frequency spectrum to extract the background?
def moving_avg(Y,width):
    """moving_avg(Y, width)
    Compute moving average by differencing the cumulative sum.
    """
    Ycum = np.cumsum(Y)
    Ysmooth=np.zeros(len(Y))+0j
    Ysmooth[width:-width]=(Ycum[2*width:]-Ycum[:-2*width])/(2*width)
    return Ysmooth    

def invert_fft(Y):
    """invert_fft(Y)
    Helper routine to carry out fftshifts, invert the fft,
    and then take the real part of a presumed Fourier transform Y.
    """
    #undo the fftshifts, invert fft, and take the real part
    y=np.fft.fftshift(Y)
    y=np.fft.ifft(y)
    y=np.fft.fftshift(y)
    y=np.real(y)
    return y


def avg_extremes(df,window=2):
    """avg_extremes(df)
    Replace extreme outliers, or zero values with the average on either side.
    Suitable for occasional anomalous readings.
    """
    mu=df.mean()
    sd=df.std()
    msk1=(df-mu)>4*sd
    msk2 = df==0
    msk=msk1|msk2
    print( "Number of extreme values {}. Number of zero values {}".format(sum(msk1),sum(msk2)))
    ind= np.arange(len(df))[msk]
    for i in ind:
        df.iloc[i]=(df.iloc[i-window]+df.iloc[i-window])/2

    return df

def remove_na(df,window=2):
    """remove_na(df)
    Replace all NA values with the mean value of the series.
    """
    na_msk=np.isnan(df.values)
    #first pass:replace them all with the mean value - if a whole day is missing.
    print( "Number of NA values {}".format(sum(na_msk)))
    df[na_msk]=df.mean()

    ind= np.arange(len(df))[na_msk]
    #for isolated values, replace by the average on either side.    
    for i in ind:
        df.iloc[i]=(df.iloc[i-window]+df.iloc[i-window])/2
    return df

def run_plots(df_joint):
    """run_plots

    Script function to plot trends, extract trends.

    """
    dem_t=df_joint['Demand']['2015-07':'2016-06'].copy()
    dem_t=avg_extremes(dem_t)
    dem_t=remove_na(dem_t)
    dem_tv=dem_t.values

    #set up FFT time/frequency scales
    Nt = len(dem_tv)
    #scale time to days.
    Tmax = Nt/24
    dt = 1/24
    t = np.arange(0,Tmax,dt)
    df = 1/Tmax
    fmax=0.5/dt
    f = np.arange(-fmax,fmax,df)

    #carry out fft 
    dem_f=np.fft.fftshift(dem_tv)
    dem_f=np.fft.fft(dem_f)
    dem_f=np.fft.ifftshift(dem_f)

    plt.figure(figsize=(8,5))
    spec=abs(dem_f)**2
    spec/=sum(spec)
    plt.semilogy(f,spec)
    fcut=1/7
    plt.axis([-10*fcut,10*fcut,1E-6,1])
    plt.xlabel('Frequency (1/day)')
    plt.ylabel('Normalized Demand Power Spectrum')
    plt.show()

    #remove the trend by assuming square peaks, with a width 4/365.
    #Does seem like a sinc would be a better choice. Perhaps forgot some phases, because that did worse. 
    f_trend_tot,f_detrend = fft_detrend(dem_f,f,4/365,remove_square_peak)
    #now take a rolling average of the remainder.
    dem_f_s=moving_avg(f_detrend,50)
    f_trend_tot+=dem_f_s
    f_detrend-=dem_f_s
    plt.figure(figsize=(8,5))
    w=1
    plt.axis([-0.2,5,1E3,1E8])
    plt.semilogy(f,abs(f_trend_tot),label='Estimated Trend')
    plt.semilogy(f,abs(f_detrend),label='Detrended Spectrum')
    plt.semilogy(f,5E4/(1+(f/w)**2),label='Estimated Background')
    plt.show()

    #check out what this detrending looks like.
    t_trend=invert_fft(f_trend_tot)
    t_detrend=invert_fft(f_detrend)

    t_trend=pd.Series(t_trend,index=dem_t.index)
    t_detrend=pd.Series(t_detrend,index=dem_t.index)

    ti = dem_t.index
    plt.figure(figsize=(8,5))
    plt.plot(ti,dem_t,'b',ti,t_trend,'r',ti,t_detrend,'g')
    #plt.axis([550,560,min(t_detrend),max(dem_t)])
    plt.legend(['Demand','FFT Trend','Demand-Trend'])
    plt.show()
