#A Bunch of FFT filtering routines I wrote to help detrend the
# hourly electricity data from the EBA dataset, from the EIA.
# Goes with EBA_explore.ipynb.

#Also includes filtering functions to handle missing data, and average
#down peaks: remove_na

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf, arma_order_select_ic

def avg_extremes(df,window=2):
    """avg_extremes(df)
    Replace extreme outliers, or zero values with the average on either side.
    Suitable for occasional anomalous readings.
    Input: df - pandas dataframe
           window - integer for number of steps to use in averaging
    Returns df - modified input dataframe
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
    Input: df - pandas dataframe
           window - integer for number of steps to use in averaging
    Returns df - modified input dataframe
    """
    na_msk=np.isnan(df.values)
    #first pass:replace them all with the mean value - if a whole day is missing.
    print( "Number of NA values {}".format(sum(na_msk)))
    #replace with mean for that column.
    df.loc[na_msk]=df.mean()

    ind= np.arange(len(df))[na_msk]
    #for isolated values, replace by the average on either side.    
    for i in ind:
        df.iloc[i]=(df.iloc[i-window]+df.iloc[i-window])/2
    return df

def make_seasonal_plots(dem,temp,per,nlags):
    """Make seasonal decomposition of temperature, and demand curves.
    Plots those decompositions, and their correlation/autocorrelation plots.
    dem- input demand series
    temp-input temperature series
    per - input date to index on for plotting, e.g. '2016-03'
    nlags - number of lags for correlation plots.
    Side_effects: 4x1 array of plots decomposing signals
    2x2 array of autocorrelation plots for residuals and raw data.
    """
    #Carry out the "demand" and "temperature" seasonal decompositions.
    dem_decomposition = seasonal_decompose(dem,two_sided=False)
    dem_mu=dem.mean()
    dem_trend = dem_decomposition.trend/dem_mu  #Find rolling average over most important period.
    dem_seasonal = dem_decomposition.seasonal/dem_mu  #Find the dominant frequency components
    dem_residual = dem_decomposition.resid/dem_mu  #Whatever is left.

    temp_decomposition = seasonal_decompose(temp,two_sided=False)
    temp_mu=temp.mean()
    temp_trend = temp_decomposition.trend/temp_mu  #Find rolling average over most important period.
    temp_seasonal = temp_decomposition.seasonal/temp_mu  #Find the dominant frequency components
    temp_residual = temp_decomposition.resid/temp_mu  #Whatever is left.

    #Plot out the decompositions
    plt.figure(figsize=(8,6))
    plt.title('Normalized Seasonal Decomposition')
    plt.subplot(411)
    plt.plot(dem_trend[per],'b',temp_trend[per],'k')
    plt.ylabel('Trend')
    plt.subplot(412)
    plt.plot(dem_seasonal[per],'b',temp_seasonal[per],'k')
    plt.ylabel('Seasonal Oscillation')
    plt.subplot(413)
    plt.plot(dem_residual[per],'b',temp_residual[per],'k')
    plt.ylabel('Residuals')
    plt.subplot(414)
    plt.plot(dem[per]/dem_mu,'b',temp[per]/temp_mu,'k')
    plt.ylabel('Raw Data')
    plt.show()

    #Plot the auto-correlation plots.
    nlags=np.min([len(dem[per])-1,nlags,len(temp[per])-1])
    print('Nlags',nlags)
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,5))
    plot_acf(temp_residual[per],'b-x','Temp Residual',ax1,ax2,nl=nlags)
    plot_acf(dem_residual[per],'r-+','Demand Residual',ax1,ax2,nl=nlags)
    plt.legend()
    plt.title('ACF for Residual')
    plt.show()
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,5))
    plot_acf(temp[per],'b-x','Temp',ax1,ax2,nl=nlags)
    plot_acf(dem[per],'r-+','Demand',ax1,ax2,nl=nlags)
    plt.title('ACF for Raw')
    plt.legend()
    plt.show()

    return None

def plot_acf(ts,ls,line_label,ax1,ax2,nl=50):
    """plot_acf(ts,ls,line_label,ax1,ax2,nl)
    Plot the auto-correlation plots for a timeseries (ts) up to a given number of lags (nl)
    Give a specific linestyle (ls), and label.
    Inputs:
    ts - time series
    ls - line style to use when plotting
    line_label - label for this times seris
    ax1, ax2 - axes for sub-plots
    nl - maximum number of lags to consider
    """
    #Actually do those auto-corellations, on the series, and its absolute value.
    ts2 = ts[np.isfinite(ts)]
    lag_acf = acf(ts2,nlags=nl)
    lag_pacf=pacf(ts2,nlags=nl,method='ols')
    #5% confidence intervals.
    sd = 1.96/np.sqrt(len(ts2))
    #Make some pretty subplots.
    ax1.axhline(y=sd,color='gray')
    ax1.axhline(y=-sd,color='gray')
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('Auto Correlation',labelpad=2)
    ax1.plot(lag_acf,ls,label=line_label)
    
    ax2.axhline(y=sd,color='gray')
    ax2.axhline(y=-sd,color='gray')
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Partial Auto Correlation',labelpad=2)    
    ax2.plot(lag_pacf,ls,label=line_label)    
    return None



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
    """detrend(dem_f,f,width,remove_func)
    
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

    F_detrend=dem_f
    F_trend_tot=np.zeros(len(dem_f))+0j
    #remove mean/annual oscillations
    F_trend,F_detrend=remove_func(dem_f,f,0,width)
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


def plot_pred(series_list,label_list):
    """make plot to compare fitted parameters"""
    for s,l in zip(series_list,label_list):
        plt.plot(s,label=l)    
    plt.legend()
    plt.show()

def rmse(x,y):
    z = np.sqrt(np.sum((x-y)*(x-y))/len(x))
    return z

def mape(x,y):
    z = np.mean(np.abs((1-x/y)))
    return z    


