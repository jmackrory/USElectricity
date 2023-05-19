import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def convert_data(df):
    """convert_data(df)
    Converts data frame row-by-row
    from an array to a time-series.

    Input: df - pandas dataframe with rows being 2D arrays,

    Return data_array - list of pandas time-series
    """
    Nrows = len(df)
    print("Nrows", Nrows)
    data_array = []
    for i in range(0, Nrows):
        # check there's actually data there.
        # use next line since the read in dataframe has returned a string.
        # print('Converting #',i)
        init_series = np.asarray(eval(df.iloc[i]["data"]))
        dat2 = init_series[:, 1].astype(float)
        f = df.iloc[i]["f"]
        periodindex = pd.PeriodIndex(init_series[:, 0], freq=f)
        s = pd.Series(dat2, index=periodindex)
        data_array.append(s.to_timestamp())
    return data_array


def plot_data_frame(df, title, xlabel, ylabel, labels=None, logy=False, save_name=None):
    """plot_data_frame(df,title,xlabel,ylabel,labels=None,logy=False
    save_name=None)

    Plots the time series from a dataframe,
    stored in column 'data2'.

    Input:
    df - dataframe
    title - string for title of plot
    xlabel - string for xlabel for plot
    ylabel - string for ylabel
    labels - optional list of strings
    logy  - optional boolean for enforcing semilogy scale

    Return: None
    Side effect - makes matplotlib plot.
    """

    if labels is None:
        labels = df["name"].values
    for i in range(0, len(df)):
        if logy is True:
            plt.semilogy(df.iloc[i]["data2"], label=labels[i])
        else:
            plt.plot(df.iloc[i]["data2"], label=labels[i])
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if save_name is not None:
        plt.savefig(save_name)
    plt.show()
    return
