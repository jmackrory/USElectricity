import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read in SGDE data files.
# Re-arrange as individual data series.
# Eventually output to SQL for easier access later.


# make time column to datetimes.

# #find unique names.
# for name in df['ID'].unique():
#     msk= df['ID']==name
#     d0 =
#     times=df[msk,'LocalDateTime']
#     k

# for each name, make a time series.

# Found that direct attempts to mask were taking way, way too long.
# Given structure in the data easier to find indices where the labels changed.


# find indices where names change (using fact series is grouped, in order)
def find_change(namevec, start, step=1000):
    end = len(namevec)
    pos = start
    old_name = namevec[start]
    while namevec[pos] == old_name:
        pos_old = pos
        pos = pos + step
        if pos >= end:
            print("hit end of vector, returning end")
            return end - 1

    start_bkt = pos_old
    end_bkt = pos
    # now do bisection to search for change of index.
    start_name = namevec[start_bkt]
    end_name = namevec[end_bkt]
    while (end_bkt - start_bkt) > 1:
        mid = int((start_bkt + end_bkt) / 2)
        mid_name = namevec[mid]
        if start_name == mid_name:
            start_bkt = mid
            start_name = mid_name
        elif end_name == mid_name:
            end_bkt = mid
            end_name = mid_name
    # change_indx = end
    return end_bkt


# loop over entire list, finding all integer indices of all changes
def find_all_changes(namevec):
    change_vec = list()
    tot = len(namevec) - 1
    pos = 0
    change_vec.append(pos)
    while pos < tot:
        pos = find_change(namevec, pos)
        change_vec.append(pos)
        print(pos)
    change_vec = np.array(change_vec)
    return change_vec


# should use multiindex for datetimes.


# Now make existing dataframe into a list of time series.
def make_elec_series(df):
    indx_vec = find_all_changes(df["ID"])
    num_series = len(indx_vec)
    name_vec = df.loc[indx_vec, "ID"]
    name_vec.index = range(num_series)
    ts_tot = list()
    # df_new=pd.DataFrame(columns=['ID','start','end','data'])
    tf = "%d%b%y:%H:%M:%S"
    for i in range(num_series - 1):
        # name = name_vec[i]
        span = slice(indx_vec[i], indx_vec[i + 1] - 1)
        times = pd.to_datetime(df.loc[span, "LocalDateTime"], format=tf)
        # start_time=pd.to_datetime(df.loc[indx_vec[i],'LocalDateTime'],format=tf)
        # end_time=pd.to_datetime(df.loc[indx_vec[i+1]-1,'LocalDateTime'],format=tf)
        # # timeindex=pd.DatetimeIndex(start=start_time,end=end_time,freq='15 min')
        span = slice(indx_vec[i], indx_vec[i + 1] - 1)
        elec = df.loc[span, "kWh"].values
        ts = pd.Series(data=elec, index=times)
        ts_tot.append(ts)
    return ts_tot


# Now make existing dataframe into a list of time series.
def make_elec_dataframe(df):
    indx_vec = find_all_changes(df["ID"])
    num_series = len(indx_vec)
    name_vec = df.loc[indx_vec, "ID"]
    name_vec.index = range(num_series)
    # ts_tot = list()
    # make initial dataframe
    df_new = pd.DataFrame(
        index=pd.DatetimeIndex(start="2011", end="2015", freq="15 min")
    )
    tf = "%d%b%y:%H:%M:%S"
    for i in range(num_series - 1):
        # name = name_vec[i]
        span = slice(indx_vec[i], indx_vec[i + 1] - 1)
        times = pd.to_datetime(df.loc[span, "LocalDateTime"], format=tf)
        # start_time=pd.to_datetime(df.loc[indx_vec[i],'LocalDateTime'],format=tf)
        # end_time=pd.to_datetime(df.loc[indx_vec[i+1]-1,'LocalDateTime'],format=tf)
        # times=pd.DatetimeIndex(start=start_time,end=end_time,freq='15 min')
        elec = df.loc[span, "kWh"].values
        df_loc = pd.DataFrame(elec, index=times, columns=[name_vec[i]])
        print(df_loc.head())
        df_new = df_new.join(df_loc, how="outer")
    return df_new


if __name__ == "__main__":
    nr = int(1e8)
    df = pd.read_csv(
        "SDGE_StartThru2016Q3.zip", nrows=nr, usecols=["ID", "LocalDateTime", "kWh"]
    )

    df1 = make_elec_dataframe(df)


# df1= make_elec_series(df)
# Plot a list of timeseries
# def plot_data(date,df):
#     for s in df:
#         try:plt.plot(s[date])
#         except:print(date+' not found')
#     plt.show()
#     return

# #Testing function to playwith joining pandas objects.
# def make_dummy_frame():
#     time=[pd.DatetimeIndex(['2012','2014','2015']),
#           pd.DatetimeIndex(['2011','2012','2013','2014']),
#           pd.DatetimeIndex(['2011','2012','2015','2016'])]
#     data=[[1,3.4,9],[1.0,0.0,-1.0,0.1],[2,3,5,7]]

#     df=pd.DataFrame()
#     names=['name1','name2','name3']
#     #df=pd.DataFrame(data[0],index=time[0],columns=[names[0]])
#     for i in range(3):
#         df_loc=pd.DataFrame(data[i],index=time[i],columns=[names[i]])
#         df=df.join(df_loc,how='outer')
#     #df.columns=names
#     return df

# dummyframe=make_dummy_frame()
# When getting satellite data restrict from 5am-7pm.

# Randomly sample 10% of days as test set.

# Try to train simple model for total output on a given day.

# Simple questions:
# What is the seasonal variation?
#  * Could measure via average integrated power for a day, plot for whole dataset on yearly basis.
#  * Take weekly chunks to find variance.  For each sensor, take the mean over a particular week,
#   then plot the variance.
# How correlated are the data from different sensors at the same time?

# What commercial questions would I want to answer?
# * How reliable is solar power? How much does solar electricity output fluctuate?
# * What is the potential scope for rooftop solar? (How much capacity? Comparison with demand?)
# * How large is that for household needs?  What of industrial/commerical purposes?
# * How much of total energy usage is electricity?
