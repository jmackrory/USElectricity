import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util.sql_lib import get_dataframe

def convert_data(df):
    """convert_data(df)
    Converts data frame row-by-row
    from an array to a time-series.

    Input: df - pandas dataframe with rows being 2D arrays,

    Return data_array - list of pandas time-series
    """
    Nrows=len(df)
    print('Nrows',Nrows)
    data_array=[];
    for i in range(0,Nrows):
        #check there's actually data there.
        #use next line since the read in dataframe has returned a string.
        #print('Converting #',i)
        init_series=np.asarray(eval(df.iloc[i]['data']))
        dat2=init_series[:,1].astype(float);
        f = df.iloc[i]['f']
        periodindex=pd.PeriodIndex(init_series[:,0],freq=f)
        s=pd.Series(dat2,index=periodindex)
        data_array.append(s.to_timestamp())
    return data_array

def plot_data_frame(df,title,xlabel,ylabel,labels=None,logy=False):
    """plot_data_frame(df,title,xlabel,ylabel,labels=None,logy=False)
   
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
        labels=df['name'].values
    for i in range(0,len(df)):
        if (logy==True):
           plt.semilogy(df.iloc[i]['data2'],label=labels[i])
        else:
           plt.plot(df.iloc[i]['data2'],label=labels[i])           
    plt.legend(loc='upper left',bbox_to_anchor=(1,1))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)        
    plt.title(title)
    plt.show()
    return

def plot_generation_by_state(cur,state):
    """plot_generation_by_state
    Takes a states name, and looks up net generation amounts across all 
    sectors on a monthly frequency from SQL.  
    Then plots the results.

    Inputs: cur - psycopg2 cursor
    state - string with full state name, e.g. Oregon.

    """
    out_col=('name','data','start','end','f')
    match_names=['Net generation',': '+state+' :',': all sectors :'];    
    state_gen=get_dataframe(cur,'ELEC',out_col,match_names,freq='M');
    data0=convert_data(state_gen)
    state_gen['data2']=data0

    state_gen_max=state_gen['data2'].apply(max)
    plt_ind=state_gen_max.sort_values(ascending=False).index
    #extract out source part of labels via regex, select first match, and convert to array
    gen_labels=state_gen.iloc[plt_ind]['name'].str.extractall('Net generation : ([\s\w\(\)-]+):')[0].values

    plot_data_frame(state_gen.iloc[plt_ind],
    xlabel='Date',
    ylabel='Net Generation (GWh)',title='Generation Across '+state+' by source',
    labels=gen_labels,logy=False)

def plot_retail_price(cur,region):
    us_price=pd.DataFrame()
    out_col=('name','data','start','end','f')    
    match_names=['retail price',': '+region+' :'];    
    us_price=get_dataframe(cur,'ELEC',out_col,match_names,freq='M');
    data0=convert_data(us_price)
    us_price['data2']=data0
    labels=us_price['name'].str.split(':').apply(lambda x:x[2])    
    plot_data_frame(us_price,
                    xlabel='Date',
                    ylabel='Average cost (c/kWh)',
                    title='Average Retail Price of Electricity Across '+region,
                    labels=labels,
                    logy=False)    

def plot_customers(cur, region):
    """plot_customers(cur,region)
    
    Plot number of customer accounts from ELEC table in a region
    Inputs: cur - psycopg2 cursor
    region - string for State/Region name
    """
    us_price=pd.DataFrame()
    out_col=('name','data','start','end','f')        
    match_names=['customer accounts',': '+region+' :'];    
    df=get_dataframe(cur,'ELEC',out_col,match_names,freq='M');
    data0=convert_data(df)
    df['data2']=data0
    labels=df['name'].str.split(':').apply(lambda x:x[2])        
    plot_data_frame(df,
                    xlabel='Date',
                    ylabel='Number',
                    title='Number of Customers Across '+region,
                    labels=labels,
                    logy=False)

def get_state_data(cur,year):
    """get_state_data
    Gets all generation data by state for a given year.
    Useful for plotting usage on map.

    """
    #get a list of series IDs
    state_names = (
    'New Jersey',    'Rhode Island',    'Massachusetts',    'Connecticut',
    'Maryland',    'New York',    'Delaware',    'Florida',
    'Ohio',    'Pennsylvania',    'Illinois',    'California',
    'Hawaii',    'Virginia',    'Michigan',    'Indiana',
    'North Carolina',    'Georgia',    'Tennessee',    'New Hampshire',
    'South Carolina',    'Louisiana',    'Kentucky',    'Wisconsin',
    'Washington',    'Alabama',    'Missouri',    'Texas',
    'West Virginia',    'Vermont',    'Minnesota',    'Mississippi',
    'Iowa',    'Arkansas',    'Oklahoma',    'Arizona',
    'Colorado',    'Maine',    'Oregon',    'Kansas',
    'Utah',    'Nebraska',    'Nevada',    'Idaho',
    'New Mexico',    'South Dakota',    'North Dakota',    'Montana',
    'Wyoming',    'Alaska')

    state_abbr=(
    'NJ',  'RI',    'MA',    'CT',
    'MD',    'NY',    'DE',    'FL',
    'OH',    'PA',    'IL',    'CA',
    'HI',    'VA',    'MI',    'IN',
    'NC',    'GA',    'TN',    'NH',
    'SC',    'LA',    'KY',    'WI',
    'WA',    'AL',    'MO',    'TX',
    'WV',    'VT',    'MN',    'MS',
    'IA',    'AR',    'OK',    'AZ',
    'CO',    'ME',    'OR',    'KS',
    'UT',    'NE',    'NV',    'ID',
    'NM',    'SD',    'ND',    'MT',
    'WY',    'AK')

    gen_types=['HYC','TSN','WND','COW','NUC','NG','ALL']

    df_results=pd.DataFrame(columns=gen_types,index=state_names)

    for i,state in enumerate(state_abbr):
        for j,gen in enumerate(gen_types):
            sql_str="""
            SELECT series_id,obs_date,obs_val FROM "elec_gen"
            WHERE series_id LIKE 'ELEC.GEN.{ser}-{state}-99.A' AND obs_date= DATE '{year}-12-31'""".format(ser=gen,state=state,year=year)
            cur.execute(sql_str)
            r=cur.fetchone()
            if r is not None:
                df_results.iloc[i,j]=r[2]
    return df_results
    
