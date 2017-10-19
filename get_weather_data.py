# Download data sets from NOAA.
import requests
import json
import pandas as pd
import numpy as np
import ftplib 


#Get API keys from JSON file
with open('keys.json') as key_file:
    keys=json.load(key_file)
#NCDC's call to website.

biggest_cities=[
    ['AL','Birmingham','Mobile','Huntsville'],
    ['AK','Anchorage','Fairbanks','Juneau'],
    ['AZ','Phoenix','Tucson','Mesa'],
    ['AR','Little Rock','Fort Smith','Fayetteville'],
    ['CA','Los Angeles','San Diego','San Jose'],
    ['CO','Denver','Colorado Springs','Aurora'],
    ['CT','Bridgeport','New Haven','Hartford'],
    ['DE','Wilmington','Dover','Newark'],
    ['FL','Jacksonville','Miami','Tampa'],
    ['GA','Atlanta','Augusta','Columbus'],
    ['HI','Honolulu','Hilo','Kailua'],
    ['ID','Boise','Nampa','Idaho Falls'],
    ['IL','Chicago','Aurora','Rockford'],
    ['IN','Indianapolis','Fort Wayne','Evansville'],
    ['IA','Des Moines','Cedar Rapids','Davenport'],
    ['KS','Wichita','Overland Park','Kansas City'],
    ['KY','Louisville','Lexington','Owensboro'],
    ['LA','New Orleans','Shreveport','Baton Rouge'],
    ['ME','Portland','Lewiston','Bangor'],
    ['MD','Baltimore','Frederick','Gaithersburg'],
    ['MA','Boston','Worcester','Springfield'],
    ['MI','Detroit','Grand Rapids','Warren'],
    ['MN','Minneapolis','St. Paul','Rochester'],
    ['MS','Jackson','Gulfport','Biloxi'],
    ['MO','Kansas City','St. Louis','Springfield'],
    ['MT','Billings','Missoula','Great Falls'],
    ['NE','Omaha','Lincoln','Bellevue'],
    ['NV','Las Vegas','Reno','Henderson'],
    ['NH','Manchester','Nashua','Concord'],
    ['NJ','Newark','Jersey City','Paterson'],
    ['NM','Albuquerque','Las Cruces','Rio Rancho'],
    ['NY','New York','Buffalo','Rochester'],
    ['NC','Charlotte','Raleigh','Greensboro'],
    ['ND','Fargo','Bismarck','Grand Forks'],
    ['OH','Columbus','Cleveland','Cincinnati'],
    ['OK','Oklahoma City','Tulsa','Norman'],
    ['OR','Portland','Salem','Eugene'],
    ['PA','Philadelphia','Pittsburgh','Allentown'],
    ['RI','Providence','Warwick','Cranston'],
    ['SC','Charleston','Columbia','North Charleston'],
    ['SD','Sioux Falls','Rapid City','Aberdeen'],
    ['TN','Memphis','Nashville','Knoxville'],
    ['TX','Houston','San Antonio','Dallas'],
    ['UT','Salt Lake City','West Valley City','Provo'],
    ['VT','Burlington','South Burlington','Rutland'],
    ['VA','Virginia Beach','Norfolk','Chesapeake'],
    ['WA','Seattle','Spokane','Tacoma'],
    ['WV','Charleston','Huntington','Parkersburg'],
    ['WI','Milwaukee','Madison','Green Bay'],
    ['WY','Cheyenne','Casper','Laramie']];

# #read in DataFrame of largest 3 cities in each US state. (list from Wikipedia)
# def make_bigcity_list(biggest_city,depth=3):
#     names=['State','L1','L2','L3']
#     nrows = len(biggest_cities);
#     biggest_city_list=list()
#     #now make a list of "city,state" pairs
#     for i in range(0,nrows):
#         for j in range(1,depth+1):
#             city_str=biggest_cities[i][j]+', '+biggest_cities[i][0]
#             biggest_city_list.append(city_str)

#     return biggest_city_list    
# city_list=make_bigcity_list(biggest_cities,depth=3)

#read in list of airports
airport_df = pd.read_csv('data/airports.dat',skiprows=1,na_values='\\N')
print(len(airport_df))
#only keep US airports, and name,city, and ICAO codes
msk2=airport_df['Country']=='United States'
airport_df=airport_df[msk2][['Name','City','ICAO']]
print(len(airport_df))

# #now make a dict of city names, and station locations.
# #Find allowed ID number corresponding to largest cities.
# #Note not all of these have entries.  
def get_airport_code(dataframe,city_list,depth=3):
    aircode_df=pd.DataFrame()
    nrows=len(city_list)
    for i in range(0,nrows):
        for j in range(1,depth+1):
            city=city_list[i][j]
            state=city_list[i][0]
            msk=dataframe['City'].str.contains(city)
            #If more than one entry, just pick the first one.
            if (sum(msk)>=1):
                df_small=dataframe[msk].head(n=1)
                df_small['State']=state
                aircode_df=aircode_df.append(df_small)
            else:
                print('could not find city:'+city)
    aircode_df=aircode_df.rename(columns={'ICAO':'CALL'})
    return aircode_df

airport_codes=get_airport_code(airport_df,biggest_cities)

#now compare with stations from ISD database.
isd_name_df=pd.read_fwf('data/ISD/isd-history.txt',skiprows=20)
isd_name_df=isd_name_df[['USAF','WBAN','CALL']]

#merge together.
airport_codes2 = pd.merge(airport_codes,isd_name_df,on='CALL')

#drop the entries with 999999.
msk1 = airport_codes2['USAF']!=999999
msk2 = airport_codes2['WBAN']!=99999
airport_codes2=airport_codes2[msk1&msk2]

#now download the data from NOAA:

# def noaa_ftp():
#     ftp=ftplib.FTP('ftp.ncdc.noaa.gov')
#     ftp.cwd('pub/data/noaa/isd-lite')
#     return ftp


# def download_city_data(ftp,USAF,WBAN,yearstr):
#     ftp.cwd(str(yearstr))

#     cmd_name='RETR '+USAF+'-'+WBAN+'-'+yearstr+'.gz'
#     ftp.retrbinary(cmd_name)

def wget_data(USAF,WBAN,yearstr):
    base_url='ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-lite'
    url=base_url+'/'+yearstr+'/'+USAF+'-'+WBAN+'-'yearstr'.gz'
    wget.download(url,out='data/ISD')
# ftp=noaa_ftp()
# cd data/isd_data

    
# #Find all cities with Local Climatological Data.
# #This dataset has hourly temperature/preciptation measurements.
# def get_lcd_data_locations():
#     count = 0
#     Nretrieve=1000
#     keep_going=True
#     lcd_frame=pd.DataFrame()
#     #loop through all available results.
#     #took around 10 sec.
#     while (keep_going==True):
#         print('count=',count)
#         offset=count*Nretrieve
#         url1=baseurl+'locations?datasetid=LCD&limit=1000&offset={}'.format(offset,)
#         response1=requests.get(url1,headers=h1)
#         lcd_list=eval(response1.text)
#         df=pd.DataFrame.from_dict(lcd_list["results"])
#         df_length=df.shape[0]
#         lcd_frame=lcd_frame.append(df)
#         count+=1
#         if (df_length != Nretrieve):
#             print('Hit the end:',df_length,Nretrieve)
#             keep_going=False
#             lcd_frame.index=np.arange(0,lcd_frame.shape[0])
#     return lcd_frame
# #dump that dataframe to file.
# #lcd_frame=get_lcd_data_locations()
# #lcd_frame.to_json('data/lcd_city_list.json',orient='records',lines=True)
# #fgfd
# ## Read in from previously read in 
# #lcd_frame=pd.read_json('data/lcd_city_list.json',orient='records',lines=True)

# #now make a dict of city names, and station locations.
# #Find allowed ID number corresponding to largest cities.
# #Note not all of these have entries.  
def get_ids(dataframe,loc_list):
    loc_dict=dict()
    for city in loc_list:
        msk=dataframe['name'].str.contains(city)
        #uses fact that in returned list that the first entries in this list
        # are all of the big entries for cities with multiple zip codes.
        if (sum(msk)>0):
            id=dataframe['id'][msk]
        else:
            print('couldn\'t find city:'+city)
            id=None
        loc_dict[city]=id
    return loc_dict


# #works?
# url0='https://www.ncdc.noaa.gov/cdo-web/api/v2/'\
#     +'data?datasetid=GHCND&locationid=ZIP:97218&startdate=2016-05-01&enddate=2016-05-01'

# def make_url(dataset=None,loc_id=None,start=None,end=None):
#     url='data?datasetid={}&locationid={}&startdate={}&enddate={}'.format(dataset,loc_id,start,end)
#     return url


# #taken from location page.
# def call_website(url_ext):
#     baseurl='https://www.ncdc.noaa.gov/cdo-web/api/v2/'
#     url=baseurl+url_ext
#     print('new url:\n',url)
#     response2=requests.get(url,headers=h1)
#     # #gets a json response
#     try:
#         data=eval(response2.text)
#     except:
#         print('cant read first response')
#         #try old one:
#     # print('old url:\n',url0)
#     # response2=requests.get(url0,headers=h1)
#     # # #gets a json response
#     # try:
#     #     data=eval(response2.text)
#     # except:
#     #     print('cant read second response')
#     return data

# #download data set for specified date range.
# #put dataset into SQL table.

