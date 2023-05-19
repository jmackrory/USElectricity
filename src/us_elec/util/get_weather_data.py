# Download data sets from NOAA.
# See weather_dataframe.py for converting this data into a combined dataframe.
from time import sleep
from ftplib import FTP
import os
import requests
import urllib

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

# FTP_CONNECTIONS = {}
FTP_NOAA = "ftp.ncdc.noaa.gov"

# #Get API keys from JSON file
# with open('keys.json') as key_file:
#     keys=json.load(key_file)
# NCDC's call to website.

# try to map states to power producing regions.
# This is not quite correct, since some states are split between
# multiple regions (TN, MS, ND).  But will try as first attempt.
region_dict = {
    "AL": "Southeast",
    "AK": "AK",
    "AZ": "Southwest",
    "AR": "Midwest",
    "CA": "California",
    "CO": "Northwest",
    "CT": "Northeast",
    "DE": "Northeast",
    "FL": "Florida",
    "GA": "Southeast",
    "HI": "HI",
    "ID": "Northwest",
    "IL": "Midwest",
    "IN": "Midwest",
    "IA": "Midwest",
    "KS": "Central",
    "KY": "Mid-Atlantic",
    "LA": "Midwest",
    "ME": "Northeast",
    "MD": "Mid-Atlantic",
    "MA": "Northeast",
    "MI": "Midwest",
    "MN": "Midwest",
    "MS": "Midwest",
    "MO": "Midwest",
    "MT": "Northwest",
    "NE": "Central",
    "NV": "Southwest",
    "NH": "Northeast",
    "NJ": "Mid-Atlantic",
    "NM": "Southwest",
    "NY": "New York",
    "NC": "Carolinas",
    "ND": "Central",
    "OH": "Mid-Atlantic",
    "OK": "Central",
    "OR": "Northwest",
    "PA": "Mid-Atlantic",
    "RI": "Northeast",
    "SC": "Carolinas",
    "SD": "Central",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Northwest",
    "VT": "New England",
    "VA": "Mid-Atlantic",
    "WA": "Northwest",
    "WV": "Mid-Atlantic",
    "WI": "Midwest",
    "WY": "Northwest",
}

DATA_PATH = "/tf/data"
AIR_CSV = os.path.join(DATA_PATH, "airports.csv")
ISD_HISTORY = os.path.join(DATA_PATH, "ISD/isd-history.txt")

START_YR_MONTH = "2015-07"
END_YR_MONTH = "2023-03"
START_YR = 2015
END_YR = 2023


def make_airport_df():
    """make_airport_df
    Read in a list of global airports, and extract their ICAO codes.
    Restrict then to large and medium airports across the US.
    """
    # read in list of airports
    airport_df = pd.read_csv(AIR_CSV)
    # only keep US airports, and name,city, and ICAO codes
    msk2 = (airport_df["iso_country"] == "US") & airport_df["type"].apply(
        lambda x: x in ["large_airport", "medium_airport"]
    )

    airport_df = airport_df[msk2][["name", "municipality", "ident"]].rename(
        columns={"ident": "CALL", "municipality": "City"}
    )
    # airport_codes = get_airport_code(airport_df, biggest_cities)

    return airport_df


# #now make a dict of city names, and station locations.
# #Find allowed ID number corresponding to largest cities.
# #Note not all of these have entries.
def get_airport_code(dataframe, city_list, depth=3):
    """get_airport_code(dataframe,city_list,depth=3)
    Extract the ICAO code/callsign for one airport in each city.
    Return a dataframe with the city,state and callsign.

    dataframe: initial dataframe with list of global airports, locations, ICAO callsigns.
    city_list: list of lists cities to find the callsigns for. Containts cities in states.
    depth: how many cities in each state to look for.

    """
    msk2 = (dataframe["iso_country"] == "US") & dataframe["type"].apply(
        lambda x: x in ["large_airport", "medium_airport"]
    )
    aircode_df = dataframe[msk2]
    aircode_df = aircode_df.rename(
        columns={"ident": "CALL", "municipality": "City", "name": "Name"}
    )
    return aircode_df


def read_isd_df():
    """make_airport_df
    Read in list of weather stations and USAF-WBAN codes for the weather stations.
    Trim to only stations that have operated since 2015.
    """

    # now compare with stations from ISD database.
    isd_name_df = pd.read_fwf(ISD_HISTORY, skiprows=20)
    # also only keep airports still operational in time period.
    msk = isd_name_df["END"] > 20150000
    isd_name_df = isd_name_df[msk]
    isd_name_df = isd_name_df[["USAF", "WBAN", "CALL", "LAT", "LON", "ST"]]
    return isd_name_df


def merge_air_isd(airport_codes, isd_name_df):
    """merge_air_isd_df

    Merge the airport and weather data frames on name.
    Trim out duplicates.

    """
    airport_total = pd.merge(airport_codes, isd_name_df, on="CALL")

    # drop any duplicated entries. (i.e. multiple at same airport)
    msk3 = airport_total["CALL"].duplicated().values
    print("Duplicated values for:")
    print(airport_total[msk3][["CALL", "City"]])
    airport_total = airport_total[~msk3]

    # make these codes integers.
    airport_total["USAF"] = airport_total["USAF"].astype(int)
    airport_total["WBAN"] = airport_total["WBAN"].astype(int)

    return airport_total


def plot_airports(air_df):
    """plot_airports(air_df)
    Plot the locations of the airports contained within air_df.
    Useful for eyeballing if there are systematic flaws in the locations
    that made the cut.
    """
    fig = plt.figure(figsize=(20, 20))
    crs = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=crs)
    ax.set_extent([-130, -65, 25, 50], crs=crs)

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)

    # actually draw the map
    lons = air_df["LON"].values
    lats = air_df["LAT"].values
    ax.plot(lons, lats, "x", transform=crs)
    plt.show()
    return None


# now download the data from NOAA:
def wget_data(USAF, WBAN, yearstr, city, airport):
    """wget_data(USAF,WBAN,yearstr,city,airport)
    Download automated weather station data from NOAA for a given year at a given airport.

    USAF: USAF 6 digit code for airport.
    WBAN: NOAA code for weather station at airport
    yearstr: a string containing the 4 digit year.
    city: city the airport is located in
    airport: Name of the airport.
    """
    base_url = "ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-lite/"
    file_name = isd_filename(yearstr, USAF, WBAN)
    url = base_url + file_name
    try:
        print(url)
        local_filepath = get_local_isd_path(yearstr, USAF, WBAN)
        req = urllib.request.Request(url)
        response = urllib.request.urlopen(req)
        with open(local_filepath, "wb") as f:
            f.write(response.read())
    except urllib.error.URLError as err:
        print("\n could not download data from city:", city, airport)
        print(err.reason)
    return None


def get_noaa_ftp_conn():
    ftp = FTP(FTP_NOAA)
    s = ftp.login()
    print(s)
    return ftp


def close_ftp_conn(ftp):
    ftp.quit()


# now download the data from NOAA:
def ftp_download_data(ftp, USAF, WBAN, yearstr, city, airport):
    """ftp_download_data(USAF,WBAN,yearstr,city,airport)
    Download automated weather station data from NOAA for a given year at a given airport.

    USAF: USAF 6 digit code for airport.
    WBAN: NOAA code for weather station at airport
    yearstr: a string containing the 4 digit year.
    city: city the airport is located in
    airport: Name of the airport.
    """
    # url = base_url + file_name
    # print(url)
    # req = urllib.request.Request(url)
    # response = urllib.request.urlopen(req)
    # with open(local_filepath, 'wb') as f:
    #     f.write(response.read())

    local_filepath = get_local_isd_path(yearstr, USAF, WBAN)
    with open(local_filepath, "wb") as fp:
        file_name = isd_filename(yearstr, USAF, WBAN)
        print(ftp.pwd())
        try:
            ftp.retrbinary(f"RETR {file_name}", fp.write)
        except Exception as e:
            print(f"Couldn't find info for {city} {airport} {file_name}")
            print(e)
    return None


def isd_filename(yearstr, USAF, WBAN):
    """isd_filename(yearstr, USAF, WBAN)
    Make filename corresponding to zipped file names used in ISD database.
    """
    # put in some padding {:0>5} for shorter codes.
    fn = "{1}-{2:0>5}-{0}.gz".format(yearstr, str(USAF), str(WBAN))
    return fn


def get_local_isd_path(yearstr, usaf, wban):
    return os.path.join(DATA_PATH, "ISD", isd_filename(yearstr, usaf, wban))


# download weather data for all of the airports specified in aircode
def get_all_data_ftp(aircode, start_year=2015, end_year=2020):
    """get_all_data(aircode, years=['2015', '2016', '2017'])
    Download the data for all airports we could find weather stations for in desired cities.

    aircode: datafram containing airport codes, NOAA station numbers, airports
    years: array of strings for the years to seek data.
    """
    # Nc = len(aircode)
    ftp = get_noaa_ftp_conn()
    for year in range(start_year, end_year):
        yearstr = str(year)
        ftp.cwd(f"/pub/data/noaa/isd-lite/{yearstr}")
        for i in tqdm(range(len(aircode))):
            ap = aircode.iloc[i]
            usaf = ap["USAF"]
            wban = ap["WBAN"]
            city = ap["City"]
            airport = ap["name"]
            ftp_download_data(ftp, usaf, wban, str(year), city, airport)
            sleep(0.01)
    close_ftp_conn(ftp)
    return None


def get_missing_data_ftp(aircode, start_year=2015, end_year=2020):
    """get_all_data(aircode, years=['2015', '2016', '2017'])
    Download the data for all airports we could find weather stations for in desired cities.

    aircode: datafram containing airport codes, NOAA station numbers, airports
    years: array of strings for the years to seek data.
    """
    # Nc = len(aircode)
    found_count = 0
    ftp = get_noaa_ftp_conn()
    for year in range(start_year, end_year):
        yearstr = str(year)
        ftp.cwd(f"/pub/data/noaa/isd-lite/{yearstr}")
        for i in tqdm(range(len(aircode))):
            ap = aircode.iloc[i]
            usaf = ap["USAF"]
            wban = ap["WBAN"]
            city = ap["City"]
            airport = ap["name"]
            fn = get_local_isd_path(year, usaf, wban)
            if not os.path.exists(fn):
                print(f"{fn} missing for {airport}")
                ftp_download_data(ftp, usaf, wban, str(year), city, airport)
            else:
                found_count += 1
    close_ftp_conn(ftp)
    print(found_count)
    return None


def get_http_isd_url(yearstr, USAF, WBAN):
    fn = isd_filename(yearstr, USAF, WBAN)
    url = f"https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{yearstr}/{fn}"
    return url


# now download the data from NOAA:
def http_download_data(USAF, WBAN, yearstr, city, airport):
    """http_download_data(USAF,WBAN,yearstr,city,airport)
    Download automated weather station data from NOAA for a given year at a given airport.

    USAF: USAF 6 digit code for airport.
    WBAN: NOAA code for weather station at airport
    yearstr: a string containing the 4 digit year.
    city: city the airport is located in
    airport: Name of the airport.
    """

    local_filepath = get_local_isd_path(yearstr, USAF, WBAN)
    url = get_http_isd_url(yearstr, USAF, WBAN)
    # with open(local_filepath, 'wb') as fp:
    #     download_chunked_file(url, fp)
    req = urllib.request.Request(url)

    try:
        # ftp.retrbinary(f'RETR {file_name}', fp.write)
        response = urllib.request.urlopen(req)
        with open(local_filepath, "wb") as f:
            f.write(response.read())

    except Exception as e:
        print(f"Couldn't find info for {city} {airport} {local_filepath}")
        print(e)

    return local_filepath


# def download_chunked_file(url:str, fp):
#     """Taken from answer in:
#     https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests
#     For large files
#     """
#     # NOTE the stream=True parameter below

#     with requests.get(url, stream=True) as r:
#         r.raise_for_status()
#         for chunk in r.iter_content(chunk_size=8192):
#             # If you have chunk encoded response uncomment if
#             # and set chunk_size parameter to None.
#             #if chunk:
#             fp.write(chunk)
#     return None


# download weather data for all of the airports specified in aircode
def get_all_data_http(aircode, start_year=2015, end_year=2020):
    """get_all_data_http (aircode, years=['2015', '2016', '2017'])
    Download the data for all airports we could find weather stations for in desired cities.

    aircode: datafram containing airport codes, NOAA station numbers, airports
    years: array of strings for the years to seek data.
    """
    # Nc = len(aircode)
    for year in range(start_year, end_year):
        # yearstr = str(year)
        for i in tqdm(range(len(aircode))):
            ap = aircode.iloc[i]
            usaf = ap["USAF"]
            wban = ap["WBAN"]
            city = ap["City"]
            airport = ap["name"]
            http_download_data(usaf, wban, str(year), city, airport)
            sleep(0.01)
    return None


# now read it in, convert to time-series.


def load_isd_df(filename, tzstr=None):
    """
    load_isd_df(filename)

    Read in a automated weather stations data from file.
    Data is space separated columns, with format given in
    "isd-lite-format.txt".
    Converts to pandas dataframe using date/time columns as DateTimeIndex.
    Note: times are all in UTC format
    Format info:
    1: Year
    2: Month
    3: Day
    4: Hour
    5: Temperature (x10) in celcius
    6: Dew point temperature (x10) in celcius
    7: Sea level pressure (x10 in hectopascals)
    8: Wind direction (degrees from north)
    9: Wind speed (x10 in meters per second)
    10: Cloud Coverage (categorical)
    11: Precipitation for One Hour (x10, in mm)
    12: Precipitation total for Six hours (x10 in mm)

    All missing values are -9999.
    """
    # use fixed width format to read in (isd-lite-format has data format)
    col_names = [
        "year",
        "month",
        "day",
        "hour",
        "Temp",
        "DewTemp",
        "Pressure",
        "WindDir",
        "WindSpeed",
        "CloudCover",
        "Precip-1hr",
        "Precip-6hr",
    ]
    df = pd.read_fwf(
        filename, compression="gzip", na_values=["-9999", "999"], names=col_names
    )

    # make a time index.
    times = pd.to_datetime(
        {"year": df["year"], "month": df["month"], "day": df["day"], "hour": df["hour"]}
    )

    Tindex = pd.DatetimeIndex(times)
    df.index = (
        Tindex  # .tz_localize(tzstr)#, ambiguous='infer', nonexistent='shift_forward')
    )
    return df


def convert_isd_to_df(filename, city, state):
    """
    convert_to_df(filename)

    Read in a automated weather stations data from file.
    Data is space separated columns, with format given in
    "isd-lite-format.txt".
    Converts to pandas dataframe using date/time columns as DateTimeIndex.
    Format info:
    1: Year
    2: Month
    3: Day
    4: Hour
    5: Temperature (x10) in celcius
    6: Dew point temperature (x10) in celcius
    7: Sea level pressure (x10 in hectopascals)
    8: Wind direction (degrees from north)
    9: Wind speed (x10 in meters per second)
    10: Cloud Coverage (categorical)
    11: Precipitation for One Hour (x10, in mm)
    12: Precipitation total for Six hours (x10 in mm)

    All missing values are -9999.
    """
    df = load_isd_df(filename)
    # df.index = pd.MultiIndex.from_product([Tindex,[city_st]])
    # delete those columns
    df = df.drop(labels=["year", "month", "day", "hour"], axis=1)
    df["city"] = city
    df["state"] = state
    city_st = city + ", " + state
    df["city, state"] = city_st
    df["region"] = region_dict[state]

    return df


def convert_state_isd(air_df, ST):
    """convert_all_isd(air_df)
    convert the weather files for a particular state into
    one big data frame.
    """
    # data_dir = "data/ISD/"
    Tindex = pd.date_range(start=START_YR_MONTH, end=END_YR_MONTH, freq="h")
    df_tot = pd.DataFrame(index=Tindex)
    # select out only the entries for the desired state.
    msk = air_df["ST"] == ST
    air_msk = air_df[msk]
    for i in range(len(air_msk)):
        for year in range(START_YR, END_YR):
            yearstr = str(year)
            ap = air_msk.iloc[i]
            usaf = ap["USAF"]
            wban = ap["WBAN"]
            city = ap["City"]
            state = ap["ST"]
            file_name = get_local_isd_path(yearstr, usaf, wban)
            df = convert_isd_to_df(file_name, city, state)
            df_tot = pd.concat([df_tot, df])
        print("done with {}".format(ap["name"]))
    return df_tot


def convert_all_isd(air_df):
    """convert_all_isd(air_df)
    convert all the weather files for all stations and all years into
    one big data frame.
    """
    # data_dir = "data/ISD/"
    Tindex = pd.DatetimeIndex(start=START_YR_MONTH, end=END_YR_MONTH, freq="h")
    df_tot = pd.DataFrame(index=Tindex)
    nmax = len(air_df)
    for i in range(nmax):
        for yearstr in range(START_YR, END_YR):
            # yearstr = year
            ap = air_df.iloc[i]
            usaf = ap["USAF"]
            wban = ap["WBAN"]
            city = ap["City"]
            state = ap["State"]
            file_name = get_local_isd_path(yearstr, str(usaf), str(wban))
            df = convert_isd_to_df(file_name, city, state)
            df_tot = pd.concat([df_tot, df])
        print("done with {}".format(ap["Name"]))
    return df_tot


# make dataframes with codes.
if __name__ == "__main__":
    try:
        air_df = pd.read_csv("/tf/data/air_code_df.gz")
    except Exception as e:
        print("Didnt find prexisting air_code_df.  Computing directly.")
        print(e)
        airport_codes = make_airport_df()
        isd_names = read_isd_df()
        air_df = merge_air_isd(airport_codes, isd_names)
        # write output to csv
        air_df.to_csv("/tf/data/air_code_df.gz", compression="gzip", header=True)
