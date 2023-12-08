# replicating airport_play.ipynb commands for getting data together
# also need to handle bulk download of ELEC data.

import os
import sys
from time import time

import pandas as pd

from us_elec.util.get_weather_data import (
    make_airport_df,
    read_isd_df,
    merge_air_isd,
    get_all_data_http,
)

# Download data across all types and years
from us_elec.util.get_forecast_data import NDFDForecast

import logging


START_YEAR = 2015
END_YEAR = 2024


def download_isd_data():
    air_df = make_airport_df()
    isd_df = read_isd_df()
    merge_df = merge_air_isd(air_df, isd_df)

    merge_df.to_csv("/tf/data/air_merge_df.csv.gz")
    get_all_data_http(merge_df, START_YEAR, END_YEAR)


def download_ndfd_data():
    """
    Download all NDFD data for temperature, wind-speed, direction and cover from S3.
    """
    logger = logging.getLogger()

    t0 = time()

    for kv in ["temp", "wspd", "wdir", "sky"]:
        logger.info(f"\tStarting {kv}")
        NDFDForecast.get_all_ndfd_files(kv)
        t1 = time()

        logger.info(f"\tDone with {kv} in {(t1-t0)/60:.1f} min")
        t0 = t1


if __name__ == "__main__":
    # parse params from command line with defaults?
    download_isd_data()
    # download_ndfd_data()
