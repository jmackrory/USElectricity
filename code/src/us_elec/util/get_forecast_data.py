import logging
import os
import sys
from datetime import datetime
from time import time
from typing import List, Optional

import boto3
from us_elec.SQL.constants import DATA_DIR

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

NDFD_BUCKET = "noaa-ndfd-pds"

NDFD_LOCAL_DIR = os.path.join(DATA_DIR, "NDFD")

# start 2020/4/16
SUB_TYPES = {"temp": "YEU", "wdir": "YBU", "wspd": "YCU", "sky": "YAU"}
ACC_LEVEL = "Z98"  # new 2.5km, hourly data

START_YEAR = 2020
END_YEAR = 2024

FUTURE = False
min_s3_date = datetime(2020, 4, 16)


def mkdir_p(path):
    sub_dir, _ = os.path.split(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


class NDFDForecast:
    @classmethod
    def download_s3_ndfd_file(
        cls,
        s3_dir: str,
        file_prefix: str,
        sub_type: str,
        use_newest: bool = False,
        force: bool = False,
    ) -> Optional[str]:
        """
        Get list of files from NDFD S3 bucket and select one, then download locally.
        By default only download if file does not already exist locally.
        """
        s3_client = boto3.client("s3")
        files = s3_client.list_objects(Bucket=NDFD_BUCKET, Prefix=s3_dir, Delimiter="/")
        if not files.get("Contents"):
            logger.info(f" No files found for ({s3_dir})")
            return None
        file = cls.get_relevant_file(files["Contents"], file_prefix, use_newest)
        if not file:
            logger.info(f"No file found for ({s3_dir, file_prefix, sub_type})")
            return None
        local_path = cls.get_local_file(file, sub_type)
        mkdir_p(local_path)
        if os.path.exists(local_path) and not force:
            return local_path
        s3_client.download_file(Bucket=NDFD_BUCKET, Key=file, Filename=local_path)
        return local_path

    @classmethod
    def get_relevant_file(cls, files: List, file_prefix: str, use_newest: bool = False):
        """Find most relevant file"""
        if not files:
            return None
        files = sorted(files, key=lambda x: x["Key"], reverse=use_newest)
        for file in files:
            fn = file["Key"]
            if file_prefix in fn:
                return fn

    @classmethod
    def get_local_file(cls, s3_filepath: str, sub_type: str) -> str:
        pieces = os.path.split(s3_filepath)
        return os.path.join(NDFD_LOCAL_DIR, sub_type, pieces[-1])

    @classmethod
    def get_s3_dir(cls, year, month, day, var="temp"):
        """Get S3 bucket directory path for var"""
        return f"wmo/{var}/{year}/{month:02d}/{day:02d}/"

    @classmethod
    def get_prefix(cls, var: str) -> str:
        return SUB_TYPES[var] + ACC_LEVEL

    @classmethod
    def get_file(cls, year, month, day, var, force=False):
        file_prefix = cls.get_prefix(var)
        s3_dir = cls.get_s3_dir(year, month, day, var)
        fn = cls.download_s3_ndfd_file(s3_dir, file_prefix, var, force=force)
        return fn

    @classmethod
    def check_date(cls, now: datetime, dt: datetime) -> Optional[bool]:
        if dt < min_s3_date:
            return None
        if dt > now:
            return FUTURE
        return True

    @classmethod
    def get_all_ndfd_files(cls, var: str = "temp", force: bool = False) -> List[str]:
        """Download all desired files from S3 to local.
        Only allow downloads within range and stop for future downloads.
        """
        files: List[str] = []
        t0 = time()
        start_date = datetime.now()
        approx_tot = int((END_YEAR - START_YEAR - 0.3) * 12 * 30)
        for year in range(START_YEAR, END_YEAR):
            for month in range(1, 13):
                t1 = time()
                ts = f"{(t1-t0)/60:.2f}"
                logger.info(f"\tDownloading {year}/{month} for {var} ")
                logger.info(f"{len(files)} of {approx_tot} after {ts}min")
                for day in range(1, 31):
                    try:
                        dt = datetime(year, month, day)
                    except:
                        continue
                    cs = cls.check_date(start_date, dt)
                    if cs is None:
                        continue
                    elif cs == FUTURE:
                        break
                    fn = cls.get_file(year, month, day, var, force=force)
                    if fn:
                        files.append(fn)
        return files
