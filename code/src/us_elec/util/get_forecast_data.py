import os
import sys
from typing import List
import boto3
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

NDFD_BUCKET = "noaa-ndfd-pds"

NDFD_LOCAL_DIR = "/tf/data/NDFD"

# start 2020/4/16
SUB_TYPES = {"temp": "YEU", "wind_dir": "YBU", "wind_speed": "YCU", "sky_cover": "YAU"}
ACC_LEVEL = "Z88"  # old 5km grid

START_YEAR = 2020
END_YEAR = 2023


def mkdir_p(path):
    sub_dir, _ = os.path.split(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


def download_s3_ndfd_file(s3_dir, file_prefix, sub_type, use_newest=True):
    # import ipdb; ipdb.set_trace()
    s3_client = boto3.client("s3")
    # s3 = boto3.resource('s3')
    files = s3_client.list_objects(Bucket=NDFD_BUCKET, Prefix=s3_dir, Delimiter="/")
    if not files.get("Contents"):
        logger.info(f" No files found for ({s3_dir})")
        return
    file = get_relevant_file(files["Contents"], file_prefix, use_newest)
    if not file:
        logger.info(f"No file found for ({s3_dir, file_prefix, sub_type})")
        return None
    local_path = get_local_file(file, sub_type)
    mkdir_p(local_path)
    s3_client.download_file(Bucket=NDFD_BUCKET, Key=file, Filename=local_path)
    return local_path


def get_relevant_file(files: List, file_prefix: str, use_newest: bool = True):
    """Find most recent file"""
    if not files:
        return None
    files = sorted(files, key=lambda x: x["Key"], reverse=use_newest)
    for file in files:
        fn = file["Key"]
        if file_prefix in fn:
            return fn


def get_local_file(s3_filepath: str, sub_type: str) -> str:
    pieces = os.path.split(s3_filepath)
    return os.path.join(NDFD_LOCAL_DIR, sub_type, pieces[-1])


def get_s3_dir(year, month, day, var="temp"):
    """Get S3 bucket directory path for var"""
    return f"wmo/{var}/{year}/{month:02d}/{day:02d}/"


def get_prefix(var: str) -> str:
    return SUB_TYPES[var] + ACC_LEVEL


def get_file(year, month, day, var):
    file_prefix = get_prefix(var)
    s3_dir = get_s3_dir(year, month, day, var)
    fn = download_s3_ndfd_file(s3_dir, file_prefix, var)
    return fn


def get_all_files(var: str = "temp"):
    files = []
    for year in range(START_YEAR, END_YEAR):
        for month in range(1, 13):
            for day in range(1, 31):
                fn = get_file(year, month, day, var)
                if fn:
                    files.append(fn)
    return fn
