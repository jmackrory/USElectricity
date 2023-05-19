import csv
import json
import os
import random
import re
from typing import Dict, List, Optional, Tuple, Union
import subprocess
import jsonlines

import psycopg2
from tzfpy import get_tz
import pandas as pd
from tqdm import tqdm

from functools import lru_cache

# from us_elec.SQL.sql_query import eba_table_template, isd_table_template, insert_eba_data
from us_elec.util.get_weather_data import get_local_isd_path, load_isd_df


class TableType:
    EBA = "eba"
    NDFD = "ndfd"
    ISD = "isd"


class SQLVar:
    int = "integer"
    float = "float"
    str = "string"
    timestamptz = "timestamp with time zone"


class EBAName:
    EBA = "EBA"
    DEMAND = "demand"
    DEMAND_FORECAST = "demand_forecast"
    NET_GENERATION = "net_generation"
    INTERCHANGE = "EBA_interchange"


class ColName:
    TS = "ts"
    SOURCE = "source"
    CALL = "callsign"
    MEASURE = "measure"
    DEST = "dest"
    VAL = "val"


class EBAAbbr:
    NG = "Net Generation"
    ID = "Net Interchange"
    DF = "Demand Forecast"
    D = "Demand"
    TI = "Total Interchange"


class EBAGenAbbr:
    COL = "Coal"
    WND = "Wind"
    WAT = "Hydro"
    SOL = "Solar"
    NUC = "Nuclear"
    NG = "Natural Gas"
    OTH = "Other"


class ISDName:
    ISD = "ISD"
    TEMPERATURE = "temp"
    WIND_DIR = "wind_dir"
    WIND_SPEED = "wind_spd"
    PRECIP_1HR = "precip_1hr"
    META = "air_meta"


class ISDDFName:
    TIME = "Time"
    TEMPERATURE = "Temp"
    DEW_TEMPERATURE = "DewTemp"
    PRESSURE = "Pressure"
    WIND_DIR = "WindDir"
    WIND_SPEED = "WindSpeed"
    CLOUD_COVER = "CloudCover"
    PRECIP_1HR = "Precip-1hr"
    PRECIP_6HR = "Precip-6hr"


ISD_COLS = [
    "Year",
    "Month",
    "Day",
    "Hour",
    "Temp",
    "DewTemp",
    "Pressure",
    "WindDir",
    "WindSpeed",
    "CloudCover",
    "Precip-1hr",
    "Precip-6hr",
]


ALLOWED_TYPES = [SQLVar.int, SQLVar.float]

DATA_DIR = "/tf/data"

AIR_SIGN_PATH = "./meta/air_signs.csv"
EBA_NAME_PATH = "./meta/iso_names.csv"

YEARS = list(range(2015, 2024))


@lru_cache(1)
def get_air_names(fn=AIR_SIGN_PATH):
    with open(fn, "r") as fp:
        return fp.readlines()


@lru_cache(1)
def get_eba_names(fn=EBA_NAME_PATH):
    with open(fn, "r") as fp:
        return fp.readlines()


class EBAMeta:
    """Class for extracting metadata about EBA dataset and saving to disk"""

    EBA_FILE = "EBA.txt"
    META_FILE = "metaseries.txt"
    ISO_NAME_FILE = "iso_name_file.json"

    EBA_TABLES = [
        EBAName.EBA,
        EBAName.INTERCHANGE,
    ]

    def __init__(self, eba_path="/tf/data/EBA/EBA20230302/"):
        self.eba_filename = os.path.join(eba_path, self.EBA_FILE)
        self.meta_file = os.path.join(eba_path, self.META_FILE)
        self.iso_file_map = os.path.join(eba_path, self.ISO_NAME_FILE)
        self.sqldr = SQLDriver()

    def extract_meta_data(self):
        # need checking on location and if file exists
        os.system(f"grep -r 'category_id' {self.eba_filename} > {self.meta_file}")

    def load_metadata(self) -> pd.DataFrame:
        meta_df = pd.read_json(self.meta_file, lines=True)
        return pd.DataFrame(meta_df)

    def parse_metadata(self, df: pd.DataFrame) -> Dict:
        """Grab names, abbreviations and category ids and save to dict"""
        iso_map = {}
        for _, row in df.iterrows():
            if "(" in row["name"]:
                tokens = re.findall(r"(\w+)", row["name"])
                name = " ".join(tokens[:-1])
                abbrv = tokens[-1]
                if abbrv == abbrv.upper():
                    iso_map[abbrv] = name

            # for ch in row.childseries
        return iso_map

    def save_iso_dict_json(self) -> str:
        """Load up meta data, extract names, save to json"""
        df = self.load_metadata()
        iso_map = self.parse_metadata(df)

        with open(self.iso_file_map, "w") as fp:
            json.dump(iso_map, fp)
            return self.iso_file_map

    @lru_cache()
    def load_iso_dict_json(self) -> Dict:
        with open(self.iso_file_map, "r") as fp:
            out_d = json.load(fp)
        return out_d

    def create_tables(self):
        """Create all relevant Tables for EBA data"""
        sql_comm = f"""CREATE TABLE IF NOT EXISTS {EBAName.EBA}
            ({ColName.TS} timestamp with time zone,
            {ColName.SOURCE} varchar(4),
            {ColName.MEASURE} varchar(20),
            {ColName.VAL} float,
            UNIQUE ({ColName.TS}, {ColName.SOURCE}, {ColName.MEASURE})
            );"""
        self.sqldr.execute_with_rollback(sql_comm, verbose=True)

        sql_comm = f"""CREATE TABLE IF NOT EXISTS {EBAName.INTERCHANGE}
            ({ColName.TS} timestamp with time zone,
            {ColName.SOURCE} varchar(4),
            {ColName.DEST} varchar(4),
            {ColName.VAL} float,
            UNIQUE ({ColName.TS}, {ColName.SOURCE}, {ColName.DEST})
            );"""
        self.sqldr.execute_with_rollback(sql_comm, verbose=True)

    def create_indexes(self):
        """Create all relevant Tables for EBA data"""

        sql_comm = f"""CREATE INDEX ix_{EBAName.EBA} ON {EBAName.EBA}
        ({ColName.TS}, {ColName.MEASURE}, {ColName.SOURCE});"""
        self.sqldr.execute_with_rollback(sql_comm, verbose=True)

        sql_comm = f"""CREATE INDEX ix_{EBAName.INTERCHANGE} ON {EBAName.INTERCHANGE}
        ({ColName.TS}, {ColName.SOURCE}, {ColName.DEST});"""
        self.sqldr.execute_with_rollback(sql_comm, verbose=True)

    def drop_tables(self, execute=False):
        """Drop the tables!
        execute=False means only print commands.
        execute=True will execute!
        """
        for table_name in [EBAName.EBA, EBAName.INTERCHANGE]:
            sql_comm = f"DROP TABLE IF EXISTS {table_name};"
            print(sql_comm)
            if execute is True:
                self.sqldr.execute_with_rollback(sql_comm, verbose=True)

    # def _get_create_eba_table_sql_wide(self, table_name: str, var_type: str) -> str:
    #     """Get String SQL Command to create SQL table for EIA EBA data for ISOs."""
    #     if var_type not in ALLOWED_TYPES:
    #         raise RuntimeError(f"{var_type} not in {ALLOWED_TYPES}!")

    #     str_list = [
    #         f"CREATE TABLE IF NOT EXISTS {table_name} ",
    #         "(",
    #         "ts timestamp,",
    #     ]
    #     if table_name == EBAName.INTERCHANGE:
    #         str_list += ["source varchar(4)," "dest varchar(4)," "val float)"]
    #     else:
    #         eba_names = self.load_iso_dict_json().keys()
    #         many_str_list = [f"{eba} {var_type}" for eba in eba_names]
    #         str_list += [", ".join(many_str_list)]
    #         str_list += [") PRIMARY KEY ts;"]
    #     return " ".join(str_list)

    def parse_eba_series_id(self, str):
        sub = str.split(".")[1:]  # drop the EBA
        source, dest = sub[0].split("-")
        tag = sub[1:-1].join("-")
        time = sub[-1]
        return source, dest, tag, time

    def load_data(self):
        """Load in relevant data series.  Only keep the ones on UTC time."""
        # load files one by one.
        # if in desired type, then insert.
        # before insert, check if times exists.
        # otherwise update based on time overlap.

        int_str = re.compile("Actual Net Interchange")
        utc_str = re.compile("UTC")

        # name_reg = re.compile(f'{name_lookup}') if name_lookup else None
        with jsonlines.open(self.eba_filename, "r") as fh:
            for dat in tqdm(fh):
                if not dat["series_id"] and not dat["data"]:
                    continue
                if utc_str not in dat["name"]:
                    continue
                source, dest, tag, time = self.parse_eba_series_id(dat["series_id"])
                if int_str in dat["name"]:
                    table_name = EBAName.INTERCHANGE
                    cols = ["ts", "source", "dest", "value"]
                    col_types = [SQLVar.timestamp, SQLVar.str, SQLVar.str, SQLVar.float]
                    unique_list = ["ts", "source", "dest"]
                    data_list = [(x[0], source, dest, x[1]) for x in dat["data"]]
                else:
                    table_name = EBAName.INTERCHANGE
                    cols = ["ts", "source", "measure", "value"]
                    col_types = [SQLVar.timestamp, SQLVar.str, SQLVar.str, SQLVar.float]
                    unique_list = ["ts", "source", "measure"]
                    data_list = [(x[0], source, tag, x[1]) for x in dat["data"]]
                self.sqldr.upsert_data_column(
                    table_name, cols, col_types, data_list, unique_list
                )


class ISDMeta:
    """Utils for getting Airport sensor data, setting up sql tables and loading data in."""

    def __init__(
        self,
        meta_file="/tf/data/air_merge_df.csv.gz",
        sign_file="/tf/data/air_signs.csv",
    ):
        self.meta_file = meta_file
        self.sign_file = sign_file
        self.sqldr = SQLDriver()
        self.ISD_TABLES = [ISDName.ISD]
        self.ISD_MEASURES = [
            (ISDName.TEMPERATURE, ISDDFName.TEMPERATURE),
            # (ISDName.WIND_DIR, ISDDFName.WIND_DIR),
            (ISDName.WIND_SPEED, ISDDFName.WIND_SPEED),
            (ISDName.PRECIP_1HR, ISDDFName.PRECIP_1HR),
        ]

    def get_air_meta_df(self) -> pd.DataFrame:
        air_df = pd.read_csv(self.meta_file, index_col=0)
        return air_df

    def save_callsigns(self):
        df = self.get_air_meta_df()
        df.sort_values(["ST", "CALL"])["CALL"].to_csv(
            self.sign_file, header=True, index=False
        )

    @lru_cache()
    def load_callsigns(self) -> List:
        return pd.read_csv(self.sign_file)["CALL"].tolist()

    def create_isd_meta(self):
        air_meta_create = """
        CREATE TABLE IF NOT EXISTS air_meta
        (id integer,
        name varchar(100),
        city varchar(100),
        state char(2),
        callsign char(4),
        usaf integer,
        wban integer,
        lat float,
        lng float);
        """
        with self.sqldr.conn.cursor() as cur:
            print(air_meta_create)
            cur.execute(air_meta_create)
            self.sqldr.conn.commit()

    def populate_isd_meta(self) -> List:
        """Populate SQL table with Airport metadata from Pandas DF"""
        air_df = self.get_air_meta_df()
        data_cols = ["name", "City", "ST", "CALL", "USAF", "WBAN", "LAT", "LON"]
        sub_data = air_df[data_cols].values.tolist()
        sql_cols = [
            ("id", SQLVar.int),
            ("name", SQLVar.str),
            ("city", SQLVar.str),
            ("state", SQLVar.str),
            ("callsign", SQLVar.str),
            ("usaf", SQLVar.int),
            ("wban", SQLVar.int),
            ("lat", SQLVar.float),
            ("lng", SQLVar.float),
        ]
        col_types = [x[1] for x in sql_cols]
        col_names = [x[0] for x in sql_cols]
        for i, D in enumerate(sub_data):
            D.insert(0, i)
        data_strings = [format_insert_str(D, col_types) for D in sub_data]

        self.sqldr.insert_data(ISDName.META, data_strings, col_list=col_names)
        return data_strings[:5]

    def create_tables(self):
        """Make time-series tables for ISD data.  Currently Temp, Wind speed, Precip"""
        sql_comm = f"""
        CREATE TABLE IF NOT EXISTS {ISDName.ISD}
            ({ColName.TS} timestamp with time zone,
            {ColName.CALL} char(4),
            {ColName.MEASURE} varchar(20),
            {ColName.VAL} float,
            UNIQUE ({ColName.TS}, {ColName.CALL}, {ColName.MEASURE})
            );
        """
        self.sqldr.execute_with_rollback(sql_comm, verbose=True)

    def create_indexes(self):
        """Make time-series tables for ISD data.  Currently Temp, Wind speed, Precip"""
        sql_comm = f"CREATE INDEX ix_{ISDName.ISD} ON {ISDName.ISD} ({ColName.TS}, {ColName.MEASURE}, {ColName.CALL});"
        self.sqldr.execute_with_rollback(sql_comm, verbose=True)

    # def _get_create_isd_table_sql_wide(self, table_name: str, var_type: str) -> str:
    #     """Get String SQL Command to create SQL table for NOAA ISD data from Airports.
    #     Old form with wide table format with one col per airport"""
    #     if var_type not in ALLOWED_TYPES:
    #         raise RuntimeError(f"{var_type} not in {ALLOWED_TYPES}!")
    #     str_list = [
    #         f"CREATE TABLE IF NOT EXISTS {table_name} ",
    #         "(ts timestamp,",
    #     ]
    #     call_signs = self.load_callsigns()
    #     many_str_list = [f"{cs} {var_type}" for cs in call_signs]
    #     str_list += [", ".join(many_str_list)]
    #     str_list += [") PRIMARY KEY ts;"]
    #     # index column on ts? or combine year/month?
    #     # How to handle bulk update? temp table with update?
    #     return " ".join(str_list)

    def drop_tables(self, execute=False):
        """Drop the tables!  Only sould be used when cleaning up setup.
        execute is not True means only print commands.
        execute = True will execute, and drop the table!
        """
        for table_name in [ISDName.ISD]:
            sql_comm = f"DROP TABLE IF EXISTS {table_name};"
            print(sql_comm)
            if execute is True:
                print(f"Dropping table {table_name}!")
                self.sqldr.execute_with_rollback(sql_comm, verbose=True)

    def get_isd_filenames(self):
        """Use ISD Meta table to build up known"""
        wban_usaf_list = self.sqldr.get_data(
            f"SELECT USAF, WBAN, CALLSIGN, LAT, LNG FROM {ISDName.META}"
        )
        file_list = []
        for usaf, wban, callsign, lat, lng in wban_usaf_list:
            for year in YEARS:
                filename = get_local_isd_path(str(year), usaf, wban)
                if os.path.exists(filename) and os.path.getsize(filename) > 0:
                    tzstr = get_tz(lng, lat)
                    file_list.append((filename, callsign, tzstr))
        return file_list

    def load_data(self, Nst=-1, Nmax=-1):
        """Load data for each station by year and insert desired data into columns of relevant tables.
        Tables / Columns governed by ISD_TABLES.  Converts all data timestamps appropriately to UTC.
        Each table has columns for each callsign.
        """
        files = self.get_isd_filenames()[:Nst]

        # out_sql = []
        for file, callsign, tzstr in tqdm(files):
            df = load_isd_df(file, tzstr)
            for measure, df_col in self.ISD_MEASURES:
                data = self.get_df_data_cols(df, df_col)[:Nmax]
                data_types = (SQLVar.timestamptz, SQLVar.str, SQLVar.str, SQLVar.float)
                cols = [ColName.TS, ColName.CALL, ColName.MEASURE, ColName.VAL]
                unique_list = [ColName.TS, ColName.CALL, ColName.MEASURE]
                data = [(x[0], callsign, measure, x[1]) for x in data]
                s_c = self.sqldr.upsert_data_column(
                    table_name=ISDName.ISD,
                    col_list=cols,
                    col_types=data_types,
                    data=data,
                    unique_list=unique_list,
                    val_col=ColName.VAL,
                )
                # out_sql.append(s_c)
        # return out_sql

    def get_df_data_cols(self, df: pd.DataFrame, df_col: pd.DataFrame) -> List:
        """Get list of columns suitable for loading into SQL.  Drop any null values."""
        sub_ser = df[df_col]
        sub_ser = sub_ser[sub_ser == sub_ser]
        times = sub_ser.index.map(
            lambda x: x.isoformat(timespec="seconds")  # .replace('T', ' ')
        ).values.tolist()  # get list of strings.
        data_vals = sub_ser.values.tolist()
        return list(zip(times, data_vals))


class SQLDriver:
    def __init__(self):
        self.conn = self.get_connection()

    def get_connection(self):
        """Get default connection"""
        db = os.environ.get("POSTGRES_DB", None)
        if not db:
            raise RuntimeError("SQLDriver could not find Postgres DB Name")

        pw = os.environ.get("POSTGRES_PASSWORD", "")
        if not pw:
            raise RuntimeError("SQLDriver could not find Postgres DB Password")
        user = os.environ.get("POSTGRES_USER", "")
        if not user:
            raise RuntimeError("SQLDriver could not find Postgres DB User")
        # pg_url = f"postgres://db:5432"
        conn = psycopg2.connect(
            dbname=db, user=user, password=pw, host="postgres", port=5432
        )
        return conn

    def rollback(self):
        """Rollback failed SQL transaction"""
        with self.conn.cursor() as cur:
            cur.execute("ROLLBACK")
            self.conn.commit()

    def execute_with_rollback(self, sql_com: str, verbose: bool = False):
        """Execute SQL command with rollback on exception."""
        try:
            if verbose:
                print(sql_com)
            with self.conn.cursor() as cur:
                rv = cur.execute(sql_com)
            self.conn.commit()
        except Exception as e:
            print("Exception raised!")
            print(repr(e))
            self.rollback()
            rv = None
        return rv

    def insert_data(
        self, table_name: str, data: List, col_list: Optional[List[str]] = None
    ):
        sql_com = self.get_insert_statement(table_name, data, col_list)
        rv = self.execute_with_rollback(sql_com)
        return rv

    def get_insert_statement(
        self, table_name: str, data_strings: list, col_list: Optional[List[str]] = None
    ):
        """Build up Postgres insert command."""
        sql_com = f"INSERT INTO {table_name}"
        if col_list:
            sql_com += f" ({', '.join(col_list)})"
        sql_com += " VALUES "
        sql_com += ", ".join(data_strings)
        sql_com += ";"
        return sql_com

    def upsert_data_column(
        self,
        table_name: str,
        col_list: List[str],
        col_types: List[str],
        unique_list: List[str],
        val_col: str,
        data: List[List],
    ) -> str:
        """Upsert Data

        Update columns with existing initial columns, and otherwise insert row.
        Useful for loading in column by column for wide tables.
        """
        col_str = ",".join(col_list)
        unique_str = ",".join(unique_list)
        sql_comm = f"INSERT INTO {table_name}({col_str}) VALUES"
        sql_comm += ",\n".join([format_insert_str(D, col_types) for D in data])
        sql_comm += (
            f"ON CONFLICT ({unique_str}) DO UPDATE SET {val_col}=EXCLUDED.{val_col};"
        )
        self.execute_with_rollback(sql_comm)
        return sql_comm

    def drop_table(self, table_name: str, force: bool = False):
        sql_com = f"DROP TABLE IF EXISTS {table_name};"
        if not force:
            print(sql_com)
        print(f"DELETING TABLE {table_name}")
        self.execute_with_rollback(sql_com)

    def get_data(self, sql_qry: str) -> List:
        """Execute a select query and return results."""
        with self.conn.cursor() as cur:
            cur.execute(sql_qry)
            rv = cur.fetchall()
        return rv

    def get_columns(self, table_name: str) -> List:
        """Return list of (name, column number, type) tuples for specified table"""
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM information_schema.columns WHERE table_name = '{table_name}'"
            )
            rv = cur.fetchall()
        # Get name, column number, and type
        rv = [(x[3], x[4], x[7]) for x in rv]
        return rv


# feels like there should be a structure for var type encapsulating name, sql column, and formating
# And this is the path to how you end up writing SQLAlchemy.


def format_insert_str(data: List, st_type_list: List) -> str:
    """Create insert string for inserting data record into SQL"""
    out_str = "("
    Ncol = len(st_type_list)
    for i, st in enumerate(data):
        st_type = st_type_list[i]
        if st_type == SQLVar.str:
            # just remove quotes.
            st0 = st.replace("'", "").replace('"', "")
            out_str += f"'{st0}'"
        elif st_type == SQLVar.timestamptz:
            out_str += f"'{st}'"
        else:
            out_str += f"{st}"
        if i < Ncol - 1:
            out_str += ", "
    out_str += ")"
    return out_str


# Going to stick to assumption that inserting code handles proper conversion.
