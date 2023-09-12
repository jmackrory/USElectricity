import gzip
import json
import os
import re
from typing import Dict, List, Optional
import jsonlines

import psycopg2
import pandas as pd
from tqdm import tqdm

from functools import lru_cache

from us_elec.util.get_weather_data import get_local_isd_path


class TableType:
    EBA = "eba"
    NDFD = "ndfd"
    ISD = "isd"


class SQLVar:
    int = "integer"
    float = "float"
    str = "string"
    timestamptz = "timestamp with time zone"


class ColName:
    # Column names used across tables
    TS = "ts"
    ID = "id"
    SOURCE = "source"
    SOURCE_ID = "source_id"
    CALL = "callsign"
    CALL_ID = "callsign_id"
    MEASURE = "measure"
    MEASURE_ID = "measure_id"
    DEST = "dest"
    DEST_ID = "dest_id"
    VAL = "val"
    FULL_NAME = "full_name"
    ABBR = "abbr"


class EBAName:
    # Table names
    EBA = "eba"
    ISO_META = "eba_iso_meta"
    MEASURE = "eba_measure"
    INTERCHANGE = "eba_interchange"


class EBAAbbr:
    # measure names for EBA
    NG = "Net Generation"
    ID = "Net Interchange"
    DF = "Demand Forecast"
    D = "Demand"
    TI = "Total Interchange"


class EBAGenAbbr:
    # measure names for subtypes of EBA generation
    COL = "Generation - Coal"
    WAT = "Generation - Hydro"
    NG = "Generation - Natural Gas"
    NUC = "Generation - Nuclear"
    OIL = "Generation - Oil"
    OTH = "Generation - Other"
    SUN = "Generation - Solar"
    TOT = "Generation - Total"
    WND = "Generation - Wind"


class EBAExtra:
    # measure names for other regions
    CAN = "Canada (region)"
    MEX = "Mexico (region)"


class ISDName:
    # ISD Table names
    ISD = "ISD"
    AIR_META = "air_meta"
    MEASURE = "isd_measure"
    # TEMPERATURE = "temp"
    # WIND_DIR = "wind_dir"
    # WIND_SPEED = "wind_spd"
    # PRECIP_1HR = "precip_1hr"


class ISDDF:
    # ISD column names
    TIME = "time"
    YEAR = ("year",)
    MONTH = ("month",)
    DAY = ("day",)
    HOUR = ("hour",)
    TEMPERATURE = "temp"
    DEW_TEMPERATURE = "dew_temp"
    PRESSURE = "pressure"
    WIND_DIR = "wind_dir"
    WIND_SPEED = "wind_spd"
    CLOUD_COVER = "cloud_cov"
    PRECIP_1HR = "precip_1hr"
    PRECIP_6HR = "precip_6hr"

    ISD_COLS = [
        TIME,
        TEMPERATURE,
        DEW_TEMPERATURE,
        PRESSURE,
        WIND_DIR,
        WIND_SPEED,
        CLOUD_COVER,
        PRECIP_1HR,
        PRECIP_6HR,
    ]
    ind_name_lookup = {i: name for i, name in enumerate(ISD_COLS)}
    name_ind_lookup = {name: i for i, name in ind_name_lookup.items()}

    @classmethod
    def load_fwf_isd_file(cls, filename: str) -> List[List]:
        """Load ISD FWF datafile in directly.
        Currently relying on space between entries.
        """
        with gzip.open(filename, "r") as gz:
            lines = gz.readlines()
            vals = [[cls.parse_val(x) for x in L.split()] for L in lines]
            out_list = [cls.parse_times(V) for V in vals]
        return out_list

    @classmethod
    def parse_val(cls, x, null_vals=[-999, -9999]):
        ix = int(x)
        return ix if ix not in null_vals else None

    @classmethod
    def parse_times(cls, V):
        tstr = f"{V[0]}-{V[1]:02}-{V[2]:02}T{V[3]:02}:00:00"
        out_list = [tstr] + V[4:]
        return out_list

    @classmethod
    def get_cols(cls, cols: List[str], data_list: List[List]) -> List:
        col_ind = [cls.name_ind_lookup[x] for x in cols]
        out_data = [[x[i] for i in col_ind] for x in data_list]
        return out_data


ALLOWED_TYPES = [SQLVar.int, SQLVar.float]

DATA_DIR = "/tf/data"

AIR_SIGN_PATH = "./meta/air_signs.csv"
EBA_NAME_PATH = "./meta/iso_names.csv"

YEARS = list(range(2015, 2024))


@lru_cache()
def get_cls_attr_dict(cls):
    return {k: v for k, v in cls.__dict__.items() if not k.startswith("__")}


@lru_cache()
def get_reverse_cls_attr_dict(cls):
    return {v: k for k, v in cls.__dict__.items() if not k.startswith("__")}


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
        # eba table for demand/forecast/generation
        sql_comm = f"""CREATE TABLE IF NOT EXISTS {EBAName.EBA}
            ({ColName.TS} timestamp with time zone,
            {ColName.SOURCE_ID} smallint,
            {ColName.MEASURE_ID} smallint,
            {ColName.VAL} float
            );"""
        self.sqldr.execute_with_rollback(sql_comm, verbose=True)

        # interchange
        sql_comm = f"""CREATE TABLE IF NOT EXISTS {EBAName.INTERCHANGE}
            ({ColName.TS} timestamp with time zone,
            {ColName.SOURCE_ID} smallint,
            {ColName.DEST_ID} smallint,
            {ColName.VAL} float
            );"""
        self.sqldr.execute_with_rollback(sql_comm, verbose=True)

        # source meta
        sql_comm = f"""CREATE TABLE IF NOT EXISTS {EBAName.ISO_META}
            (id SMALLSERIAL,
            {ColName.FULL_NAME} varchar(100),
            {ColName.ABBR} varchar(10) UNIQUE
            );"""
        self.sqldr.execute_with_rollback(sql_comm, verbose=True)

        # measure meta
        sql_comm = f"""CREATE TABLE IF NOT EXISTS {EBAName.MEASURE}
            (id SMALLSERIAL,
            {ColName.FULL_NAME} varchar(100),
            {ColName.ABBR} varchar(10) UNIQUE
            );"""
        self.sqldr.execute_with_rollback(sql_comm, verbose=True)

    def create_indexes(self):
        """Create all relevant Tables for EBA data"""

        sql_comm = f"""CREATE INDEX ix_{EBAName.EBA}_t ON {EBAName.EBA}
        ({ColName.TS}, {ColName.MEASURE_ID}, {ColName.SOURCE_ID});"""
        self.sqldr.execute_with_rollback(sql_comm, verbose=True)

        sql_comm = f"""CREATE INDEX ix_{EBAName.INTERCHANGE}_t ON {EBAName.INTERCHANGE}
        ({ColName.TS}, {ColName.SOURCE_ID}, {ColName.DEST_ID});"""
        self.sqldr.execute_with_rollback(sql_comm, verbose=True)

    def drop_tables(self, execute=False):
        """Drop the tables!
        execute=False means only print commands.
        execute=True will execute!
        """
        for table_name in [
            EBAName.EBA,
            EBAName.INTERCHANGE,
            EBAName.MEASURE,
            EBAName.ISO_META,
        ]:
            sql_comm = f"DROP TABLE IF EXISTS {table_name};"
            print(sql_comm)
            if execute is True:
                self.sqldr.execute_with_rollback(sql_comm, verbose=True)

    def drop_indexes(self):
        """Make time-series tables for ISD data.  Currently Temp, Wind speed, Precip"""
        sql_comm = f"DROP INDEX IF EXISTS ix_{EBAName.EBA}_t;"
        self.sqldr.execute_with_rollback(sql_comm, verbose=True)

        sql_comm = f"DROP INDEX IF EXISTS ix_{EBAName.INTERCHANGE}_t;"
        self.sqldr.execute_with_rollback(sql_comm, verbose=True)

    @classmethod
    def get_name_abbr(cls, st):
        out_list = []
        for h in re.findall(r"for ([\w\s\,\.\-]+) \((\w+)\)", st):
            out_list.append((h[1], h[0]))
        for h in re.findall(r"to ([\w\s\,\.\-]+) \((\w+)\)", st):
            out_list.append((h[1], h[0]))
        return out_list

    @classmethod
    def find_dups(cls, data):
        found_data = set()
        for D in data:
            if D in found_data:
                print("dupe!", D)
            found_data.update([D])
        print(found_data)

    def _populate_iso_meta(self, abbr_name_tups):
        cols = [ColName.ABBR, ColName.FULL_NAME]
        data_list = [f"('{abbr}', '{fullname}')" for abbr, fullname in abbr_name_tups]
        self.sqldr.insert_data_column(
            table_name=EBAName.ISO_META,
            col_list=cols,
            data=data_list,
            unique_list=[ColName.ABBR],
            update=True,
            val_col=ColName.FULL_NAME,
        )

    def populate_meta_tables(self):
        """Populate EBA metadata about ISOs and Measure abbreviations"""
        # iso_map = self.load_iso_dict_json()
        df = self.load_metadata()
        iso_map = self.parse_metadata(df)

        more_names = get_cls_attr_dict(EBAExtra)

        self._populate_iso_meta(list(iso_map.items()))
        self._populate_iso_meta(list(more_names.items()))

        cols = [ColName.ABBR, ColName.FULL_NAME]
        eba_names = get_cls_attr_dict(EBAAbbr)
        data_list = [
            f"('{abbr}', '{fullname}')" for abbr, fullname in eba_names.items()
        ]
        gen_names = get_cls_attr_dict(EBAGenAbbr)
        # hard-coded abbreviation munging
        data_list += [
            f"('NG-{abbr}', '{fullname}')" for abbr, fullname in gen_names.items()
        ]
        self.sqldr.insert_data_column(
            table_name=EBAName.MEASURE,
            col_list=cols,
            # col_types=[SQLVar.str, SQLVar.str],
            data=data_list,
            unique_list=[ColName.ABBR],
            val_col=ColName.FULL_NAME,
            update=True,
        )

    def parse_eba_series_id(self, str):
        sub = str.split(".")[1:]  # drop the EBA
        source, dest = sub[0].split("-")
        tag = "-".join(sub[1:-1])
        time = sub[-1]
        return source, dest, tag, time

    def read_metadata(self):
        """Read in name / series_ids."""
        # load files one by one.
        # if in desired type, then insert.
        # before insert, check if times exists.
        # otherwise update based on time overlap.
        out_list = []
        with jsonlines.open(self.eba_filename, "r") as fh:
            for dat in tqdm(fh):
                if not dat.get("series_id") and not dat.get("data"):
                    continue
                out_list.append((dat["name"], dat["series_id"]))
        return out_list

    def load_data(self, Nseries=-1, Ntime=-1, bulk=True, update=False, Ncommit=50):
        """Load in relevant data series.  Only keep the ones on UTC time."""

        int_re = re.compile("Actual Net Interchange")
        utc_re = re.compile("UTC")

        file_count = 0
        with jsonlines.open(self.eba_filename, "r") as fh:
            for dat in tqdm(fh):
                name = dat["name"]
                if not dat.get("series_id") and not dat.get("data"):
                    continue
                if not utc_re.search(name):
                    continue
                if Nseries > 0 and file_count > Nseries:
                    break

                file_count += 1

                if int_re.search(name):
                    table_name = EBAName.INTERCHANGE
                    cols, unique_list, data_list = self._get_interchange_sql_inputs(
                        dat, Ntime=Ntime
                    )
                else:
                    table_name = EBAName.EBA
                    cols, unique_list, data_list = self._get_regular_sql_inputs(
                        dat, Ntime=Ntime
                    )
                self.sqldr.insert_data_column(
                    table_name=table_name,
                    col_list=cols,
                    data=data_list,
                    unique_list=unique_list,
                    val_col=ColName.VAL,
                    bulk=bulk,
                    update=update,
                )
                if file_count % Ncommit == 0 and bulk:
                    print("commiting")
                    self.sqldr.commit()
        self.sqldr.commit()

    def _get_interchange_sql_inputs(self, dat, Ntime=-1):
        source, dest, tag, _ = self.parse_eba_series_id(dat["series_id"])
        # print(dat["name"], dat["series_id"], source, dest, tag)
        source_id = self.get_eba_source_id(source, dat["name"])
        dest_id = self.get_eba_source_id(dest, dat["name"])

        cols = [ColName.TS, ColName.SOURCE_ID, ColName.DEST_ID, ColName.VAL]
        unique_list = [ColName.TS, ColName.SOURCE_ID, ColName.DEST_ID]
        sub_data = dat["data"][:Ntime]
        data_list = [
            self.get_data_insert_str(x, source_id, dest_id)
            # f"('{self.parse_time_str(x[0])}', {source_id}, {dest_id}, {x[1]})"
            for x in sub_data
        ]
        return cols, unique_list, data_list

    def _get_regular_sql_inputs(self, dat: Dict, Ntime: int = -1):
        source, dest, tag, _ = self.parse_eba_series_id(dat["series_id"])
        # dest_id = self.get_eba_source_id(dest)
        # print(dat["name"], dat["series_id"], source, dest, tag)
        source_id = self.get_eba_source_id(source, dat["name"])
        measure_id = self.get_eba_measure_id(tag)

        cols = [ColName.TS, ColName.SOURCE_ID, ColName.MEASURE_ID, ColName.VAL]
        unique_list = [ColName.TS, ColName.SOURCE_ID, ColName.MEASURE_ID]
        sub_data = dat["data"][:Ntime]
        data_list = [
            self.get_data_insert_str(x, source_id, measure_id)
            # f"('{self.parse_time_str(x[0])}', {source_id}, {measure_id}, {x[1]})"
            for x in sub_data
        ]
        return cols, unique_list, data_list

    @classmethod
    def parse_time_str(cls, V):
        # convert time strings like : YYYYMMDDTHHZ to YYYY-MM-DDTHH:00:00
        return f"{V[:4]}-{V[4:6]}-{V[6:11]}:00:00"

    @classmethod
    def get_data_insert_str(cls, x, source_id, measure_id):
        if x[1]:
            return f"('{cls.parse_time_str(x[0])}', {source_id}, {measure_id}, {x[1]})"
        else:
            return f"('{cls.parse_time_str(x[0])}', {source_id}, {measure_id}, NULL)"

    def get_eba_source_id(self, source, name, dpth=0):
        qry_str = (
            f"SELECT id FROM {EBAName.ISO_META} WHERE {ColName.ABBR} = '{source}';"
        )
        source_id = self.sqldr.get_data(qry_str)

        if dpth > 2:
            raise RuntimeError(f"EBA - No source found for {source_id}")

        if not source_id:
            # try to insert source,
            print(f"EBA - No source found for {source_id}.  Trying update")
            abbr_name_tups = self.get_name_abbr(name)
            self._populate_iso_meta(abbr_name_tups)
            return self.get_eba_source_id(source, name, dpth + 1)
        return source_id[0][0]

    def get_eba_measure_id(self, measure):
        qry_str = (
            f"SELECT id FROM {EBAName.MEASURE} WHERE {ColName.ABBR} = '{measure}';"
        )
        measure_id = self.sqldr.get_data(qry_str)
        if not measure_id:
            raise RuntimeError(f"No measure found for {measure}")
        return measure_id[0][0]


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
            ISDDF.TEMPERATURE,
            ISDDF.WIND_SPEED,
            ISDDF.PRECIP_1HR,
        ]

    def dummy_method(self):
        print("changed it more! in dummy")

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
        # table for data about stations
        air_meta_create = f"""
        CREATE TABLE IF NOT EXISTS {ISDName.AIR_META}
        (id integer,
        name varchar(100),
        city varchar(100),
        state char(2),
        callsign char(4) UNIQUE,
        usaf integer,
        wban integer,
        lat float,
        lng float);
        """
        with self.sqldr.conn.cursor() as cur:
            print(air_meta_create)
            cur.execute(air_meta_create)
            self.sqldr.conn.commit()

        # data table to store measure names.
        meta_table_create = f"""
        CREATE TABLE IF NOT EXISTS {ISDName.MEASURE}
        (id SMALLSERIAL,
        measure varchar(20) UNIQUE
        );
        """
        with self.sqldr.conn.cursor() as cur:
            print(meta_table_create)
            cur.execute(meta_table_create)
            self.sqldr.conn.commit()
        return

    def populate_isd_meta(self):
        """Populate SQL table with Airport metadata from Pandas DF.  Also populate measure table"""
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
        # effectively do nothing on conflict.
        val_col = ColName.CALL
        self.sqldr.insert_data_column(
            table_name=ISDName.AIR_META,
            data=data_strings,
            col_list=col_names,
            val_col=val_col,
            unique_list=[ColName.CALL],
            update=True,
        )
        return

    def populate_measures(self):
        # populate measure table
        data_list = [
            f"({idx}, '{abbr}')" for idx, abbr in ISDDF.ind_name_lookup.items()
        ]
        print(data_list)
        cols = ["id", "measure"]
        self.sqldr.insert_data_column(
            table_name=ISDName.MEASURE,
            col_list=cols,
            data=data_list,
            unique_list=[ColName.MEASURE],
            val_col=ColName.MEASURE,
            update=True,
        )

    def create_tables(self):
        """Make time-series tables for ISD data.  Currently Temp, Wind speed, Precip"""
        sql_comm = f"""
        CREATE TABLE IF NOT EXISTS {ISDName.ISD}
            ({ColName.TS} timestamp with time zone,
            {ColName.CALL_ID} smallint,
            {ColName.MEASURE_ID} smallint,
            {ColName.VAL} float
            );
        """
        # removed: UNIQUE ({ColName.TS}, {ColName.CALL}, {ColName.MEASURE})
        # could alter to use call_id and measure_id as integers
        self.sqldr.execute_with_rollback(sql_comm, verbose=True)

    def create_indexes(self):
        """Make time-series tables for ISD data.  Currently Temp, Wind speed, Precip"""
        sql_comm = f"""CREATE INDEX ix_{ISDName.ISD} ON {ISDName.ISD}
        ({ColName.TS}, {ColName.MEASURE_ID}, {ColName.CALL_ID});"""
        self.sqldr.execute_with_rollback(sql_comm, verbose=True)

    def drop_tables(self, execute=False):
        """Drop the tables!  Only sould be used when cleaning up setup.
        execute is not True means only print commands.
        execute = True will execute, and drop the table!
        """
        for table_name in [ISDName.ISD, ISDName.AIR_META]:
            sql_comm = f"DROP TABLE IF EXISTS {table_name};"
            if execute is True:
                print(f"Dropping table {table_name}!")
                self.sqldr.execute_with_rollback(sql_comm, verbose=True)

    def drop_indexes(self):
        """Make time-series tables for ISD data.  Currently Temp, Wind speed, Precip"""
        sql_comm = f"DROP INDEX IF EXISTS ix_{ISDName.ISD};"
        self.sqldr.execute_with_rollback(sql_comm, verbose=True)

    def get_isd_filenames(self):
        """Use ISD Meta table to build up known file list"""
        wban_usaf_list = self.sqldr.get_data(
            f"SELECT USAF, WBAN, CALLSIGN FROM {ISDName.AIR_META} ORDER BY CALLSIGN"
        )
        file_list = []
        for usaf, wban, callsign in wban_usaf_list:
            for year in YEARS:
                filename = get_local_isd_path(str(year), usaf, wban)
                if os.path.exists(filename) and os.path.getsize(filename) > 0:
                    file_list.append((filename, callsign))
        return file_list

    def load_data(self, Nstation=-1, Ntime=-1, bulk=True, update=False, Ncommit=50):
        """Load data for each station by year and insert desired data into columns of relevant tables.
        Tables / Columns governed by ISD_TABLES.  Converts all data timestamps appropriately to UTC.
        Each table has columns for each callsign.
        """
        files = self.get_isd_filenames()[:Nstation]

        # out_sql = []
        for file_count, (file, callsign) in enumerate(tqdm(files)):
            # print(file, callsign)
            # df = load_isd_df(file)
            data_list = ISDDF.load_fwf_isd_file(file)
            callsign_id = self.get_callsign_id(callsign)
            for measure in self.ISD_MEASURES:
                measure_id = self.get_measure_id(measure)
                df_cols = [ISDDF.TIME, measure]
                sub_data = ISDDF.get_cols(df_cols, data_list)[:Ntime]

                data = [
                    self.get_data_insert_str(x, callsign_id, measure_id)
                    for x in sub_data
                ]
                cols = [ColName.TS, ColName.CALL_ID, ColName.MEASURE_ID, ColName.VAL]
                unique_list = [ColName.TS, ColName.CALL_ID, ColName.MEASURE_ID]

                self.sqldr.insert_data_column(
                    table_name=ISDName.ISD,
                    col_list=cols,
                    data=data,
                    unique_list=unique_list,
                    val_col=ColName.VAL,
                    bulk=bulk,
                    update=update,
                )
            if file_count % Ncommit == 0 and bulk:
                self.sqldr.commit()
        self.sqldr.commit()

    def get_callsign_id(self, callsign):
        return self.sqldr.get_data(
            f"SELECT {ColName.ID} FROM {ISDName.AIR_META} WHERE {ColName.CALL}='{callsign}'"
        )[0][0]

    def get_measure_id(self, measure):
        return self.sqldr.get_data(
            f"SELECT {ColName.ID} FROM {ISDName.MEASURE} WHERE {ColName.MEASURE}='{measure}'"
        )[0][0]

    @classmethod
    def get_data_insert_str(cls, x, callsign, measure):
        if x[1]:
            return f"('{x[0]}', {callsign}, {measure}, {x[1]})"
        else:
            return f"('{x[0]}', {callsign}, {measure}, NULL)"


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

    def commit(self):
        self.conn.commit()

    def execute_with_rollback(
        self, sql_com: str, verbose: bool = False, bulk: bool = False
    ):
        """Execute SQL command with rollback on exception."""
        try:
            if verbose:
                print(sql_com)
            with self.conn.cursor() as cur:
                rv = cur.execute(sql_com)
            if not bulk:
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

    def insert_data_column(
        self,
        table_name: str,
        col_list: List[str],
        # col_types: List[str],
        unique_list: List[str],
        val_col: str,
        data: List[str],
        update: bool = False,
        bulk: bool = False,
    ) -> str:
        """Upsert Data

        Update columns with existing initial columns, and otherwise insert row.
        Useful for loading in column by column for wide tables, or when refreshing.
        """
        col_str = ",".join(col_list)
        sql_comm = f"INSERT INTO {table_name}({col_str}) VALUES"
        sql_comm += ",".join(data)
        if update:
            unique_str = ",".join(unique_list)
            sql_comm += f"ON CONFLICT ({unique_str}) DO UPDATE SET {val_col}=EXCLUDED.{val_col};"
        else:
            sql_comm += ";"
        self.execute_with_rollback(sql_comm, bulk=bulk)
        return sql_comm

    def drop_table(self, table_name: str, force: bool = False):
        sql_com = f"DROP TABLE IF EXISTS {table_name};"
        if not force:
            print(sql_com)
        print(f"DELETING TABLE {table_name}")
        self.execute_with_rollback(sql_com)
        return

    def get_data(self, sql_qry: str) -> Optional[List]:
        """Execute a select query and return results."""
        with self.conn.cursor() as cur:
            try:
                cur.execute(sql_qry)
                rv = cur.fetchall()
                return rv
            except Exception as e:
                print(f"SQLDriver.get_data: Failed on {sql_qry}")
                print(e)
                self.rollback()
        return list()

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
    """Create insert string for inserting data record into SQL
    Converts each entry in data into a record to insert.
    Takes data = [["1992/01/01 12:00:00", "kblah", "temperature", 1],...]
          st_type_list = [datetime, str, str, float]
    And spits out a big string.

    """
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
