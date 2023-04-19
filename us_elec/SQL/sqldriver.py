import json
import os
import re
from typing import List, Optional, Tuple
import psycopg2
import subprocess

import pandas as pd

from functools import lru_cache

# from us_elec.SQL.sql_query import eba_table_template, isd_table_template, insert_eba_data


class TableType:
    EBA = "eba"
    NDFD = "ndfd"
    ISD = "isd"


class SQLVar:
    int = "integer"
    float = "float"
    str = "string"


class EBAName:
    DEMAND = "demand"
    DEMAND_FORECAST = "demand_forecast"
    NET_GENERATION = "net_generation"
    INTERCHANGE = "interchange"


class ISDName:
    TEMPERATURE = "temperature"
    WIND_DIR = "wind_dir"
    WIND_SPEED = "wind_speed"
    PRECIP_1HR = "precip_1hr"
    META = "air_meta"


ALLOWED_TYPES = [SQLVar.int, SQLVar.float]

DATA_DIR = "/tf/data"

AIR_SIGN_PATH = "./meta/air_signs.csv"
EBA_NAME_PATH = "./meta/iso_names.csv"


@lru_cache(1)
def get_air_names(fn=AIR_SIGN_PATH):
    with open(fn, "r") as fp:
        return fp.readlines()


@lru_cache(1)
def get_eba_names(fn=EBA_NAME_PATH):
    with open(fn, "r") as fp:
        return fp.readlines()


class EBAMeta:
    EBA_FILE = "EBA.txt"
    META_FILE = "metaseries.txt"
    ISO_NAME_FILE = "iso_name_file.json"
    """Class for extracting metadata about EBA dataset and saving to disk"""

    def __init__(self, eba_path="/tf/data/EBA/EBA20230302/"):
        self.eba_filename = os.path.join(eba_path, self.EBA_FILE)
        self.meta_file = os.path.join(eba_path, self.META_FILE)
        self.iso_file_map = os.path.join(eba_path, self.ISO_NAME_FILE)

    def extract_meta_data(self):
        # need checking on location and if file exists
        os.system(f"grep -r 'category_id' {self.eba_filename} > {self.meta_file}")

    def load_metadata(self):
        meta_df = pd.read_json(self.meta_file, lines=True)
        return meta_df

    def parse_metadata(self, df):
        """Grab names, abbreviations and category ids"""
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

    def save_iso_dict_json(self):
        """Load up meta data, extract names, save to json"""
        df = self.load_metadata()
        iso_map = self.parse_metadata(df)

        with open(self.iso_file_map, "w") as fp:
            json.dump(iso_map, fp)
            return self.iso_file_map

    @lru_cache()
    def load_iso_dict_json(self):
        with open(self.iso_file_map, "r") as fp:
            out_d = json.load(fp)
        return out_d


class AirMeta:
    """Utils for getting call signs, sorted by state"""

    def __init__(
        self,
        meta_file="/tf/data/air_merge_df.csv.gz",
        sign_file="/tf/data/air_signs.csv",
    ):
        self.meta_file = meta_file
        self.sign_file = sign_file

    def get_air_meta_df(self):
        air_df = pd.read_csv(self.meta_file, index_col=0)
        return air_df

    def save_callsigns(self):
        df = self.get_air_meta_df()
        df.sort_values(["ST", "CALL"])["CALL"].to_csv(
            self.sign_file, header=True, index=False
        )

    @lru_cache()
    def load_callsigns(self):
        return pd.read_csv(self.sign_file)["CALL"].tolist()


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

    def create_tables(self, name_type_var_list: List[Tuple[str]]) -> bool:
        for table_type, table_name, var_type in name_type_var_list:
            sql_comm = self._get_create_sql_template(table_type, table_name, var_type)
            self.execute_with_rollback(sql_comm, verbose=True)

    def _get_create_sql_template(
        self, table_type: str, table_name: str, var_type: str
    ) -> str:
        if table_type == TableType.EBA:
            # iterate over EBA columns
            return self._get_create_eba_table_sql(table_name, var_type)
        elif table_type == TableType.ISD:
            # iterate over ISD
            return self._get_create_isd_table_sql(table_name, var_type)

    def _get_create_eba_table_sql(self, table_name: str, var_type: str) -> str:
        """Get String SQL Command to create SQL table for EIA EBA data for ISOs."""
        if var_type not in ALLOWED_TYPES:
            raise RuntimeError(f"{var_type} not in {ALLOWED_TYPES}!")

        str_list = [
            f"CREATE TABLE IF NOT EXISTS {table_name} ",
            "(",
            "ts timestamp,",
        ]
        if table_name == EBAName.INTERCHANGE:
            str_list += ["source varchar(4)," "dest varchar(4)," "val float)"]
        else:
            eba_names = EBAMeta().load_iso_dict_json().keys()
            many_str_list = [f"{eba} {var_type}" for eba in eba_names]
            str_list += [", ".join(many_str_list)]
            str_list += [");"]
        return " ".join(str_list)

    def _get_create_isd_table_sql(self, table_name: str, var_type: str) -> str:
        """Get String SQL Command to create SQL table for NOAA ISD data from Airports."""
        if var_type not in ALLOWED_TYPES:
            raise RuntimeError(f"{var_type} not in {ALLOWED_TYPES}!")
        str_list = [
            f"CREATE TABLE IF NOT EXISTS {table_name} ",
            "(",
            "ts timestamp,",
        ]
        call_signs = AirMeta().load_callsigns()
        many_str_list = [f"{cs} {var_type}" for cs in call_signs]
        str_list += [", ".join(many_str_list)]
        str_list += [");"]
        # index column on ts? or combine year/month?
        # How to handle bulk update? temp table with update?
        return " ".join(str_list)

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
        sql_com += ', '.join(data_strings)
        sql_com += ";"
        return sql_com

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

    def rollback(self):
        """Rollback failed SQL transaction"""
        with self.conn.cursor() as cur:
            cur.execute("ROLLBACK")
            self.conn.commit()

    def get_columns(self, table_name: str) -> List:
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM information_schema.columns WHERE table_name = {table_name}"
            )
            rv = cur.fetchall()
        return rv


def create_eba_tables():
    sqldr = SQLDriver()
    EBA_TABLES = [
        (TableType.EBA, EBAName.DEMAND, SQLVar.float),
        (TableType.EBA, EBAName.DEMAND_FORECAST, SQLVar.float),
        (TableType.EBA, EBAName.NET_GENERATION, SQLVar.float),
        (TableType.EBA, EBAName.INTERCHANGE, SQLVar.float),
    ]
    sqldr.create_tables(EBA_TABLES)


def create_isd_tables():
    sqldr = SQLDriver()
    ISD_TABLES = [
        (TableType.ISD, ISDName.TEMPERATURE, SQLVar.float),
        (TableType.ISD, ISDName.WIND_DIR, SQLVar.float),
        (TableType.ISD, ISDName.WIND_SPEED, SQLVar.float),
        (TableType.ISD, ISDName.PRECIP_1HR, SQLVar.float),
    ]
    sqldr.create_tables(ISD_TABLES)


def create_isd_meta():
    sqldr = SQLDriver()
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
    with sqldr.conn.cursor() as cur:
        print(air_meta_create)
        cur.execute(air_meta_create)
        sqldr.conn.commit()


def populate_isd_meta():
    """Populate SQL table with Airport metadata from Pandas DF"""
    air_df = AirMeta().get_air_meta_df()
    data_cols = ["name", "City", "ST", "CALL", "USAF", "WBAN", "LAT", "LON"]
    sub_data = air_df[data_cols].values.tolist()
    sql_cols = [("id", SQLVar.int),
                ("name", SQLVar.str),
                ("city", SQLVar.str),
                ("state", SQLVar.str), 
                ("callsign", SQLVar.str),
                ("usaf", SQLVar.int),
                ("wban", SQLVar.int),
                ("lat", SQLVar.float),
                ("lng", SQLVar.float)]
    col_types = [x[1] for x in sql_cols]
    col_names = [x[0] for x in sql_cols]
    for i, D in enumerate(sub_data):
        D.insert(0, i)
    data_strings = [format_insert_str(D, col_types) for D in sub_data]
    sqldr = SQLDriver()

    sqldr.insert_data(ISDName.META, data_strings, col_list=col_names)
    return data_strings[:5]        

# feels like there should be a structure for var type encapsulating name, sql column, and formating
# And this is the path to how you end up writing SQLAlchemy.

def format_insert_str(data:List, st_type_list: List) -> str:
    """Create insert string for inserting data record into SQL
    """
    out_str = "("
    Ncol = len(st_type_list)
    for i, st in enumerate(data):
        st_type = st_type_list[i]
        if st_type == SQLVar.str:
            # just remove quotes.
            st0 = st.replace("\'",'').replace('\"','')
            out_str += f"'{st0}'"
        else:
            out_str += f"{st}"
        if i < Ncol-1:
            out_str += ", "
    out_str += ")"
    return out_str





# Going to stick to assumption that inserting code handles proper conversion.