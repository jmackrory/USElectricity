import json
import os
import re
import psycopg2
import subprocess

import pandas as pd


from functools import lru_cache

#from us_elec.SQL.sql_query import eba_table_template, isd_table_template, insert_eba_data

class TableType:
    EBA = 'eba'
    NDFD = 'ndfd'
    ISD = 'isd'


DATA_DIR = '/tf/data'

AIR_SIGN_PATH = "./meta/air_signs.csv"
EBA_NAME_PATH = "./meta/iso_names.csv"

@lru_cache(1)
def get_air_names(fn=AIR_SIGN_PATH):
    with open(fn, 'r') as fp:
        return fp.readlines()


@lru_cache(1)
def get_eba_names(fn=EBA_NAME_PATH):
    with open(fn, 'r') as fp:
        return fp.readlines()


class EBAMeta():
    EBA_FILE = 'EBA.txt'
    META_FILE = 'metaseries.txt'
    ISO_NAME_FILE = 'iso_name_file.json'
    """Class for extracting metadata about EBA dataset and saving int"""
    def __init__(self, eba_path = '/tf/data/EBA/EBA20230302/'):
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
        parent_map = {}
        iso_map = {}
        for _, row in df.iterrows():
            if '(' in row['name']:
                tokens = re.findall('(\w+)', row['name'])
                name = ' '.join(tokens[:-1])
                abbrv = tokens[-1]
                if abbrv == abbrv.upper():
                    iso_map[abbrv] = name

            #for ch in row.childseries
        return iso_map

    def save_iso_dict_json(self):
        """Load up meta data, extract names, save to json"""
        df = self.load_metadata()
        iso_map = self.parse_metadata(df)

        with open(self.iso_file_map, 'w') as fp:
            json.dump(iso_map, fp)
            return self.iso_file_map

    def load_iso_dict_json(self):
        with open(self.iso_file_map, 'r') as fp:
            out_d = json.load(fp)
        return out_d


class AirMeta():
    """Utils for getting call signs, sorted by state"""
    def __init__(self, meta_file='/tf/data/air_merge_df.csv.gz', sign_file='/tf/data/air_signs.csv'):
        self.meta_file = meta_file
        self.sign_file = sign_file

    def get_air_meta_df(self):
        air_df = pd.read_csv(self.meta_file, index_col=0)
        return air_df

    def save_callsigns(self):
        df = self.get_air_meta_df()
        df.sort_values(['ST', 'CALL'])['CALL'].to_csv(self.sign_file, header=True, index=False)

    def load_callsigns(self):
        return pd.read_csv(self.sign_file)['CALL'].tolist()


class SQLDriver():

    def __init__(self):
        self.conn = self.get_connection()

    def get_connection(self):
        """Get default connection"""
        db = os.environ.get('POSTGRES_DB', None)
        if not db:
            raise RuntimeError('SQLDriver could not find Postgres DB Name')

        pw = os.environ.get('POSTGRES_PASSWORD', '')
        if not pw:
            raise RuntimeError('SQLDriver could not find Postgres DB Password')
        user = os.environ.get('POSTGRES_USER', '')
        if not user:
            raise RuntimeError('SQLDriver could not find Postgres DB User')
        pg_url=f'postgres://db:5432'
        conn = psycopg2.connect(dbname=db, user=user, password=pw, host='postgres', port=5432)
        return conn

    def create_tables(self, table_names, table_type):
        for name, table_type in table_names:
            sql_template = self._get_create_sql_template(table_type)
            sql_comm = elec_ta
        pass

    def _get_create_sql_template(table_type: str) -> str:
        if table_type == TableType.EBA:
            # iterate over EBA columns
            return eba_table_template
        elif table_type == TableType.ISD:
            # iterate over ISD
            return isd_table_template

    def insert_data(table, data):
        if table_type == TableType.EBA:
            return eba_table_template
        elif table_type == TableType.ISD:
            return isd_table_template

        pass

    def get_data(self, sql_qry):
        with self.conn.cursor() as cur:
            cur.execute(sql_qry)
            rv = cur.fetchall()
        return rv
