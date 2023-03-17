import json
import os
import psycopg2

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
    """Class for extracting metadata about EBA dataset and saving int"""
    def __init__(self, fn):
        self.eba_filename = fn

    def extract_meta_data(self, save_loc):
        # need checking on location and if file exists

        # subprocess.run("grep", "-r", "'category_id'", "/tf/data/EBA/EBA20230302/EBA.txt", ">", self.meta_file)
        pass

    def load_metadata(self, meta_loc):
        fn = '/tf/data/EBA/EBA20230302/metaseries.txt'
        meta_df = pd.read_json(fn, lines=True)

    def parse_metadata(df):
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

    def save_iso_map(idict, fn='/tf/data/EBA/EBA20230302_iso_map.json'):
        with open(fn, 'w') as fp:
            json.dump(idict, fp)
            return fn

    def load_iso_map(fn='/tf/data/EBA/EBA20230302_iso_map.json'):
        with open(fn, 'r') as fp:
            out_d = json.load(fp)
        return out_d


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
