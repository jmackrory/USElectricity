import os
import psycopg2

#from us_elec.SQL.sql_query import eba_table_template, isd_table_template, insert_eba_data

class TableType:
    EBA = 'eba'
    NDFD = 'ndfd'
    ISD = 'isd'


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
            return eba_table_template
        elif table_type == TableType.ISD:
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
