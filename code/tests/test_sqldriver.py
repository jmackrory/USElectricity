import os
from unittest import TestCase
from unittest.mock import patch

from us_elec.SQL.sqldriver import EBAMeta, ISDMeta, SQLDriver

from tests.utilities import get_mock_creds


def get_conn_db(conn):
    conn_ps = conn.get_dsn_parameters().get("dbname")


@patch("us_elec.SQL.sqldriver.get_creds", get_mock_creds)
class Tests(TestCase):
    @classmethod
    def setUpClass(cls):
        import subprocess

        r0 = subprocess.run("id")
        print(r0)
        cls.eba = EBAMeta()
        cls.eba.create_tables()
        # cls.eba.create_meta_table()
        cls.isd = ISDMeta()
        cls.isd.create_tables()
        pass

    @classmethod
    def tearDownClass(cls):
        eba_db = cls.eba.sqldr.get_db_name()
        if eba_db == "test":
            cls.eba.drop_tables(execute=True)
            cls.eba.drop_indexes()

        isd_db = cls.isd.sqldr.get_db_name()
        if isd_db == "test":
            cls.isd.drop_tables(execute=True)
            cls.isd.drop_indexes()

    def test_basic(self):
        self.assertEqual(2 + 2, 4)

    def test_mock(self):
        sql = SQLDriver()
        conn_ps = sql.conn.get_dsn_parameters()
        print(conn_ps)
        self.assertTrue(conn_ps["dbname"] == "test")
