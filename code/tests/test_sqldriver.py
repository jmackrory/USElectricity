import os
from unittest import TestCase
from unittest.mock import patch

from tests.utilities import get_mock_creds
from us_elec.SQL.sqldriver import EBAMeta, ISDMeta, SQLDriver


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

        cls.sqldr = SQLDriver(get_mock_creds())

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
        conn_ps = self.sqldr.conn.get_dsn_parameters()
        print(conn_ps)
        self.assertTrue(conn_ps["dbname"] == "test")

    def test_select(self):
        rv = self.sqldr.get_data("SELECT * FROM ISD_META LIMIT 5")
        print(rv)
