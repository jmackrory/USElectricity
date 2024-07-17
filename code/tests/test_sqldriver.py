import os
from unittest import TestCase
from unittest.mock import patch

import pytest
from test_utilities import get_mock_creds
from us_elec.SQL.sqldriver import EBADriver, ISDDriver, SQLDriver

# Set up Fixtures for Start


# Set up Fixtures for Stop


def drop_test_tables(driver):
    eba_db = driver.sqldr.get_db_name()
    if eba_db == "test":
        print(f"Dropping Test Tables for {driver.NAME}")
        driver.drop_tables(execute=True)
        driver.drop_indexes()
    else:
        print(f"Not dropping {eba_db}")


class Tests(TestCase):
    @classmethod
    def setUpClass(cls):
        import subprocess

        r0 = subprocess.run("id")
        print(r0)

        sql_creds = get_mock_creds()
        cls.eba = EBADriver(sql_creds)
        print("Dropping")
        drop_test_tables(cls.eba)
        print("Creating")
        cls.eba.create_tables()
        # cls.eba.create_meta_table()
        # cls.eba.populate_meta_tables()
        #

        cls.isd = ISDDriver(sql_creds)
        print("Dropping")
        drop_test_tables(cls.isd)
        print("Creating")
        cls.isd.create_tables()
        cls.isd.create_isd_meta()
        # cls.isd.populate_isd_meta()
        # cls.isd.populate_measures()

        cls.sqldr = SQLDriver(sql_creds)

    @classmethod
    def tearDownClass(cls):
        drop_test_tables(cls.eba)
        drop_test_tables(cls.isd)

    def test_basic(self):
        self.assertEqual(2 + 2, 4)

    def test_mock(self):
        conn_ps = self.sqldr.conn.get_dsn_parameters()
        print(conn_ps)
        self.assertTrue(conn_ps["dbname"] == "test")

    def test_select(self):
        # pytest.set_trace()
        rv = self.sqldr.get_data("SELECT * FROM AIRPORT LIMIT 5;")
        print(rv)
