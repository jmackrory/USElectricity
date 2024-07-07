import pytest
from test_utilities import get_mock_creds
from us_elec.SQL.sa_sql import (
    EBA,
    ISD,
    EBAMeta,
    ISDMeta,
    create_indexes,
    create_tables,
    drop_indexes,
    drop_tables,
    init_sqlalchemy,
)


def get_conn_db(conn):
    conn_ps = conn.get_dsn_parameters().get("dbname")


class Tests:
    @classmethod
    def setup_class(cls):
        print("Setting up class")
        import subprocess

        r0 = subprocess.run("id")
        print(r0)
        test_creds = get_mock_creds()
        init_sqlalchemy(test_creds)

        create_tables(test_creds.db)
        create_indexes(test_creds.db)

    @classmethod
    def teardown_class(cls):
        print("Tearing down class")
        test_creds = get_mock_creds()
        drop_tables(test_creds.db)
        drop_indexes(test_creds.db)

    def test_creds(self):
        print("in first test")
        assert True

    def test_basic(setup_dbs):
        print("in sa_sql tests")
        assert 2 + 2 == 4
