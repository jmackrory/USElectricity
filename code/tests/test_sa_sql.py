from unittest import TestCase
from unittest.mock import patch

from tests.utilities import get_mock_creds
from us_elec.SQL.sa_sql import (
    EBAData,
    EBAMeta,
    ISDData,
    ISDMeta,
    create_indexes,
    create_tables,
    drop_tables,
    init_sqlalchemy,
)


def get_conn_db(conn):
    conn_ps = conn.get_dsn_parameters().get("dbname")


@patch("us_elec.SQL.sa_sql.get_creds", get_mock_creds)
class Tests(TestCase):
    @classmethod
    def setUpClass(cls):
        import subprocess

        r0 = subprocess.run("id")
        print(r0)
        test_creds = get_mock_creds()
        init_sqlalchemy(test_creds)
        create_tables()
        # create_indexes()

    @classmethod
    def tearDownClass(cls):
        pass
        # drop_tables()
        # eba_db = cls.eba.sqldr.get_db_name()
        # if eba_db == "test":
        #    cls.eba.drop_tables(execute=True)
        #   cls.eba.drop_indexes()

        # isd_db = cls.isd.sqldr.get_db_name()
        # if isd_db == "test":
        #    cls.isd.drop_tables(execute=True)
        #    cls.isd.drop_indexes()

    def test_creds(self):
        from us_elec.SQL.sa_sql import cool_func

        cool_func()
        self.assertTrue(True)

    def test_basic(self):
        print("in sa_sql tests")
        self.assertEqual(2 + 2, 4)
