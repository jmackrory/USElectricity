import pytest
from sqlalchemy import select
from test_utilities import get_mock_creds
from us_elec.SQL.sa_sql import (
    Airport,
    DBSession,
    EBAData,
    EBAMeasure,
    EBAMeta,
    ISDData,
    ISDMeasure,
    create_indexes,
    create_tables,
    drop_indexes,
    drop_tables,
    init_sqlalchemy,
)
from us_elec.SQL.sqldriver import ISDDF

count = 0


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

        cls.EBAData = EBAData()
        cls.EBAData.populate_meta_tables()

        cls.ISDData = ISDData()
        cls.ISDData.populate_isd_meta()
        cls.ISDData.populate_measures()

    @classmethod
    def teardown_class(cls):
        print("Tearing down class")
        test_creds = get_mock_creds()
        drop_tables(test_creds.db)
        drop_indexes(test_creds.db)

    def test_check_eba_populate_metadata(self):
        global count
        print(f"executing test #{count}: populate EBA")
        count += 1

        # Check Locations
        with DBSession() as sess, sess.begin():
            ca_rv = sess.execute(
                select(EBAMeta.id, EBAMeta.full_name).where(
                    EBAMeta.full_name.contains("%California%")
                )
            ).all()
        print(ca_rv)
        assert len(ca_rv) > 0
        # Check Measures
        with DBSession() as sess, sess.begin():
            rv = sess.execute(
                select(EBAMeasure.id, EBAMeasure.full_name).where(
                    EBAMeasure.full_name.contains("%Solar%")
                )
            ).all()
        assert len(rv) > 0
        print(rv)

    def test_check_isd_populate_metadata(self):
        global count
        print(f"executing test #{count}: populate ISD")
        count += 1
        # Check Locations Populated
        with DBSession() as sess, sess.begin():
            ca_rv = sess.scalars(
                select(Airport.callsign).where(Airport.state == "CA")
            ).all()
        print(ca_rv)
        assert len(ca_rv) > 0

        # Check Measures Populated
        with DBSession() as sess, sess.begin():
            rv = sess.execute(
                select(ISDMeasure.id, ISDMeasure.abbr).where(
                    ISDMeasure.abbr == ISDDF.TEMP
                )
            ).first()
        assert len(rv) > 0

    def _get_subset_isd_data(self):
        with DBSession() as sess, sess.begin():
            ca_rv = sess.scalars(
                select(Airport.callsign).where(Airport.state == "CA")
            ).all()
        ca_callsigns = set(ca_rv)

        isd_filenames = ISDData().get_isd_filenames(ca_callsigns)
        sub_filenames = [
            f[0]
            for f in isd_filenames
            if ("2022" in f[0]) and f[1] in ("KSFO", "KLAX", "KSAN")
        ]
        return sub_filenames


[
    "/home/root/data/ISD/722950-23174-2022.gz",
    "/home/root/data/ISD/722900-23188-2022.gz",
    "/home/root/data/ISD/724940-23234-2022.gz",
]
