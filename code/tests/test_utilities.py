import os
from collections import namedtuple

from us_elec.SQL.sqldriver import Creds


def get_mock_creds():
    """Mock testing function.  Reproduces us_elec.SQL.sqldriver.get_creds.get_mock_creds
    Avoid putting testing branch logic inside util code."""
    db = os.environ.get("PG_TEST_DB", None)
    if not db:
        raise RuntimeError("SQLDriver could not find Test Postgres DB Name")
    print("Got DB name", db)
    if db != "test":
        raise RuntimeError("Must be Test DB!")

    pw = os.environ.get("PG_TEST_PASSWORD", "")
    if not pw:
        raise RuntimeError("SQLDriver could not find Test Postgres DB Password")
    user = os.environ.get("PG_TEST_USER", "")
    if not user:
        raise RuntimeError("SQLDriver could not find Test Postgres DB User")
    return Creds(db, pw, user)
