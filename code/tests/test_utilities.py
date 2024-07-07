import json
import os
import re
from collections import namedtuple

import jsonlines
from us_elec.SQL.constants import DATA_DIR
from us_elec.SQL.sqldriver import ISDDF, Creds


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


def get_isd_fixture_files():
    isddf = ISDDF()
    # load air_df
    # filter to callsigns in KLAX, KSFO, KSAN
    # get files
    # copy to /test/fixtures


def get_eba_fixture_files(
    target_file="EBA/EBA_CA.txt",
    yearmonth="202207",
    out_name="/home/root/code/test/fixtures/EBA/EBA_CA202207.txt",
):
    full_filename = os.path.join(DATA_DIR, target_file)
    utc_re = re.compile("UTC")

    out_list = []
    with jsonlines.open(full_filename, "r") as fh:
        for dat in fh:
            name = dat["name"]
            if not dat.get("series_id") and not dat.get("data"):
                continue
            if not utc_re.search(name):
                continue
            out_D = {k: v for k, v in dat.items() if k != "data"}
            out_D["data"] = [D for D in dat["data"] if D[0].startswith(yearmonth)]
            out_list.append(out_D)
    with jsonlines.open(out_name), "wb" as fh:
        fh.write_all([json.dumps(f) for f in out_list])
    return out_list
