"""
 Use SQLAlchemy to define and handle table creation rather than bodging together
 my own ORM/abstraction layers.

"""
import gzip
import json
import os
import re
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

import jsonlines
import pandas as pd
import tqdm
from sqlalchemy import Index, create_engine, insert, select
from sqlalchemy.orm import (
    Mapped,
    Session,
    declarative_base,
    mapped_column,
    scoped_session,
    sessionmaker,
)
from us_elec.SQL.constants import (
    DATA_DIR,
    YEARS,
    ColName,
    EBAAbbr,
    EBAExtra,
    EBAGenAbbr,
    TableName,
)
from us_elec.SQL.sqldriver import ISDDF, Creds, get_cls_attr_dict
from us_elec.util.get_weather_data import get_local_isd_path

# What exactly is going on here?
# Is this a global session?  Should I not be using local, temporary sessions?

SQLBase = declarative_base()
DBSession = scoped_session(sessionmaker())
engine = None

isddf = ISDDF()


def get_engine(creds: Creds):
    print(f"Creating Engine for {creds.user} in {creds.db}")

    engine = create_engine(
        f"postgresql+psycopg2://{creds.user}:{creds.pw}@postgres:5432/{creds.db}"
    )
    return engine


def init_sqlalchemy(creds: Tuple[str]):
    global engine, DBSession
    engine = get_engine(creds)
    DBSession.remove()
    DBSession.configure(bind=engine, autoflush=False, expire_on_commit=False)


def create_tables(db: str):
    global engine
    if engine.url.database != db:
        return
    SQLBase.metadata.create_all(engine, checkfirst=True)


def drop_tables(db: str):
    global engine
    if engine.url.database != db:
        return
    SQLBase.metadata.drop_all(engine, checkfirst=True)


#########################################################
#
# EBA SQL Alchemy Tables
#
#########################################################


class EBA(SQLBase):
    """SQLAlchemy table for EBA Data"""

    __tablename__ = TableName.EBA
    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime]
    source_id: Mapped[int]
    measure_id: Mapped[int]
    val: Mapped[float]


class EBAInter(SQLBase):
    """SQLAlchemy table for EBA Interchange Data"""

    __tablename__ = TableName.INTERCHANGE
    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime]
    source_id: Mapped[int]
    dest_id: Mapped[int]
    val: Mapped[float]


class EBAMeta(SQLBase):
    """SQLAlchemy table for EBA MetaData"""

    __tablename__ = TableName.EBA_META
    id: Mapped[int] = mapped_column(primary_key=True)
    full_name: Mapped[str]
    abbr: Mapped[str]


class EBAMeasure(SQLBase):
    __tablename__ = TableName.EBA_MEASURE
    id: Mapped[int] = mapped_column(primary_key=True)
    full_name: Mapped[str]
    abbr: Mapped[str]


#########################################################
#
# ISD SQL Alchemy Tables
#
#########################################################


class ISD(SQLBase):
    __tablename__ = TableName.ISD
    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime]
    call_id: Mapped[int]
    measure_id: Mapped[int]
    val: Mapped[float]


class ISDMeasure(SQLBase):
    __tablename__ = TableName.ISD_MEASURE
    id: Mapped[int] = mapped_column(primary_key=True)
    abbr: Mapped[str]


class Airport(SQLBase):
    __tablename__ = TableName.AIRPORT
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    city: Mapped[str]
    state: Mapped[str]
    callsign: Mapped[str]
    usaf: Mapped[int]
    wban: Mapped[int]
    lat: Mapped[float]
    lng: Mapped[float]


###############################################################
#
#  Indexes
#
#################################################################

# Create indexes afterwards to allow bulk inserts without updating indexing.
eba_index = Index("eba_idx", EBA.ts, EBA.measure_id, EBA.source_id)
inter_index = Index("inter_idx", EBAInter.ts, EBAInter.source_id)
isd_index = Index("isd_idx", ISD.ts, ISD.measure_id, ISD.call_id)


def create_indexes(db: str):
    global engine
    if engine.url.database != db:
        return
    for index in [eba_index, inter_index, isd_index]:
        try:
            index.create(bind=engine, checkfirst=True)
        except Exception as e:
            print(e)


def drop_indexes(db):
    global engine
    if engine.url.database != db:
        return
    for index in [eba_index, inter_index, isd_index]:
        index.drop(bind=engine, checkfirst=True)


class EBAData:
    """Class for extracting metadata about EBA dataset and saving to SQL"""

    EBA_FILE = "EBA.txt"
    META_FILE = "metaseries.txt"
    ISO_NAME_FILE = "iso_name_file.json"

    def __init__(self, eba_path="/home/root/data/EBA/EBA20230302/"):
        self.eba_filename = os.path.join(eba_path, self.EBA_FILE)
        self.meta_file = os.path.join(eba_path, self.META_FILE)
        self.iso_file_map = os.path.join(eba_path, self.ISO_NAME_FILE)

    def extract_meta_data(self):
        # need checking on location and if file exists
        os.system(f"grep -r 'category_id' {self.eba_filename} > {self.meta_file}")

    def load_metadata(self) -> pd.DataFrame:
        meta_df = pd.read_json(self.meta_file, lines=True)
        return pd.DataFrame(meta_df)

    def parse_metadata(self, df: pd.DataFrame) -> Dict:
        """Grab names, abbreviations and category ids and save to dict"""
        iso_map = {}
        for _, row in df.iterrows():
            if "(" in row["name"]:
                tokens = re.findall(r"(\w+)", row["name"])
                name = " ".join(tokens[:-1])
                abbrv = tokens[-1]
                if abbrv == abbrv.upper():
                    iso_map[abbrv] = name

        return iso_map

    def save_iso_dict_json(self) -> str:
        """Load up meta data, extract names, save to json"""
        df = self.load_metadata()
        iso_map = self.parse_metadata(df)

        with open(self.iso_file_map, "w") as fp:
            json.dump(iso_map, fp)
            return self.iso_file_map

    @lru_cache()
    def load_iso_dict_json(self) -> Dict:
        with open(self.iso_file_map, "r") as fp:
            out_d = json.load(fp)
        return out_d

    @classmethod
    def get_name_abbr(cls, st):
        out_list = []
        for h in re.findall(r"for ([\w\s\,\.\-]+) \((\w+)\)", st):
            out_list.append((h[1], h[0]))
        for h in re.findall(r"to ([\w\s\,\.\-]+) \((\w+)\)", st):
            out_list.append((h[1], h[0]))
        return out_list

    @classmethod
    def find_dups(cls, data):
        found_data = set()
        for D in data:
            if D in found_data:
                print("dupe!", D)
            found_data.update([D])
        print(found_data)

    def populate_meta_tables(self):
        """Populate EBA metadata about ISOs and Measure abbreviations"""
        # iso_map = self.load_iso_dict_json()
        df = self.load_metadata()
        iso_map = self.parse_metadata(df)
        more_names = get_cls_attr_dict(EBAExtra)

        data_list = [
            {ColName.ABBR: abbr, ColName.FULL_NAME: full_name}
            for abbr, full_name in list(iso_map.items()) + list(more_names.items())
        ]
        with DBSession() as sess, sess.begin():
            sess.execute(insert(EBAMeta), data_list)

        eba_names = get_cls_attr_dict(EBAAbbr)
        data_list = [
            {ColName.ABBR: abbr, ColName.FULL_NAME: full_name}
            for abbr, full_name in eba_names.items()
        ]

        gen_names = get_cls_attr_dict(EBAGenAbbr)
        # hard-coded abbreviation munging
        data_list += [
            {ColName.ABBR: f"NG-{abbr}", ColName.FULL_NAME: full_name}
            for abbr, full_name in gen_names.items()
        ]
        with DBSession() as sess, sess.begin():
            sess.execute(insert(EBAMeasure), data_list)

    def parse_eba_series_id(self, str):
        sub = str.split(".")[1:]  # drop the EBA
        source, dest = sub[0].split("-")
        tag = "-".join(sub[1:-1])
        time = sub[-1]
        return source, dest, tag, time

    def read_metadata(self) -> List:
        """Read in name / series_ids."""
        # load files one by one.
        # if in desired type, then insert.
        # before insert, check if times exists.
        # otherwise update based on time overlap.
        out_list = []
        with jsonlines.open(self.eba_filename, "r") as fh:
            for dat in tqdm(fh):
                if not dat.get("series_id") and not dat.get("data"):
                    continue
                out_list.append((dat["name"], dat["series_id"]))
        return out_list

    def load_data(
        self,
        Nseries: int = -1,
        Ntime: int = -1,
        bulk: bool = True,
        update: bool = False,
        Ncommit: int = 50,
    ):
        """Load in relevant data series.  Only keep the ones on UTC time."""

        int_re = re.compile("Actual Net Interchange")
        utc_re = re.compile("UTC")

        file_count = 0
        with jsonlines.open(self.eba_filename, "r") as fh:
            for dat in tqdm(fh):
                name = dat["name"]
                if not dat.get("series_id") and not dat.get("data"):
                    continue
                if not utc_re.search(name):
                    continue
                if Nseries > 0 and file_count > Nseries:
                    break

                file_count += 1

                if int_re.search(name):
                    table = EBAInter
                    cols, unique_list, data_list = self._get_interchange_sql_inputs(
                        dat, Ntime=Ntime
                    )

                else:
                    table = EBA
                    cols, unique_list, data_list = self._get_regular_sql_inputs(
                        dat, Ntime=Ntime
                    )
                # TODO: Needs to be overhauled for SQLAlchemy
                self.sqldr.insert_data_column(
                    table_name=table_name,
                    col_list=cols,
                    data=data_list,
                    unique_list=unique_list,
                    val_col=ColName.VAL,
                    bulk=bulk,
                    update=update,
                )
                if file_count % Ncommit == 0 and bulk:
                    print("commiting")
                    self.sqldr.commit()
        self.sqldr.commit()

    def _get_interchange_sql_inputs(self, dat, Ntime=-1):
        # Overhaul for SQLAlchemy
        source, dest, tag, _ = self.parse_eba_series_id(dat["series_id"])
        # print(dat["name"], dat["series_id"], source, dest, tag)
        source_id = self.get_eba_source_id(source, dat["name"])
        dest_id = self.get_eba_source_id(dest, dat["name"])

        cols = [ColName.TS, ColName.SOURCE_ID, ColName.DEST_ID, ColName.VAL]
        unique_list = [ColName.TS, ColName.SOURCE_ID, ColName.DEST_ID]
        sub_data = dat["data"][:Ntime]
        data_list = [self.get_data_insert_str(x, source_id, dest_id) for x in sub_data]
        return cols, unique_list, data_list

    def _get_regular_sql_inputs(self, dat: Dict, Ntime: int = -1):
        # TODO: Overhaul for SQLAlchemy
        source, dest, tag, _ = self.parse_eba_series_id(dat["series_id"])
        # dest_id = self.get_eba_source_id(dest)
        # print(dat["name"], dat["series_id"], source, dest, tag)
        source_id = self.get_eba_source_id(source, dat["name"])
        measure_id = self.get_eba_measure_id(tag)

        cols = [ColName.TS, ColName.SOURCE_ID, ColName.MEASURE_ID, ColName.VAL]
        unique_list = [ColName.TS, ColName.SOURCE_ID, ColName.MEASURE_ID]
        sub_data = dat["data"][:Ntime]
        data_list = [
            self.get_data_insert_str(x, source_id, measure_id)
            # f"('{self.parse_time_str(x[0])}', {source_id}, {measure_id}, {x[1]})"
            for x in sub_data
        ]
        return cols, unique_list, data_list

    @classmethod
    def parse_time_str(cls, V):
        # convert time strings like : YYYYMMDDTHHZ to YYYY-MM-DDTHH:00:00
        return f"{V[:4]}-{V[4:6]}-{V[6:11]}:00:00"

    @classmethod
    def get_data_insert_str(cls, x, source_id, measure_id):
        # TODO: Probably not necessary for SQLALchemy?
        if x[1]:
            return f"('{cls.parse_time_str(x[0])}', {source_id}, {measure_id}, {x[1]})"
        else:
            return f"('{cls.parse_time_str(x[0])}', {source_id}, {measure_id}, NULL)"

    def get_eba_source_id(self, source, name, dpth=0):
        # TODO: Overhaul for SQLAlchemy
        qry_str = (
            f"SELECT id FROM {TableName.EBA_META} WHERE {ColName.ABBR} = '{source}';"
        )
        source_id = self.sqldr.get_data(qry_str)

        if dpth > 2:
            raise RuntimeError(f"EBA - No source found for {source_id}")

        if not source_id:
            # try to insert source,
            print(f"EBA - No source found for {source_id}.  Trying update")
            abbr_name_tups = self.get_name_abbr(name)
            self._populate_iso_meta(abbr_name_tups)
            return self.get_eba_source_id(source, name, dpth + 1)
        return source_id[0][0]

    def get_eba_measure_id(self, measure):
        # TODO: Update For SQLAlchemy
        measure_id = select(EBAMeasure.id).where(EBAMeasure.abbr == measure)
        return measure_id[0][0]


class ISDData:
    """Utils for getting Airport sensor data, setting up sql tables and loading data in."""

    NAME = "ISDData"

    def __init__(
        self,
        meta_file="air_merge_df.csv.gz",
        sign_file="air_signs.csv",
    ):
        self.meta_file = os.path.join(DATA_DIR, meta_file)
        self.sign_file = os.path.join(DATA_DIR, sign_file)
        self.ISD_TABLES = [TableName.ISD]
        self.ISD_MEASURES = [
            isddf.TEMP,
            isddf.WIND_SPD,
            isddf.PRECIP_1HR,
        ]

    def get_air_meta_df(self) -> pd.DataFrame:
        air_df = pd.read_csv(self.meta_file, index_col=0)
        return air_df

    def save_callsigns(self):
        df = self.get_air_meta_df()
        df.sort_values(["ST", "CALL"])["CALL"].to_csv(
            self.sign_file, header=True, index=False
        )

    @lru_cache()
    def load_callsigns(self) -> List:
        return pd.read_csv(self.sign_file)["CALL"].tolist()

    def populate_isd_meta(self):
        """Populate SQL table with Airport metadata from Pandas DF.  Also populate measure table"""

        print("Loading ISD Meta Data")
        air_df = self.get_air_meta_df()
        data_cols = ["name", "City", "ST", "CALL", "USAF", "WBAN", "LAT", "LON"]
        sub_data = air_df[data_cols].values.tolist()
        cols = [
            "name",
            "city",
            "state",
            "callsign",
            "usaf",
            "wban",
            "lat",
            "lng",
        ]
        data_list = [{cols[i]: ri for i, ri in enumerate(row)} for row in sub_data]
        with DBSession() as sess, sess.begin():
            sess.execute(insert(Airport), data_list)

    def populate_measures(self):
        # populate measure table
        data_list = [{ColName.ABBR: abbr} for abbr in isddf.ind_name_lookup.values()]
        with DBSession() as sess, sess.begin():
            sess.execute(insert(ISDMeasure), data_list)

    def get_isd_filenames(
        self, callsign_subset: Optional[Set[str]] = None
    ) -> List[Tuple[str]]:
        """Use ISD Meta table to build up known file list"""
        with DBSession() as sess, sess.begin():
            wban_usaf_list = sess.execute(
                select(Airport.usaf, Airport.wban, Airport.callsign).order_by(
                    Airport.callsign
                )
            )
        file_list = []
        for usaf, wban, callsign in wban_usaf_list:
            if callsign_subset and callsign not in callsign_subset:
                continue
            for year in YEARS:
                filename = get_local_isd_path(str(year), usaf, wban)
                if os.path.exists(filename) and os.path.getsize(filename) > 0:
                    file_list.append((filename, callsign))
        return file_list

    def load_data(
        self,
        callsign_subset=None,
        Nstation: int = -1,
        Ntime: int = -1,
        Ncommit: int = 50,
    ):
        """Load data for each station by year and insert desired data into columns of relevant tables.
        Tables / Columns governed by ISD_TABLES.  Converts all data timestamps appropriately to UTC.
        Each table has columns for each callsign.
        """
        files = self.get_isd_filenames(callsign_subset)[:Nstation]

        # out_sql = []

        for file_count, (file, callsign) in enumerate(tqdm(files)):
            # print(file, callsign)
            # df = load_isd_df(file)
            data_list = isddf.load_fwf_isd_file(file)
            callsign_id = self.get_callsign_id(callsign)
            for measure in self.ISD_MEASURES:
                measure_id = self.get_measure_id(measure)
                df_cols = [isddf.TIME, measure]
                sub_data = isddf.get_cols(df_cols, data_list)[:Ntime]

                data = [
                    self.get_data_insert_str(x, callsign_id, measure_id)
                    for x in sub_data
                ]
                cols = [ColName.TS, ColName.CALL_ID, ColName.MEASURE_ID, ColName.VAL]
                unique_list = [ColName.TS, ColName.CALL_ID, ColName.MEASURE_ID]

                self.sqldr.insert_data_column(
                    table_name=TableName.ISD,
                    col_list=cols,
                    data=data,
                    unique_list=unique_list,
                    val_col=ColName.VAL,
                    bulk=bulk,
                    update=update,
                )
            if file_count % Ncommit == 0 and bulk:
                self.sqldr.commit()
        self.sqldr.commit()

    def get_callsign_id(self, callsign: str) -> int:
        rv = DBSession.scalars(
            select(Airport.id).where(Airport.callsign == callsign)
        ).first()
        return rv

    def get_measure_id(self, measure: str) -> int:
        rv = DBSession.scalars(
            select(ISDMeasure.id).where(ISDMeasure.abbr == measure)
        ).first()
        return rv

    @classmethod
    def get_data_insert_str(cls, x, callsign, measure):
        if x[1]:
            return f"('{x[0]}', {callsign}, {measure}, {x[1]})"
        else:
            return f"('{x[0]}', {callsign}, {measure}, NULL)"
