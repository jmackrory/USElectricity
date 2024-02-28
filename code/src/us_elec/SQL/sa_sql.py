"""
 Use SQLAlchemy to define and handle table creation rather than bodging together
 my own ORM/abstraction layers.

"""
import os
from typing import Dict, List, Optional

from datetime import datetime
from sqlalchemy import Index, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def get_engine():
    db, pw, user = get_creds()
    # us db, pw, user
    engine = create_engine("postgres+psycopg2://db:5432")
    return engine


def get_creds():
    db = os.environ.get("PG_DEV_DB", None)
    if not db:
        raise RuntimeError("SQLDriver could not find Postgres DB Name")

    pw = os.environ.get("PG_DEV_PASSWORD", "")
    if not pw:
        raise RuntimeError("SQLDriver could not find Postgres DB Password")
    user = os.environ.get("PG_DEV_USER", "")
    if not user:
        raise RuntimeError("SQLDriver could not find Postgres DB User")
    return db, pw, user


class SQLBase(DeclarativeBase):
    pass


class EBA(SQLBase):
    __tablename__ = "eba"
    ts = Mapped[datetime]
    source_id = Mapped[int]
    measure_id = Mapped[int]
    val = Mapped[float]


class Interchange(SQLBase):
    __tablename__ = "eba_inter"
    ts = Mapped[datetime]
    source_id = Mapped[int]
    dest_id = Mapped[int]
    val = Mapped[float]


class EBAMeta(SQLBase):
    __tablename__ = "eba_meta"
    id = Mapped[int] = mapped_column(primary_key=True)
    full_name = Mapped[str]
    abbr = Mapped[str]


class EBAMeasure(SQLBase):
    __tablename__ = "eba_measure"
    id = Mapped[int] = mapped_column(primary_key=True)
    full_name = Mapped[str]
    abbr = Mapped[str]


class ISD(SQLBase):
    __tablename__ = "isd"
    ts = Mapped[datetime]
    call_id = Mapped[int]
    measure_id = Mapped[int]
    val = Mapped[float]


class ISDMeta(SQLBase):
    __tablename__ = "isd_meta"
    id = Mapped[int] = mapped_column(primary_key=True)
    full_name = Mapped[str]
    abbr = Mapped[str]


class ISDMeasure(SQLBase):
    __tablename__ = "isd_measure"
    id = Mapped[int] = mapped_column(primary_key=True)
    full_name = Mapped[str]
    abbr = Mapped[str]


# Create indexes afterwards to allow bulk inserts without updating indexing.
eba_index = Index("eba_idx", ISD.ts, ISD.measure_id, ISD.call_id)
inter_index = Index("inter_idx", Interchange.ts, Interchange.source_id)
isd_index = Index("isd_idx", ISD.ts, ISD.measure_id, ISD.call_id)

# if __name__ == '__main__':
#    eba_index.create(bind=engine)
