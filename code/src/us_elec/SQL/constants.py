from functools import lru_cache

DATA_DIR = "/home/root/data"
AIR_SIGN_PATH = "./meta/air_signs.csv"
EBA_NAME_PATH = "./meta/iso_names.csv"

YEARS = list(range(2015, 2024))


class TableType:
    EBA = "eba"
    NDFD = "ndfd"
    ISD = "isd"


class SQLVar:
    int = "integer"
    float = "float"
    str = "string"
    timestamptz = "timestamp with time zone"


class TableName:
    # note: must correspond to sqlalchemy base table names.
    EBA = "eba"
    INTERCHANGE = "eba_inter"
    EBA_META = "eba_meta"
    EBA_MEASURE = "eba_measure"
    ISD = "ISD"
    ISD_META = "isd_meta"
    ISD_MEASURE = "isd_measure"


class ColName:
    # Column names used across tables
    TS = "ts"
    ID = "id"
    SOURCE = "source"
    SOURCE_ID = "source_id"
    CALL = "callsign"
    CALL_ID = "callsign_id"
    MEASURE = "measure"
    MEASURE_ID = "measure_id"
    DEST = "dest"
    DEST_ID = "dest_id"
    VAL = "val"
    FULL_NAME = "full_name"
    ABBR = "abbr"


class EBAAbbr:
    # measure names for EBA
    NG = "Net Generation"
    ID = "Net Interchange"
    DF = "Demand Forecast"
    D = "Demand"
    TI = "Total Interchange"


class EBAGenAbbr:
    # measure names for subtypes of EBA generation
    COL = "Generation - Coal"
    WAT = "Generation - Hydro"
    NG = "Generation - Natural Gas"
    NUC = "Generation - Nuclear"
    OIL = "Generation - Oil"
    OTH = "Generation - Other"
    SUN = "Generation - Solar"
    TOT = "Generation - Total"
    WND = "Generation - Wind"


class EBAExtra:
    # measure names for other regions
    CAN = "Canada (region)"
    MEX = "Mexico (region)"


@lru_cache(1)
def get_air_names(fn=AIR_SIGN_PATH):
    with open(fn, "r") as fp:
        return fp.readlines()


@lru_cache(1)
def get_eba_names(fn=EBA_NAME_PATH):
    with open(fn, "r") as fp:
        return fp.readlines()
