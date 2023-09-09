import random
from datetime import datetime

import pandas as pd

from us_elec.SQL.sqldriver import ISDDF, ColName, EBAAbbr, EBAName, ISDName, SQLDriver


class DataSet:
    def __init__(
        self,
        start_date: datetime = datetime(2015, 1, 1),
        end_date: datetime = datetime(2023, 6, 1),
        Nsample: int = 10000,
        test_split: float = 0.1,
        seed: int = 7653,
        filepath: str = "/tf/data/check.json",
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.seed = seed
        self.date_range = pd.date_range(start_date, end_date)
        self.Nsample = Nsample

    def get_time(self, t1, t2):
        return random.choice(self.date_range)

    def generate_dataset(self):
        random.seed(self.seed)

    def save_dataset(self):
        pass

    def load_dataset(self):
        pass


class Record:
    def get_record(self):
        """Select data from DB"""
        pass

    def save_record(self):
        """Save record to dataset"""
        pass

    def read_record(self):
        """Read record from dataset"""
        pass


class SimpleDataSet(DataSet, Record):
    def __init__(self, **kwargs):
        self.sqldr = SQLDriver()

    def get_eba_record(self, meas, t1, t2):
        sql = f"""
        SELECT e.{ColName.TS}, im.{ColName.FULL_NAME}, e.{ColName.VAL}, me.{ColName.ABBR} FROM {EBAName.EBA} AS e
        INNER JOIN {EBAName.ISO_META} AS im ON {ColName.SOURCE_ID} = im.{ColName.ID}
        INNER JOIN {EBAName.MEASURE} AS me ON {ColName.MEASURE_ID} = me.{ColName.ID}
        WHERE (
            (e.{ColName.TS} > '{t1}') AND
            (e.{ColName.TS} <= '{t2}') AND
            (me.{ColName.ABBR} = {meas})
            )
        ORDER BY (im.{ColName.FULL_NAME}, e.{ColName.TS})
        """
        rv = self.sqldr.get_data(sql)
        # convert to DF, pivot.
        return rv

    def get_isd_record(self, meas, t1, t2):
        sql = f"""
        SELECT isd.{ColName.TS}, am.{ColName.CALL}, isd.{ColName.VAL} FROM {ISDName.ISD} AS isd
        INNER JOIN {ISDName.AIR_META} AS am ON {ColName.CALL_ID} = am.{ColName.ID}
        INNER JOIN {ISDName.MEASURE} AS me ON {ColName.MEASURE_ID} = me.{ColName.ID}
        WHERE (
            (isd.{ColName.TS} > '{t1}') AND
            (isd.{ColName.TS} <= '{t2}') AND
            (me.{ColName.MEASURE} = {meas})
            )
        """
        rv = self.sqldr.get_data(sql)
        return rv

    def get_recs(self, t1, t2, t3):
        """ """
        ts1 = self.get_datestr(t1)
        ts2 = self.get_datestr(t2)
        # ts3 = self.get_datestr(t3)
        temp_dat = self.get_isd_record(ISDDF.TEMPERATURE, ts1, ts2)
        demand_dat = self.get_eba_record(EBAAbbr.D, ts1, ts2)
        demand_forecast_dat = self.get_eba_record(EBAAbbr.DF, ts1, ts2)
        return temp_dat, demand_dat, demand_forecast_dat

    # get Demand, get forecast, get temperature

    def get_datestr(self, t):
        return t.astimezone.replace(microsecond=0).isoformat()
