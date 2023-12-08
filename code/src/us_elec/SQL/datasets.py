import random
from datetime import datetime
from typing import NamedTuple, Optional
import numpy as np

import pandas as pd

from us_elec.SQL.sqldriver import ISDDF, ColName, EBAAbbr, EBAName, ISDName, SQLDriver


class Key:
    T = "t"
    DEM = "dem"
    DEM_FORE = "dem_fore"
    TEMP = "temp"
    TEMP_FORE = "temp_fore"


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


class OldRecord:
    """OldRecord(NamedTuple)

    Tuple containing numpy arrays with:
    t - time
    dem - demand array
    temp - temperature array
    dem_fore - demand forecast array
    """

    def __init__(
        self, t: np.ndarray, dem: np.ndarray, temp: np.ndarray, dem_fore: np.ndarray
    ):
        self.t = t
        self.dem = dem
        self.temp = temp
        self.dem_fore = dem_fore

    def to_json(self):
        D = {
            "t": self.t.tolist(),
            "dem": self.dem.tolist(),
            "temp": self.temp.tolist(),
            "dem_fore": self.dem_fore.tolist(),
        }
        return D

    def from_json(str):
        pass


# TODO: figure out json serializaiton / deserialization.

# TODO: figure out current ds type hints (numpy, matplotlib, pandas, sklearn)
# current use of data-science-types are ooollllddd.


class SimpleDataSet(DataSet):
    def __init__(self, **kwargs):
        self.sqldr = SQLDriver()
        # So, saved using variable names of classes as labels in SQL
        # so need the name of variable as string to look up results.
        # self.ed = get_reverse_cls_attr_dict(EBAAbbr)
        # self.id = get_reverse_cls_attr_dict(ISDDF)

    def get_eba_record(self, meas: str, t1: str, t2: str):
        sql = f"""
        SELECT e.{ColName.TS}, im.{ColName.ABBR}, e.{ColName.VAL} FROM {EBAName.EBA} AS e
        INNER JOIN {EBAName.ISO_META} AS im ON {ColName.SOURCE_ID} = im.{ColName.ID}
        INNER JOIN {EBAName.MEASURE} AS me ON {ColName.MEASURE_ID} = me.{ColName.ID}
        WHERE (
            (e.{ColName.TS} > '{t1}') AND
            (e.{ColName.TS} <= '{t2}') AND
            (me.{ColName.FULL_NAME} = '{meas}')
            )
        ORDER BY (im.{ColName.FULL_NAME}, e.{ColName.TS});
        """
        rv = self.sqldr.get_data(sql)
        df = pd.DataFrame(
            rv, columns=[ColName.TS, ColName.ABBR, ColName.VAL]
        ).drop_duplicates()
        df = df.pivot_table(
            index=ColName.TS, columns=[ColName.ABBR], values=ColName.VAL
        )
        # JM comment: this runs into issues with potential numbers/order of stations changing over time periods.
        # Options: write out all stations. Its around 50-200 for EBA/ISD data.

        # convert to DF, pivot.
        return df

    def get_isd_record(self, meas, t1, t2):
        sql = f"""
        SELECT isd.{ColName.TS}, am.{ColName.CALL}, isd.{ColName.VAL} FROM {ISDName.ISD} AS isd
        INNER JOIN {ISDName.AIR_META} AS am ON {ColName.CALL_ID} = am.{ColName.ID}
        INNER JOIN {ISDName.MEASURE} AS me ON {ColName.MEASURE_ID} = me.{ColName.ID}
        WHERE (
            (isd.{ColName.TS} > '{t1}') AND
            (isd.{ColName.TS} <= '{t2}') AND
            (me.{ColName.MEASURE} = '{meas}')
            )
        ORDER BY (am.{ColName.CALL}, isd.{ColName.TS});
        """
        rv = self.sqldr.get_data(sql)
        df = pd.DataFrame(
            rv, columns=[ColName.TS, ColName.CALL, ColName.VAL]
        ).drop_duplicates()
        df = df.pivot_table(
            index=ColName.TS, columns=[ColName.CALL], values=ColName.VAL
        )

        return df

    def get_temp_forecast(self, t1):
        path = self._get_forecast(t1, ftype)
        # get relevant forecast
        # load forecast
        # extract relevant pieces

    def get_recs(self, t1, t2, t3):
        """
        Get previous demand / temperature data, and current temperature data
        to forecast for current demand.
        """
        ts1 = self.get_datestr(t1)
        ts2 = self.get_datestr(t2)
        ts3 = self.get_datestr(t3)
        temp_dat_A = self.get_isd_record(ISDDF.TEMPERATURE, ts1, ts2)
        demand_dat_A = self.get_eba_record(EBAAbbr.D, ts1, ts2)
        demand_fore_A = self.get_eba_record(EBAAbbr.DF, ts1, ts2)
        temp_dat_B = self.get_isd_record(ISDDF.TEMPERATURE, ts2, ts3)
        demand_dat_B = self.get_eba_record(EBAAbbr.D, ts2, ts3)
        demand_fore_B = self.get_eba_record(EBAAbbr.DF, ts2, ts3)

        # should check all columns are in expected ordered.
        # All values are present and aligned.
        input_rec = OldRecord(
            t=temp_dat_A.ts,
            dem=demand_dat_A.values,
            temp=temp_dat_A.values,
            dem_fore=demand_fore_A.values,
        )
        target_rec = OldRecord(
            t=demand_dat_B.index,
            dem=demand_dat_B.values,
            temp=temp_dat_B.values,
            dem_fore=demand_fore_B.values,
        )

        return input_rec, target_rec

    # get Demand, get forecast, get temperature

    def get_datestr(self, t):
        return t.astimezone.replace(microsecond=0).isoformat()
