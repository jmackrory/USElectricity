from datetime import datetime


class DataSet:
    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        Nsample: int,
        test_split: float,
        filepath: str,
    ):
        pass

    def generate_dataset(self):
        pass

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
    def __init__(**kwargs):
        pass
