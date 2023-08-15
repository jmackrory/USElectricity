from datetime import datetime
from argparse import ArgumentParser
from us_elec.SQL.datasets import SimpleDataSet


def create_and_save_simple_data():
    parser = ArgumentParser()
    parser.add_argument(
        "--start_date", type=datetime, default=False, help="Starting date"
    )
    parser.add_argument("--end_date", type=datetime, default=False, help="Ending date")
    parser.add_argument("--Ndata", type=int, default=1000, help="Number of samples")
    parser.add_argument(
        "--test_split", type=float, default=0.1, help="Testing fraction"
    )

    args = parser.parse_args()

    sd = SimpleDataSet(**args)

    data = sd.generate_dataset()
    sd.save_dataset(data)


if __name__ == "__main__":
    create_and_save_simple_data()
