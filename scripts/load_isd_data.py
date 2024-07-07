# Timing note: Took 32 min single threaded for 600 stations, 3 variables, and 8 years (2015-2023)
# Need to update to new setup.

from argparse import ArgumentParser

from us_elec.SQL.sqldriver import ISDDriver, get_creds


def create_isd_tables_and_load_isd_data():
    parser = ArgumentParser()
    parser.add_argument("--drop_tables", type=bool, default=False)
    parser.add_argument("--Nstation", type=int, default=-1)
    parser.add_argument("--Ntime", type=int, default=-1)
    args = parser.parse_args()

    isdm = ISDDriver(get_creds())
    if args.drop_tables is True:
        print("Dropping ISD Tables!")
        isdm.drop_tables(execute=True)
        isdm.drop_indexes()
    print("Creating and Populating Meta Table")
    isdm.create_isd_meta()
    isdm.populate_isd_meta()
    isdm.populate_measures()
    print("Creating ISD Table")
    isdm.create_tables()
    isdm.load_data(Nstation=args.Nstation, Ntime=args.Ntime)
    print("Creating Index")
    isdm.create_indexes()


if __name__ == "__main__":
    create_isd_tables_and_load_isd_data()
