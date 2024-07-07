from argparse import ArgumentParser

from us_elec.SQL.sqldriver import EBADriver, get_creds


def create_eba_tables_and_load_eba_data():
    parser = ArgumentParser()
    parser.add_argument(
        "--drop_tables", type=bool, default=False, help="Drops all EBA tables!"
    )
    parser.add_argument(
        "--Nseries", type=int, default=-1, help="Number of series to load in"
    )
    parser.add_argument(
        "--Ntime", type=int, default=-1, help="Number of times per series to load in"
    )
    args = parser.parse_args()

    ebm = EBADriver(get_creds())
    if args.drop_tables is True:
        print("Dropping ISD Tables!")
        ebm.drop_tables(execute=True)
        ebm.drop_indexes()

    print("Creating ISD Table")
    ebm.create_tables()

    print("Creating and Populating Meta Table")
    # ebm.extract_meta_data()
    # ebm.save_iso_dict_json()
    ebm.populate_meta_tables()

    print("Loading data")
    ebm.load_data(Nseries=args.Nseries, Ntime=args.Ntime)
    print("Creating Index")
    ebm.create_indexes()


if __name__ == "__main__":
    create_eba_tables_and_load_eba_data()
