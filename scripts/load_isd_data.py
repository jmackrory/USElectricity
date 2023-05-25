from argparse import ArgumentParser
from us_elec.SQL.sqldriver import ISDMeta


def create_isd_tables_and_load_isd_data():
    parser = ArgumentParser()
    parser.add_argument("--drop_tables", type=bool, default=False)
    parser.add_argument("--Nstation", type=int, default=-1)
    parser.add_argument("--Ntime", type=int, default=-1)
    args = parser.parse_args()

    isdm = ISDMeta()
    isdm.create_isd_meta()
    isdm.populate_isd_meta()
    if args.drop_tables is True:
        isdm.drop_tables()
    isdm.create_tables()
    isdm.create_indexes()
    isdm.load_data(Nstation=args.Nstation, Ntime=args.Ntime)


if __name__ == "__main__":
    create_isd_tables_and_load_isd_data()
