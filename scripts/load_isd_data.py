from argparse import ArgumentParser
from us_elec.SQL.sqldriver import ISDMeta

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--drop_tables", type=bool, default=False)
    args = parser.parse_args()

    isdm = ISDMeta()
    isdm.create_isd_meta()
    isdm.populate_isd_meta()
    if args.drop_tables is True:
        isdm.drop_tables()
    isdm.create_tables()
    isdm.create_indexes()
    isdm.load_data()
