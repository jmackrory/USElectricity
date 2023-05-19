from argparse import ArgumentParser
from us_elec.SQL.sqldriver import EBAMeta


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--drop_tables", type=bool, default=False)
    args = parser.parse_args()

    ebm = EBAMeta()
    ebm.extract_meta_data()
    ebm.save_iso_dict_json()
    if args.drop_tables is True:
        ebm.drop_tables(execute=True)
    ebm.create_tables()
    ebm.load_data()
