# SQL Templat commands for use with psycopg2
# Used for local operation on trusted data, so can be a bit loose.

ALLOWED_TYPES = ["integer", "float"]

air_table_create_template = """
 CREATE TABLE %s IF NOT EXISTS
;
"""

air_meta_create = """
   CREATE TABLE air_meta IF NOT EXISTS
   id integer,
   name varchar(100),
   city carchar(100),
   state char(2)
   callsign char(4)
   usaf integer,
   wban integer,
   lat float,
   long float;
"""


eba_insert_template = """
   BULK INSERT %s INTO %s
"""

isd_table_template = """
 CREATE TABLE %s IF NOT EXISTS
;
"""

eba_index = ''

isd_index = ''

def get_create_eba_table_sql(table_name, var_type):
    if var_type not in ALLOWED_TYPES:
        raise RuntimeError(f"{var_type} not in {ALLOWED_TYPES}!")

    str_list = [
        f"CREATE TABLE {table_name} IF NOT EXISTS",
        "id integer,",
        "ts timestamp,"]
    eba_names = EBAMeta().load_iso_dict_json().keys()
    str_list += [f"{eba} {var_type}," for eba in eba_names]
    return " ".join(str_list)


def get_create_air_table_sql(table_name, var_type):
    if var_type not in ALLOWED_TYPES:
        raise RuntimeError(f"{var_type} not in {ALLOWED_TYPES}!")
    str_list = [
        f"CREATE TABLE {table_name} IF NOT EXISTS",
        "id integer,",
        "ts timestamp,"]
    call_signs = AirMeta().load_callsigns()
    str_list += [f"{eba} {var_type}," for cs in call_signs]
    return " ".join(str_list)
