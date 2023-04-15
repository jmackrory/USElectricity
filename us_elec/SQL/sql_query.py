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

eba_index = ""

isd_index = ""
