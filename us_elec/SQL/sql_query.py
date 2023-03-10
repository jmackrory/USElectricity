# SQL Templat commands for use with psycopg2
# Used for local operation on trusted data, so can be a bit loose.

temp_table_template = """
 CREATE TABLE %s IF NOT EXISTS
;
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
