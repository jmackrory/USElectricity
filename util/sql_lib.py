#Bunch of functions for creating new SQL database.
#
# create_new_database
# create_fresh_table
# check_and_create_columns

import psycopg2
import psycopg2.sql as sql

#Make new database by logging into 
def create_new_database(database_name):
	conn=psycopg2.connect(dbname='jonathan',host='localhost')
	#need elevated permissions in order to create a new database from within python, to automatically
	#commit changes.
	conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)  #automatically commit changes
	cur = conn.cursor()
	#create table
	t1 = sql.Identifier(database_name)
	q0 = sql.SQL("CREATE DATABASE {0}").format(t1)
	try:
		cur.execute(q0)
	except:
		print('Creating database: '+database_name+' failed.')
	cur.close()
	conn.close()
	return None

#make new connection and cursor for direct connection to SQL database via psycopg
def create_conn_and_cur(database_name,table_name):
	conn=psycopg2.connect(dbname=database_name,host='localhost')
	conn.set_session(autocommit=True)
	cur = conn.cursor()
	return conn, cur


#Create fresh table in a database by dropping the old one, and putting a new blank one in.
def create_fresh_table(database_name, table_name,init_column='init'):
	conn=psycopg2.connect(dbname=database_name,host='localhost')
	conn.set_session(autocommit=True)
	cur = conn.cursor()
	t1 = sql.Identifier(table_name)

	#Drop Table to start from scratch.
	try:
		q_drop = sql.SQL("DROP TABLE {0}").format(t1)
		cur.execute(q_drop)
		conn.commit()
		print('DROPPED '+table_name)
	except:
		print('Could not DROP '+table_name)

	#Create Blank table with first column given by first column of data.
	try:
		c1 = sql.Identifier(init_column)
		q_create = sql.SQL("CREATE TABLE {0} ({1} TEXT)").format(t1,c1)
		cur.execute(q_create)
		conn.commit()
		print('CREATED '+table_name)
	except:
		print('Could not CREATE TABLE '+table_name)

	return conn,cur

#Make sure required columns are present.
def check_and_create_columns(table_name,cur,df):

	for column_name in df.columns:
		t1 = sql.Identifier(table_name)
		c1 = sql.Identifier(column_name)
		try:
			#just check if there is a column with the right name.
			#allow no records since might have to start with empty table.
			q2 = sql.SQL("SELECT {1} FROM {0} LIMIT 0").format(t1,c1)
			cur.execute(q2)
			print('Success in retrieving column: '+column_name)
		except:
			print('Trying to read from column: '+column_name+' failed.')
			print('Trying to Add column: '+column_name)
			#if not then create that column.  
			q3 = sql.SQL("ALTER TABLE {0} ADD COLUMN {1} TEXT").format(t1,c1)
			cur.execute(q3)
	return None
