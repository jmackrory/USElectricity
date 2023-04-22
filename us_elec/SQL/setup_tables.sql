/*  Old SQL queries of dubious value.

Mostly used for loading ELEC data into queryable form.
Mostly about creating tables and loading some data?  Need to retrace steps on if it is helpful

*/


/*Create a base schema common for all ISOs for EBA data.
id - Actual Net Interchange
d - demand
ti - total interchange
ng - Net Generation

ng sub varieties:
ng - Natural Gas
wat - Hydro
sun - Solar
col - Coal
wnd - Wind
oth - other (biomass etc
oil - Oil
nuc - Nuclear

- hl local time
- h UTC

*/

CREATE SCHEMA IF NOT EXISTS iso
       CREATE TABLE id_hl (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE id_h (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE d_h (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE d_hl (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_h (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_hl (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ti_h (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ti_hl (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE df_h (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE df_hl (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_ng_h (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_ng_hl (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_wat_h (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_wat_hl (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_sun_h (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_sun_hl (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_col_h (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_col_hl (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_oth_h (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_oth_hl (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_wnd_h (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_wnd_hl (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_oil_h (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_oil_hl (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_nuc_h (dt TIMESTAMP WITH TIME ZONE, val real)
       CREATE TABLE ng_nuc_hl (dt TIMESTAMP WITH TIME ZONE, val real)

/* Customer Data for General ELEC tables */
CREATE TABLE ELEC_CUSTOMERS	( series_id TEXT, regex_tmp TEXT[],
obs_date DATE, obs_val NUMERIC);

INSERT INTO ELEC_CUSTOMERS (series_id, regex_tmp)
SELECT 	sub_data.series_id, 
	regexp_matches(sub_data.data, '''([0-9Q]+)'', ([0-9.]+)','g')
	FROM 	(SELECT series_id,data FROM "ELEC" 
			WHERE series_id like '%.CUSTOMERS.%' 
			   AND series_id not like '%.PLANT.%' 
			   AND data IS NOT NULL )  sub_data;

UPDATE ELEC_CUSTOMERS SET obs_date=datestr_to_date(regex_tmp[1], right(series_id,1)),
		     obs_val=to_number(regex_tmp[2],'99999999999.9999999');

/*Made an index for faster searching */
SELECT * FROM ELEC_CUSTOMERS LIMIT 1000

--make new time_series table.

/* ELEC_GEN

Net Generation by state / time.
After shoving all ELEC data into table.  
*/
CREATE TABLE ELEC_GEN ( series_id TEXT, regex_tmp TEXT[], obs_date DATE, obs_val NUMERIC);

INSERT INTO ELEC_GEN (series_id, regex_tmp)
SELECT sub_data.series_id, 
	regexp_matches(sub_data.data, '''([0-9Q]+)'', ([0-9.]+)','g')
	FROM 	(SELECT series_id, data FROM "ELEC" 
			WHERE name like 'Net generation%' 
			   AND series_id not like '%.PLANT.%' 
			   AND data IS NOT NULL 
			   LIMIT 100000)  sub_data;
-- 
-- /*Return the final quarter for string of final day on each quarter, given the quarter string*/
-- CREATE OR REPLACE FUNCTION month_to_monthday(IN m_str TEXT) 
-- 	RETURNS TEXT AS $$
-- 	SELECT CASE 
-- 		WHEN 	m_str='01' OR m_str='03' OR m_str='05' OR 
-- 			m_str='07' OR m_str='08' OR m_str='10' OR 
-- 			m_str='12' THEN m_str ||'31' 
-- 		WHEN m_str='04' OR m_str='06' OR m_str='09' OR m_str= '11' THEN m_str ||'30' 
-- 		WHEN m_str='02' THEN m_str || '28'   
-- 		END;-- AS monthday_str;
-- 	$$ LANGUAGE SQL;
-- 
-- 
-- /*Return the final quarter for string of final day on each quarter, given the quarter string*/
-- CREATE OR REPLACE FUNCTION quarter_to_monthday(IN quarter_str TEXT) 
-- 	RETURNS TEXT AS $$
-- 	SELECT CASE quarter_str
-- 		WHEN 'Q1' THEN '0331' 
-- 		WHEN 'Q2' THEN '0630'   
-- 		WHEN 'Q3' THEN '0930'
-- 		WHEN 'Q4' THEN '1231' 
-- 		END; --AS qtr_str;
-- 	$$ LANGUAGE SQL;


/*-- CREATE TABLE month_lookup (end_str TEXT, monthday_str TEXT);
-- 
-- INSERT INTO month_lookup (end_str, monthday_str) VALUES
-- ('01','0131'), ('02','0228'),('03','0331'),('04','0430'),('05','0531'),('06','0630'),
-- ('07','0731'), ('08','0831'),('09','0930'),('10','1031'),('11','1130'),('12','1231'),
-- ('Q1','0331'), ('Q2','0630'),('Q3','0930'),('Q4','1231');
/*
/*Looks up appropriate end day for each month/quarter. 
Uses month_lookup table.*/
CREATE OR REPLACE FUNCTION QMdatestr_to_date(IN date_str TEXT)
RETURNS DATE AS $$
	SELECT to_date(
		left(date_str,4) || (SELECT monthday_str FROM month_lookup 
		  WHERE end_str=right(date_str,2) ),
		  'YYYYMMDD');
$$ LANGUAGE SQL;

/*Creates dates based on date str, and frequency*/
CREATE OR REPLACE FUNCTION datestr_to_date(IN date_str TEXT, IN freq TEXT)
RETURNS DATE AS $$
	SELECT to_date(date_conv.new_str,'YYYYMMDD') FROM 
	(
	SELECT CASE freq
		WHEN 'Q' THEN left(date_str,4) || quarter_to_monthday(right(date_str,2))
		WHEN 'M' THEN left(date_str,4) || month_to_monthday(right(date_str,2))
		WHEN 'A' THEN date_str || '1231'
		END AS new_str
	) AS date_conv;
$$ LANGUAGE SQL;
-- 
-- UPDATE ELEC_TEMP SET obs_date=QMdatestr_to_date(regex_tmp[1]) WHERE right(series_id,1) LIKE 'Q' OR right(series_id,1)='M';
-- UPDATE ELEC_TEMP SET obs_date=to_date(regex_tmp[1] || '1231','YYYYMMDD') WHERE right(series_id,1) ='A';

UPDATE ELEC_GEN SET obs_date=datestr_to_date(regex_tmp[1], right(series_id,1)),
		     obs_val=to_number(regex_tmp[2],'99999999999.9999999');

/*Made an index for faster searching */
SELECT * FROM ELEC_GEN LIMIT 1000

/* ELEC - Revenue 

*/

CREATE TABLE ELEC_REV	( series_id TEXT, regex_tmp TEXT[],
obs_date DATE, obs_val NUMERIC);

INSERT INTO ELEC_REV (series_id, regex_tmp)
SELECT 	sub_data.series_id, 
	regexp_matches(sub_data.data, '''([0-9Q]+)'', ([0-9.]+)','g')
	FROM 	(SELECT series_id,data FROM "ELEC" 
			WHERE series_id like '%.REV.%' 
			   AND series_id not like '%.PLANT.%' 
			   AND data IS NOT NULL )  sub_data;

UPDATE ELEC_REV SET obs_date=datestr_to_date(regex_tmp[1], right(series_id,1)),
		     obs_val=to_number(regex_tmp[2],'99999999999.9999999');

/*Made an index for faster searching */
SELECT * FROM ELEC_REV LIMIT 1000


/* Electricity Price

*/
CREATE TABLE ELEC_PRICE	( series_id TEXT, regex_tmp TEXT[],
obs_date DATE, obs_val NUMERIC);

INSERT INTO ELEC_PRICE (series_id, regex_tmp)
SELECT 	sub_data.series_id, 
	regexp_matches(sub_data.data, '''([0-9Q]+)'', ([0-9.]+)','g')
	FROM 	(SELECT series_id,data FROM "ELEC" 
			WHERE series_id like '%.PRICE.%' 
			   AND series_id not like '%.PLANT.%' 
			   AND data IS NOT NULL )  sub_data;

UPDATE ELEC_PRICE SET obs_date=datestr_to_date(regex_tmp[1], right(series_id,1)),
		     obs_val=to_number(regex_tmp[2],'99999999999.9999999');

/*Made an index for faster searching */
SELECT * FROM ELEC_PRICE LIMIT 1000