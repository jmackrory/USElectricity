/*Basic query to check name, series ID formatting occurs  */
--SELECT name, series_id, data FROM "ELEC" WHERE name LIKE '%Oregon%' LIMIT 1;

/* name/series IDs seem to have multiple forms.  For generation it varies at summary and plant level.

name Format seems to be: 
series type: Plant name (plant_number): fuel/generation type : frequency
which matches up with a series_id:
ELEC . series type id . plant_number - fuel/generation id . frequency

For state level it is 
series type: fuel type : state : sector/producer : frequency

*/

/* Try using regex to grab all unique top level series names */
--SELECT DISTINCT regexp_matches(name, '^[\w\s]+:', 'g') FROM "ELEC" LIMIT 100   


--find earliest date
/*
--select start date
--find first 4 characters - convert to number.
--find minimum
--min date is 2001
SELECT min(to_number(substr(start,0,5),'0000')) FROM "ELEC";
--max date is 2016
SELECT max(to_number(substr(start,0,5),'0000')) FROM "ELEC";
*/
/*Now create new table for data which has format [['date',val], ['date2',val2],..]
*/
--splits data at "], [" boundaries.  (will need to remove [[ and ]])
--SELECT regexp_split_to_array(data, '\],(\s+)\[') FROM "ELEC" LIMIT 1000

--selects out each records name/series id.
/*SELECT 	sub_data.series_id, regexp_matches(sub_data.data, '''([0-9Q]+)''','g'), regexp_matches(sub_data.data,', ([0-9,]+)','g') 
	FROM 
	(SELECT series_id,data FROM "ELEC" LIMIT 1000
	)  sub_data*/

-- returns an array of pairs
-- SELECT 	sub_data.series_id,regexp_matches(sub_data.data, '\[''([0-9Q]+)'', ([0-9.]+)\]','g')
-- 	FROM 
-- 	(SELECT series_id,data FROM "ELEC" LIMIT 1000
-- 	)  sub_data;
-- 
/* DOn't like this printing, wanted a way to try converting an entire array at once.  */
-- DROP FUNCTION print_entries(text[]);
-- CREATE FUNCTION print_entries(IN input_array TEXT[])
-- 	RETURNS TABLE (year TEXT, val TEXT) AS $$
-- 	SELECT left(input_array[1],4), input_array[2]
-- 	$$ LANGUAGE SQL; 
-- 	
-- SELECT print_entries('{2010Q1, 450}'); 

--SELECT series_id FROM "ELEC" WHERE series_id='%.PLANT.%' LIMIT 100

/*Return the final quarter for string of final day on each quarter, given the quarter string*/
CREATE OR REPLACE FUNCTION month_to_monthday(IN m_str TEXT) 
	RETURNS TEXT AS $$
	SELECT CASE 
		WHEN 	m_str='01' OR m_str='03' OR m_str='05' OR 
			m_str='07' OR m_str='08' OR m_str='10' OR 
			m_str='12' THEN m_str ||'31' 
		WHEN m_str='04' OR m_str='06' OR m_str='09' OR m_str= '11' THEN m_str ||'30' 
		WHEN m_str='02' THEN m_str || '28'   
		END;-- AS monthday_str;
	$$ LANGUAGE SQL;


/*Return the final quarter for string of final day on each quarter, given the quarter string*/
CREATE OR REPLACE FUNCTION quarter_to_monthday(IN quarter_str TEXT) 
	RETURNS TEXT AS $$
	SELECT CASE quarter_str
		WHEN 'Q1' THEN '0331' 
		WHEN 'Q2' THEN '0630'   
		WHEN 'Q3' THEN '0930'
		WHEN 'Q4' THEN '1231' 
		END; --AS qtr_str;
	$$ LANGUAGE SQL;

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

-- /*working test query to convert quarter to monthly date. */
-- SELECT to_date(date_conv.new_str, 'YYYYmm') FROM
--  (SELECT left('2014Q2',4) || quarter_to_month(right('2014Q2',2)) 
--  AS new_str) AS date_conv;

--SELECT datestr_to_date('2015Q2','Q')

/*Make a new table with a list of the categories.  
Will create separate time-pivoted tables for each of these categories.

*/
--CREATE TABLE categories (cat text[]);
	--INSERT INTO categories (cat) SELECT DISTINCT regexp_matches(series_id, '^ELEC\.([\w]+)\.', 'g') FROM "ELEC";

--So around 550k series in total.  
SELECT name FROM "ELEC" WHERE DATA IS NULL;

/*SELECT 	sub_data.series_id, 
	regexp_matches(sub_data.data, '\[''([0-9Q]+)''','g'),
	regexp_matches(sub_data.data, '([0-9.]+)\]','g')
	FROM 
	(SELECT series_id, data FROM "ELEC" LIMIT 10
	)  sub_data;
*/-- 
-- SELECT tab.splt[1:2] FROM (
-- SELECT 	sub_data.series_id AS id, 
-- 	regexp_split_to_array(sub_data.data, '\], \[') AS splt
-- 	FROM 
-- 	(SELECT series_id,data FROM "ELEC" LIMIT 10
-- 	)  sub_data
-- )tab;
-- 


/* Working "for" loop */
-- DO $$
-- DECLARE r record;
-- BEGIN
-- FOR r IN SELECT data, series_id,f FROM "ELEC" LIMIT 10 LOOP
--  	SELECT r.data;
-- END LOOP;
-- END;
-- $$