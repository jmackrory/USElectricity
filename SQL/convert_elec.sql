--make new time_series table.
--Takes around 50 sec for 1000 series, so 10 hrs.  No bueno.

DROP TABLE ELEC_TEMP;
--makes table
CREATE TABLE ELEC_TEMP	( series_id TEXT, regex_tmp TEXT[],
obs_date DATE, obs_val NUMERIC);

EXPLAIN ANALYZE INSERT INTO ELEC_TEMP (series_id, regex_tmp)
SELECT 	sub_data.series_id, 
	regexp_matches(sub_data.data, '''([0-9Q]+)'', ([0-9.]+)','g')
	FROM 	(SELECT series_id,data FROM "ELEC" 
			WHERE name like 'Net generation%' 
			   AND series_id not like '%.PLANT.%' 
			   AND data IS NOT NULL 
			   LIMIT 100	)  sub_data;
--SELECT obs_date_tmp[1] FROM ELEC_TEMP LIMIT 100;

--remove array {} symbols.
/*UPDATE ELEC_TEMP SET obs_date = btrim(obs_date, '{}');
UPDATE ELEC_TEMP SET obs_val=btrim(obs_val,'{}')*/

CREATE TABLE month_lookup (end_str TEXT, month_str TEXT)

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

UPDATE ELEC_TEMP SET obs_date=datestr_to_date(regex_tmp[1],right(series_id,1));
UPDATE ELEC_TEMP SET obs_val=to_number(regex_tmp[2],'99999999999.9999999');

/*Made an index for faster searching */
--CREATE INDEX ON "ELEC" (lower(name));
SELECT * FROM ELEC_TEMP LIMIT 1000