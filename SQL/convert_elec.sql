--make new time_series table.

--SELECT series_id FROM "ELEC" LIMIT 100
/*
DROP TABLE ELEC_TEMP;
--makes table
CREATE TABLE ELEC_TEMP 	( series_id TEXT, obs_date TEXT, obs_val TEXT);

INSERT INTO ELEC_TEMP (series_id, obs_date, obs_val)*/
SELECT 	sub_data.series_id, 
	regexp_matches(sub_data.data, '''([0-9Q]+)''','g'),
	regexp_matches(sub_data.data,', ([0-9.]+)','g') 
	FROM 
	(SELECT series_id,data FROM "ELEC" LIMIT 10
	)  sub_data;

--remove array {} symbols.
/*UPDATE ELEC_TEMP SET obs_date = btrim(obs_date, '{}');
UPDATE ELEC_TEMP SET obs_val=btrim(obs_val,'{}')*/

--ALTER TABLE ELEC_TEMP ADD year NUMERIC;
--ALTER TABLE ELEC_TEMP ADD month NUMERIC;

--UPDATE ELEC_TEMP SET year=to_number(left(obs_date,4),'9999');
--Convert date to month based on format.  
-- -- 
-- UPDATE ELEC_TEMP SET month=3*to_number(right(obs_date,1),'99') 
-- WHERE right(series_id,1)='Q';
-- 
-- UPDATE ELEC_TEMP SET month=to_number(right(obs_date,2),'99') 
-- WHERE right(series_id,1)='M';
-- 
-- UPDATE ELEC_TEMP SET month=12 
-- WHERE right(series_id,1)='A'; 


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


SELECT obs_date, obs_val FROM ELEC_TEMP GROUP BY series_id 