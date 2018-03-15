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
		)  sub_data


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

--SELECT * FROM ELEC_TEMP WHERE right(series_id,1)='M' LIMIT 100
