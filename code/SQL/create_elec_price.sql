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