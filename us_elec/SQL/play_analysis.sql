--stupid frankenstein join by source
-- SELECT wnd.obs_date, nuc.obs_val as nuc_val, wnd.obs_val as wnd_val FROM 
-- (SELECT obs_date,obs_val FROM ELEC_GEN WHERE series_id LIKE '%WND-WA-1%M' ORDER BY obs_date desc) as wnd
--  INNER JOIN (SELECT obs_date,obs_val FROM ELEC_GEN WHERE series_id LIKE '%NUC-WA-1%M' ORDER BY obs_date desc) as nuc
--  ON wnd.obs_date=nuc.obs_date;
--CREATE INDEX ON ELEC_GEN (series_id);

CREATE OR REPLACE FUNCTION extract_timeseries( series_pattern TEXT, val_name TEXT)
RETURNS table(obs_date date,obs_val numeric) AS 
$$ 
SELECT obs_date,obs_val as val_name FROM ELEC_GEN WHERE series_id LIKE series_pattern ORDER BY obs_date desc;
$$ LANGUAGE SQL;

SELECT * FROM extract_timeseries('%WND-OR-1%','wnd');
/*Queries:
Largest generation?
Most Customers?
Fraction of intermittent renewables?

*/