/*DROP TABLE play;
CREATE TABLE play (name text, data text);
--SELECT * FROM play

INSERT INTO play (name, data) VALUES ('ser_one', '[[''201507'', 60], [''201508'', 120], [''201511'',204]]') ;
INSERT INTO play (name, data) VALUES ('ser_two', '[[''2015Q3'', 20], [''2015Q4'', 20]') ;
INSERT INTO play (name, data) VALUES ('ser_two', '[[''201509'', 2000], [''201510'', 2400]]') ;

SELECT * FROM play*/
-- 
-- CREATE TABLE play_time (name text);
-- DO $$
-- DECLARE i;
-- BEGIN
-- 	FOR i in 1..12 LOOP
-- 		UPDATE TABLE play_time
-- 	END LOOP
-- END
-- $$


--one table for each state.  

--SELECT count(*) from "ELEC" WHERE name like '%: Arizona :%' AND series_id not like '%PLANT%' AND data IS NOT NULL;
SELECT * FROM "ELEC" WHERE data is NULL LIMIT 100