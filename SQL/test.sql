/*Basic query to check name, series ID formatting occurs  */
SELECT name, series_id, data FROM "ELEC" WHERE name LIKE '%Oregon%' LIMIT 1;

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

SELECT left('2014Q3',4)	