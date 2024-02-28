/*Create users with simple passwords as this is for local use only.
And separates testing and regular usage.
Will be more careful with  credentials elsewhere*/
CREATE USER dev_user PASSWORD 'dev_pw';
CREATE USER test_user PASSWORD 'test_pw';

CREATE DATABASE us_elec;
ALTER DATABASE us_elec OWNER to dev_user;
GRANT ALL PRIVILEGES ON DATABASE us_elec TO dev_user;
/*GRANT ALL PRIVILEGES ON SCHEMA public TO dev_user;*/

CREATE DATABASE test;
ALTER DATABASE test OWNER TO test_user;
GRANT ALL PRIVILEGES ON DATABASE test TO test_user;
/*GRANT ALL PRIVILEGES ON SCHEMA public TO test_user;*/