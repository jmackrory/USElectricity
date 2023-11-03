/*Create users with simple passwords as this is for local use only.
And separates testing and regular usage.
Will be more careful with  credentials elsewhere*/
CREATE USER dev_user PASSWORD 'dev_pw';
CREATE USER test_user PASSWORD 'test_pw';

CREATE DATABASE us_elec;
GRANT ALL PRIVILEGES ON DATABASE us_elec TO dev_user;

CREATE DATABASE test;
GRANT ALL PRIVILEGES ON DATABASE test TO test_user;