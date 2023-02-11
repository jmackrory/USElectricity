# User Setup Notes
- Changelog and instructions on how to run the app.
- Including motivation and changes
- Includes daily changes.

## Running Docker setup
- change dir to root of project
build container:
"docker compose -f docker/docker-compose.yml build tfjupyter"

run container:
"docker compose -f docker/docker-compose.yml run tfjupyter"
- should have whole project folder mounted to dir.

## Feb 11, 2022
- migrated uid/gid in docker-compose to use .env file.
- successfully got docker-compose to work for building TF, Postgres
Todo:
    - get mongo going
    - interact with sqldb
    - interact with mongo
    - get jupyter going
    - test tensorflow


## Feb 2022 - Docker
- Migrating to Docker on local dev.  Useful tech, also decoupled from
annoyances of maintaining Tensorflow drivers and virtual environments etc.
- Will be using PostgresSQL and MongoDB Containers
Goals:
- Set up reproducible local docker environment
- Load data into local persistent storage for Postgres and Mongo
- Be able to practice SQL, Tensorflow and analysis
- Be able to convert notebooks into format hostable on github.