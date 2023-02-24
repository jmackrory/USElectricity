# User Setup Notes
- Changelog and instructions on how to run the app.
- Including motivation and changes
- Includes daily changes.

## Running Docker setup
- change dir to root of project
build container:
"docker compose -f docker/docker-compose.yml build"

run container:
"docker compose -f docker/docker-compose.yml up"
- should have whole project folder mounted to dir.

shut it all down
"docker compose -f docker/docker-compose.yml down"

## Feb 23
- Got it to work by just mounting files into /tf directory.  Can create files and edit them now. 
- Can run JupyterLab?  Not sure it's worth the hassle if we can just use VSCode (which is far more comfortable for editting)

## Feb 15
Tried FastAPI container to try debugging what was up with networking.
Note that you need to use "up" not "run" to bring up the machine and networks
and actually run it.  

## Feb 14

Got jupyter running inside container with new command. 
Having issues accessing outside the container?  
Running like this:
"docker compose -f docker/docker-compose.yml run --service-ports tfjupyter"
hits issues?

Using command 
["/usr/local/bin/jupyter-lab", 
      "--ip", "0.0.0.0",
      "--port", "8888",
      "--no-browser",
      "--allow-root"]
is not working?  port doesn't seem to be there on host machine?

## Feb 11, 2022
- migrated uid/gid in docker-compose to use .env file.
- successfully got docker-compose to work for building TF, Postgres
Todo:
    - get mongo going
    - interact with sqldb
    - interact with mongo
    - get jupyter going
    - test tensorflow
    - move 


## Feb 2022 - Docker
- Migrating to Docker on local dev.  Useful tech, also decoupled from
annoyances of maintaining Tensorflow drivers and virtual environments etc.
- Will be using PostgresSQL and MongoDB Containers
Goals:
- Set up reproducible local docker environment
- Load data into local persistent storage for Postgres and Mongo
- Be able to practice SQL, Tensorflow and analysis
- Be able to convert notebooks into format hostable on github.