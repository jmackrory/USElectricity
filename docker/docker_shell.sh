#/bin/sh
# Redundant.
# expect to run from US-Electricity root
# Utility functions for running docker stuff from outside terminal

CONTAINER_NAME=tfjupyter-gpu
COMPOSE_YML=docker/docker-compose.gpu.yml
DOCKERFILE=Dockerfile.tfjupyter

# attempt at exporting user and group id to docker
#export DOCKER_UID=$(id -u)
#export DOCKER_GID=$(id -g)

exec_tf_container(){
    # may need to export these and set as env variables
    docker compose -f $COMPOSE_YML exec $CONTAINER_NAME /bin/bash
}

build_joint_container(){
    docker compose -f $COMPOSE_YML build
}

run_joint_container(){
    docker compose -f $COMPOSE_YML up
}

down_joint_container(){
    docker compose -f $COMPOSE_YML down
}
