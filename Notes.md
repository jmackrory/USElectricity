# User Setup Notes
- Changelog and instructions on how to run the app.
- Including motivation and changes
- Includes daily changes.

## Motivation for Docker
- Maintaining the tensorflow environment was a pain, and finicky.  They now suggest using Docker.
And Working with Docker makes it more portable,  and reproduceable, albeit at the cost of working
with Docker which has it's own finickyness.  This also helps with the databases, since there is
less concern about affecting other DBs on the personal machine.


## Running Docker setup
- change dir to root of project
build container:
`docker compose -f docker/docker-compose.yml build`

run container:
`docker compose -f docker/docker-compose.yml up`
- should have whole project folder mounted to dir.

shut it all down
`docker compose -f docker/docker-compose.yml down`

Connect by logging into to `http://localhost:8890` and providing
Password.

For GPU support change `docker-compose.yml` to
`docker-compose.gpu.yml`
May also need to change `.env` `TF_IMAGE` to use a `-gpu-jupyter` image.

To run shell:
`docker compose -f docker/docker-compose.yml run tfjupyter -u $(id -u):$(id -g) -it /bin/bash`

Note: gpu version in `docker-compose.gpu.yml` with container `tfjupyter-gpu`

#### VS Code and Jupyer
- Best to have the Jupyter Dockerfile running before trying to use the VSCode jupyter notebook as it tries to login immediately if a notebook was previously open.
  (Otherwise it will fail to login, and you have to force it to log-in again.)
- Use Jupyter plugin to log in to remote `localhost:8890`.  Note that it may be necessary to
Clear the Remote Server List.
- It's also important to make sure you select the Docker container environment kernel, as otherwise it will run in the local environment.

### Emacs and Jupyter - Emacs IPython Notebook (EIN)

Allows usage of Jupyter from within Emacs.  Useful when VSCode acting up, or browser shortcuts are annoying for code editing.

- `Ctrl-c Ctrl-x`
- `ein: notebooklist-login`
- Provide port 8890, then password.

(VSCode is increasingly annoying to work with.
  - extend emacs with EIN for notebooks
  - direx for view
  - jedi for language server (auto-complete and navigation)

### Aug 2 - Docker + GPU + Ubuntu

Docker woes.  Docker not picking up the GPU anymore.
Uninstalled and reinstalled Docker via apt.

TF container suggests running Docker image in rootless mode.
https://docs.docker.com/engine/security/rootless/

bash
```
sudo apt-get install -y docker-ce-rootless-extras
dockerd-rootless-setuptool.sh install
```

Automatically start on login.
bash
```
 systemctl --user enable docker
 sudo loginctl enable-linger $(whoami)
```

Delete the line with credsStore from ~/.docker/config.json.


- Can test the GPU a couple ways.
1. Just with nvidia images.  This needs the docker toolkit installed.
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
Can test with:
Nvidia uses:
`
sudo docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
`
Followed advice here: https://github.com/nvidia/nvidia-container-runtime#docker-engine-setup to setup runtime
- Edited /etc/docker/daemon.conf to:
```
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    },
    "default-runtime": "nvidia"
}
```

Also did:
```
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo tee /etc/systemd/system/docker.service.d/override.conf <<EOF
[Service]
ExecStart=
ExecStart=/usr/bin/dockerd --host=fd:// --add-runtime=nvidia=/usr/bin/nvidia-container-runtime
EOF
sudo systemctl daemon-reload
sudo systemctl restart docker
```

2. Check TF Docker container can find the GPU.
https://www.tensorflow.org/install/docker

Running:
```
docker run --gpus all -it --rm tensorflow/tensorflow:latest-gpu \
   python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))
```
And
```
docker run  --gpus all -it --rm tensorflow/tensorflow:latest-gpu \
  python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
So that seems to work at finding a GPU. BUT! BUT!
```
docker run  --gpus all -it --rm tensorflow/tensorflow:latest-gpu-jupyter \
  python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
Fails to find a GPU.  FML.

But using earlier versions of tensorflow this works in both versions.  Tried 2.4.0 it worked.  2.12.0 it fails.
2.11.0 works.  Might be related to CUDA driver versions which despite updats are pinned and not updating.

### May 25
- Got EBA script working relatively efficiently for loading in data.  Took 30 min?
Still seems like lots of room for improvement there.
  Optimized by eliminating pandas in reading data, unique constraint on SQL, and simplifying where.
  Cut time by 2/3.

## May 23
- Moved library code around to src layout with setup.cfg
  (Motivated by need to allow scripts using library code to run and import code)
  Gave up on pyproject.toml as editable support is annoying.
- Renamed library to USElectricity for giggles.
- "pip install -e ." seems to work, but importing fails, and not found in pip list.  (possible setuptools bugs?)
   - Good discussion in here: https://stackoverflow.com/questions/29905909/pip-install-e-packages-dont-appear-in-docker about interaction of pip install and docker sync conflicting.
   Fix suggested of doing something like
   `pip install -e . --src /path/to/src`
   from the module directory works.
- created local virtual_env on host machine to install to allow better editting experience with VSCode with libraries.

## Mar 16

Copied dev over to new mac.
This lacks a GPU, so need to handle GPU/non-GPU builds of Docker
files.
Can control base image via TF_IMAGE arg specified in .env, which gets
picked up by docker-compose.
But, that has specific requirements for GPU, which fail if not
present.

So, will make two docker-compose files.

## Mar 9/10

- Sketch out the table structure and desired library.
- Hit issue trying to login to SQL.
- Getting `Notebook validation failed: Additional properties are not allowed ('id' was unexpected)`
 warning.  Can silence by manually editting and make this change on file `"nbformat_minor": 5`

### Postgres Docker cleanup
- blew away default database.
- had to set 'postgres' as hostname when connecting inside docker.
- then username/pw/db work as expected

## Mar 8

- Matplotlib inline in EIN:
  %matplotlib inline
  Add `(setq ein:output-area-inlined-images t)` to .emacs.  Otherwise it was using ImageMagick

Permanently Adjusting setup:
```
Open the configuration file within the hidden folder .ipython

`~/.ipython/profile_default/ipython_kernel_config.py`

add the following line

c.IPKernelApp.matplotlib = 'inline'

add it straight after

c = get_config()
```

- Cartopy was bugging out due to shapely.  Trying to use older version of shapely.
- Figuring out the forecast file locations (recent data in S3, with particular code for temp forecast)
- Can use `xarray` and `cfgrib` to read files.  Need eccode binaries installed.


## Mar 6
- Maybe move from mounting 'us_elec'?  Done
- Need to provide a "runbook" for loading in the data.


# Mar 2
- Running into issues trying to login to local jupyter server.  This is true with docker, but also when just trying to run
Jupyter in local linux environment.
- might need https going?
Managed to get a local jupyter server going that VSCode would listen and work with.  Hard to make it forget about those.
- use `ctrl + shift + p`: Jupyter - Clear Jupyter Remote Server List to forget connections

- After forgetting previous bad attempts, could make a new connection matching the Docker container at http://localhost:8890 and enter password.
Labelled the connection as TFJupyter.
Note: Need to change the kernel to the notebooks Kernel, otherwise it would just operate in the local linux env, rather than the DOcker container's kernel.  (Can verify by running `!hostname`)

- Can also change the output in VSCode to be from specific plugins (e.g. Jupyter) in the terminal.

## Feb 26
Added password (to avoid unique token nonsense) to serve results.  Adjusted to mount local .jupyter file.
May have to give up an just use Emacs for Jupyter again since VSCode for linux is sub-par on support.

## Feb 23
- Got it to work by just mounting files into /tf directory.  Can create files and edit them now.
- Can run JupyterLab?  Not sure it's worth the hassle if we can just use VSCode (which is far more comfortable for editting)
- I think VSCode will prove annoying while trying to attach to Jupyter running inside a Docker container.

## Feb 15
Tried FastAPI container to try debugging what was up with networking.

Note that you need to use "up" not "run" to bring up the machine and networks
and actually run the whole server

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
