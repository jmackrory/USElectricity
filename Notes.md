# User Setup Notes
- Changelog and instructions on how to run the app.
- Including motivation and changes
- Includes daily changes.

## Running Docker setup
- change dir to root of project
build container:
`docker compose -f docker/docker-compose.yml build`

run container:
`docker compose -f docker/docker-compose.yml up`
- should have whole project folder mounted to dir.

shut it all down
"docker compose -f docker/docker-compose.yml down"

Connect by logging into to `http://localhost:8890` and providing Password.

#### VS Code
- Use Jupyter plugin to log in to remote `localhost:8890`.  Note that it may be necessary to
Clear the Remote Server List.
- It's also important to make sure you select the Docker container environment kernel, as otherwise it will run in the local environment.

### Emacs IPython Notebok

- `Ctrl-c Ctrl-x`
- `ein: notebooklist-login`
- Provide port 8890, then password.

## Mar 8


- Matplotlib inline:
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
- Need to provide a "runbook" for



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
