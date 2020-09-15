# Multimodal Human-in-the-Loop (HITL) Learning

Integrates eye gaze and natural language for human-in-the-loop learning

## Installation

### Dataset Download

The entire dataset containing demonstration trajectories for the _Find and Intercept the Yellow Truck_ in _AirSim_ is stored on a shared drive in Google Drive that is managed by _Data Version Control_. An e-mail address needs to be provided that can be added for access to the shared drive.

To download the dataset for experiments supported by this codebase:
* Clone the repository and enter it.
* Run ```dvc pull```. The first run of this command will require Google Authentication and entering a generated access code into the terminal. Interrupted downloads can be resumed by running ```dvc pull``` again.

To download the dataset as a standalone folder (with no versioning support etc.):
* Run,
```
dvc get https://gitlab.sitcore.net/nick.waytowich/multimodal_hitl \
          dataset
```

### Deploying in a Docker container
NOTE: Ensure Docker is setup to run without sudo for the following commands.

* Clone the repository and enter it.
* Either build the Docker image by running 
```
docker build -t hitl .
``` 
OR pull and rename a readymade image by executing 
```
docker pull ritwikbera/multimodal:latest
docker tag ritwikbera/multimodal:latest hitl
```
* Run and enter the Docker container
```
docker run \ 
    -u 0 \
    -it --rm \ 
    --name hitl_instance \ 
    --gpus all \ 
    -e DISPLAY=$DISPLAY 
    -v $(pwd):/workspace/multimodal \
    -v /tmp/.X11-unix:/tmp/.X11-unix \ 
    hitl
```
* Detach the container by executing Ctrl+P followed by Ctrl+Q
* Run AirSim in a separate terminal inside the same container by executing,
```
docker exec \
    -u 1000 \
    -it hitl_instance \
    ./airsim_static
```

* _ngrok_ is installed in the docker image to support port tunneling, so that various browser-based applications can be viewed on a local machine.
Run,
```
docker exec \
    -u 1000 \
    -it hitl_instance \
    ./ngrok http <port_id>
```
    * _Accessing MLFlow Tracking UI_: _port_id_ would be 5000. Ensure _mlflow ui_ is running in the container.

    * _Accessing TensorBoard_ : _port_id_ would be 6006. Ensure _tensorboard_ is running in the container.

### Managing Experiments

To run MLFlow's tracking service with a PostgreSQL backend, follow the following steps after installing all dependencies specified in the Dockerfile (for system packages) and requirements.txt (for python packages) on your server machine (that will host the tracking service). For the purposes of this README lets assume we are working on our local machine, i.e. database is hosted on localhost:

* We will create the admin user and a database for the MLFlow Tracking Server
```
sudo -u postgres psql
```
* In the psql console,
```
CREATE DATABASE mlflow_db;
CREATE USER mlflow_user WITH ENCRYPTED PASSWORD 'mlflow';
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;
```
* Locate your _pg_hba.conf_ file and append, to allow the SQL databases to be accessed from any IP (by the tracking server)
```
host    all    all    0.0.0.0/0    trust
```
* Restart the PostgreSQL service for the above access configuration changes to take effect
```
service postgresql restart
```
* Set the following environment variable and ensure whichever terminal your training script runs on echos the following set tracking URI (`echo $MLFLOW_TRACKING_URI`)
```
export MLFLOW_TRACKING_URI='http://0.0.0.0:8000'
```
* Run the server
```
mlflow server --backend-store-uri postgresql://mlflow_user:mlflow@localhost/mlflow_db --default-artifact-root file:<your-local-artifact_dir> -h 0.0.0.0 -p 8000
```
NOTE: You may have to create a new experiment for a clean reset of the tracking URI and artifact URI.

* To do any SQL manipulation on the _mlflow_db_ database and manipulate experimental data, you can access as _mlflow_user_ with the password using,
```
psql -h localhost -U mlflow_user -d mlflow_db
```
Note, for the above to smoothly happen, edit your _pg_hba.conf_ file by replacing the line
```
local   all             all                                     peer
```
with
```
local   all             all                                     md5
```
and restart the _postgresql_ service.

### Virtualenv method

NOTE: This installation procedure was tested on Ubuntu 18.04 LTS.  

Install _pip_ and _virtualenv_ to handle all Python3 dependencies:  
```sudo apt-get install python3-pip```  
```python3 -m pip install --user virtualenv```  

Create a new virtual environment:  
```python3 -m venv ~/venvs/multimodal_hitl```

Clone this repo and go to its folder:   
```cd multimodal_hitl```  

Activate the new environment and install dependencies:  
```source ~/venvs/multimodal_hitl/bin/activate```  
```pip install wheel```  
```pip install -r requirements.txt```

### AirSim

No need to build AirSim or Unreal Engine, just download and extract the official binaries from the latest AirSim release for Linux: https://github.com/microsoft/AirSim/releases/tag/v1.3.1-linux. Specifically, we have been using the Neighborhood binary: https://github.com/microsoft/AirSim/releases/download/v1.3.1-linux/Neighborhood.zip.

Make sure to also copy the ```settings.json``` file to ```~/Documents/AirSim```.

## Collecting Data

### AirSim

Connect a Xbox controller to your computer and run the binary shell file. In a separate terminal, activate the virtual environment ```source ~/venvs/multimodal_hitl/bin/activate```, go to the ```airsim``` folder and run ```python log_airsim.py```. All the data will be automatically logged to ```data/<experiment_name>```.

## Project Overview
__Technology stack used in this project__
![Technology stack used in this project](assets/stack.png)

__Model Overview__
![Model Overview](assets/gaze_vscl.png)

__Programmatic Flow__
![Programmatic Flow](assets/API.png)

__HDFS-based dataset storage__
![HDFS-based dataset storage](assets/h5.png)

__Analysis tool built for forensic analysis of high error/model failure points in the dataset__
![Analysis tool built for forensic analysis of high error/model failure points in the dataset](assets/tool.png)

__Human-in-the-loop data collection and policy evaluation__
![Human-in-the-loop data collection and policy evaluation](assets/interven.png)

