# WR Airflow DAG Package
This repository contains the Airflow DAG for the weather routing decision model.

It also contains one example configuration in JSON format and the needed preprocessed weather files.

## Changes needed
The weather data are currently loaded locally in `preprocess.py` in the function `load_weather_forcast`. More historic preprocessed files are at http://datalake.vessel-ai.eu/minio/pilot4-weather-data/

In `run_model.py` I included code to load the pretrained model from MLflow. Please check what changes are needed.