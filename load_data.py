"""
Creates and returns a pandas dataframe from the provided original route (json/csv).
The route file consists of waypoints and timestamps at the waypoints
waypoint_t0 is the time at which re-routing decision is requested 

@param example_file     a json/csv file of the original route
@param df               returned pandas dataframe              
"""

import json
import pandas as pd
from datetime import datetime
from airflow.operators.python import get_current_context
from airflow.models import Variable

# Currently pointing to the example route json file contained in the folder
# TO DO - if the files are stored in minio
example_file = Variable.get('dags_folder') + '/weather_routing/example_voyage.json'

def load_data():
    context = get_current_context()
    dag_run = context['dag_run']
    conf = dag_run.conf

    if conf == {}:
        print('empty conf.. using example data')
        with open(example_file) as file:
            conf = json.load(file)
            dag_run.conf['records'] = conf['records']
            print(type(conf['waypoint_t0']))
            dag_run.conf['waypoint_t0'] = conf['waypoint_t0']

    if 'waypoint_t0' not in conf:
        # set starting waypoint
        conf['waypoint_t0'] = datetime.now().isoformat()

    # TO DO - handling csv files
    df = pd.json_normalize(conf, 'records')

    return df
