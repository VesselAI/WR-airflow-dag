"""
Creates and returns a pandas dataframe from the provided original route (json/csv).
The route file consists of waypoints and timestamps at the waypoints
waypoint_t0 is the time at which re-routing decision is requested 

@param example_file     a json/csv file of the original route
@param df               returned pandas dataframe              
"""

import json
import pandas as pd

# Currently pointing to the example route json file contained in the folder
# TO DO - if the files are stored in minio
example_file = './example_voyage.json'

def load_data():
    with open(example_file) as file:
        conf = json.load(file)

    df = pd.json_normalize(conf, 'records')

    return df, conf['waypoint_t0']
