"""
Preprocesses the loaded data as follows
    (i) Coverts timestamp to ISO8601 format and longitude in the western hemisphere to be positive
    (ii) Filters the needed coordinated as per decision time (waypoint_t0) and the current position (waypoint_g0)
    (iii) 

"""

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.interpolate import interpn
from geographiclib.geodesic import Geodesic as GD
import pytz
from datetime import datetime, timedelta
from airflow.operators.python import get_current_context
from airflow.models import Variable

file_path = Variable.get('dags_folder') + '/weather_routing/'

def run_preprocessing(data):
    waypoint_t0 = pd.to_datetime(get_current_context()['dag_run'].conf['waypoint_t0'])

    data = fix_datatypes(data)
    data = filter_waypoints(data, waypoint_t0)
    waypoints = calculate_waypoints(data)
    waypoints, waypoint_g0 = interpolate_points(waypoints, waypoint_t0)
    weather_points = interpolate_corresponding_weather_attributes(waypoints, waypoint_g0, waypoint_t0)

    waypoints.timestamp = waypoints.timestamp.apply(lambda x: x.isoformat())

    return weather_points, (waypoints, waypoint_t0, waypoint_g0)


AREA_BOUNDING_BOX = [(50, 156.00000694), (25, 228.50001016)] # upper left, lower right corner

LONG_LIST = np.array([156.00000694, 157.25000699, 158.50000705, 159.7500071 ,
                      161.00000716, 162.25000722, 163.50000727, 164.75000733,
                      166.00000738, 167.25000744, 168.50000749, 169.75000755,
                      171.00000761, 172.25000766, 173.50000772, 174.75000777,
                      176.00000783, 177.25000788, 178.50000794, 179.75000799,
                      181.00000805, 182.25000811, 183.50000816, 184.75000822,
                      186.00000827, 187.25000833, 188.50000838, 189.75000844,
                      191.00000849, 192.25000855, 193.50000861, 194.75000866,
                      196.00000872, 197.25000877, 198.50000883, 199.75000888,
                      201.00000894, 202.250009  , 203.50000905, 204.75000911,
                      206.00000916, 207.25000922, 208.50000927, 209.75000933,
                      211.00000938, 212.25000944, 213.5000095 , 214.75000955,
                      216.00000961, 217.25000966, 218.50000972, 219.75000977,
                      221.00000983, 222.25000988, 223.50000994, 224.75001000,
                      226.00001005, 227.25001011, 228.50001016])

LAT_LIST = np.array([50., 49.75, 49.5 , 49.25, 49., 48.75, 48.5, 48.25, 48.,
       47.75, 47.5 , 47.25, 47.  , 46.75, 46.5 , 46.25, 46.  , 45.75,
       45.5 , 45.25, 45.  , 44.75, 44.5 , 44.25, 44.  , 43.75, 43.5 ,
       43.25, 43.  , 42.75, 42.5 , 42.25, 42.  , 41.75, 41.5 , 41.25,
       41.  , 40.75, 40.5 , 40.25, 40.  , 39.75, 39.5 , 39.25, 39.  ,
       38.75, 38.5 , 38.25, 38.  , 37.75, 37.5 , 37.25, 37.  , 36.75,
       36.5 , 36.25, 36.  , 35.75, 35.5 , 35.25, 35.  , 34.75, 34.5 ,
       34.25, 34.  , 33.75, 33.5 , 33.25, 33.  , 32.75, 32.5 , 32.25,
       32.  , 31.75, 31.5 , 31.25, 31.  , 30.75, 30.5 , 30.25, 30.  ,
       29.75, 29.5 , 29.25, 29.  , 28.75, 28.5 , 28.25, 28.  , 27.75,
       27.5 , 27.25, 27.  , 26.75, 26.5 , 26.25, 26.  , 25.75, 25.5 ,
       25.25])

# ------

def fix_datatypes(df:pd.DataFrame):
    """  
    Coversion of timestamp to ISO8601 format and longitude in the western hemisphere to be positive
    
    @param df pandas dataframe with original route waypoints
    @param gdf returned geopandas dataframe with timestamp and longitude conversions
    """

    # convert datatime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    # fix longitude - converts longitudes in the western hemisphere to positive numbers
    df.loc[df['longitude'] < 0, 'longitude'] += 360

    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4326)

# ------

def filter_waypoints(df:gpd.GeoDataFrame, waypoint_t0):
    """  
    Filter and only include the needed coordinates starting from the decision time (waypoint_t0) 
    and are within the experimental area
        
    @param gdf          geopandas dataframe with original route waypoints
    @param waypoint_t0  time at which re-routing decision is requested
    @param filtered_df  returned geopandas dataframe with waypoints within the experimental area 
                        starting from when re-routing decision was requested
    """

    filtered_df = df.cx[AREA_BOUNDING_BOX[1][1]:AREA_BOUNDING_BOX[0][1], :]

    if len(filtered_df) == 0:
        raise ValueError('Rerouting point outside the right bound of the experimenting area')
    
    filtered_df = filtered_df[filtered_df['timestamp'] >= waypoint_t0]

    index = list(filtered_df.index)

    index.extend([np.max(index) + 1, np.min(index) - 1])

    filtered_df = df.iloc[index, :]

    return filtered_df.sort_values('timestamp')

# ------

def calculate_waypoints(df:gpd.GeoDataFrame):
    """  
    Create section waypoints from the filtered data. First, cog is calculated between two points 
    using the bearing function. Then the cog is appended to the section waypoints dataframe
        
    @param E_*, L* two information of longitude, latitude & timestamp of two consecutive waypoints from filtered section df          
    @param cog calculated cog between conscutive waypoints
    @param section_waypoints
    """
    # Store information between each pair of consecutive waypoints from filtered dataframe 
    sections_waypoints = pd.DataFrame(columns=['E_latitude', 'E_longitude', 'E_timestamp', 'L_latitude',
                                                    'L_longitude',  'L_timestamp', 'cog'])
    for i in range(len(df) - 1):
        row1 = df.iloc[i]
        row2 = df.iloc[i + 1]

        # Calculate course over ground between two waypoints
        cog = get_bearing(row1['latitude'], row2['latitude'], row1['longitude'], row2['longitude'])

        # Add the information to the sections_waypoints dataframe
        sections_waypoints = pd.concat([sections_waypoints, pd.Series({
                'E_latitude': row1['latitude'],
                'E_longitude': row1['longitude'],
                'E_timestamp': row1['timestamp'],
                'L_latitude': row2['latitude'],
                'L_longitude': row2['longitude'],
                'L_timestamp': row2['timestamp'],
                'cog': cog
        }).to_frame().T], ignore_index=True)
    
    sections_waypoints['E_latitude'] = sections_waypoints['E_latitude'].astype(float)
    sections_waypoints['E_longitude'] = sections_waypoints['E_longitude'].astype(float)
    sections_waypoints['L_latitude'] = sections_waypoints['L_latitude'].astype(float)
    sections_waypoints['L_longitude'] = sections_waypoints['L_longitude'].astype(float)
    sections_waypoints['E_timestamp'] = pd.to_datetime(sections_waypoints['E_timestamp'])
    sections_waypoints['L_timestamp'] = pd.to_datetime(sections_waypoints['L_timestamp'])
    sections_waypoints['cog'] = sections_waypoints['cog'].astype(float)

    return sections_waypoints

def get_bearing(lat1, lat2, long1, long2):
    '''
    The function is used to calculate the CoG from node 1 to node 2
    '''
    
    brng = GD.WGS84.Inverse(lat1, long1, lat2, long2)['azi1']
    if brng < 0:
        brng = brng + 360
    return brng

# ------

def interpolate_points(sections_waypoints, waypoint_t0):

    interpolation_full = pd.DataFrame({'longitude': LONG_LIST})
    # Add empty columns to the DataFrame
    interpolation_full['latitude'] = np.nan  # longitudes are given according to the grids
    # latitudes are interpolated
    interpolation_full['cog'] = np.nan
    interpolation_full['timestamp'] = pd.NaT
    interpolation_full['arriving_time_latest'] = np.nan
    interpolation_full['arriving_time_previous'] = np.nan
    # return the index of the smallest longitude in grids that is larger than the current position G0[1]
    first_row = sections_waypoints[sections_waypoints.L_timestamp > waypoint_t0].iloc[:1]
    WAYPOINT_G0 = find_G0(first_row.E_longitude.values[0], first_row.L_longitude.values[0],
                          first_row.E_latitude.values[0],  first_row.L_latitude.values[0])
    s_grids_lon_index = (interpolation_full['longitude'] >= WAYPOINT_G0[1]).idxmax()
    # interpolate latitude timestamp and write cog
    interpolation_full.loc[s_grids_lon_index:, ['latitude', 'timestamp', 'cog']] = (
        interpolation_full.loc[s_grids_lon_index:, :]
        .apply(lambda x: intp_lat_ts_cog(x, sections_waypoints), axis=1))
    return interpolation_full, WAYPOINT_G0

def find_G0(lon_start, lon_end, lat_start, lat_end):
    lon = [val for val in LONG_LIST if lon_start < val < lon_end][0]
    lat = [val for val in LAT_LIST if lat_start < val < lat_end][0]
    return (lat, lon)

def intp_lat_ts_cog(row, sections_waypoints):
    """
    Finds the row in `sections_waypoints` where the logitudes in the grid `t_lon` falls between
    the `E_longitude` and `L_longitude` values. Calculates the latitude
    and timestamp for the target point T_point and sets the `latitude`,
    `timestamp`, and `cog` columns in accordingly.

    Parameters:
    t_lon (float): The longitude value to search for in `sections_waypoints`.
    sections_waypoints (pd.DataFrame): The DataFrame containing the sections and waypoints.

    Returns:
    pd.DataFrame: a row with interpolated values of latitude, timestamp and cog
    """
    # Get t_lon value
    t_lon = row['longitude']
    # Find row in `sections_waypoints` where `t_lon` falls between
    #         the `E_longitude` and `L_longitude` values
    mask = (sections_waypoints['E_longitude'] <= t_lon) & (t_lon <= sections_waypoints['L_longitude'])
    row_sw = sections_waypoints.loc[mask]
    # Calculate t_lat using linear interpolation
    point1 = np.array([row_sw['E_latitude'], row_sw['E_longitude']])
    point2 = np.array([row_sw['L_latitude'], row_sw['L_longitude']])
    t_lat = point1[0] + ((t_lon - point1[1]) / (point2[1] - point1[1])) * (point2[0] - point1[0])
    # Calculate t_timestamp using linear interpolation
    e_ts = row_sw['E_timestamp']
    l_ts = row_sw['L_timestamp']
    t_ts = e_ts + ((t_lon - point1[1])[0] / (point2[1] - point1[1])[0]) * (l_ts - e_ts)
    return pd.Series({'latitude': t_lat[0], 'timestamp': t_ts.values[0], 'cog': row_sw['cog'].iloc[0]})

# ------

def interpolate_corresponding_weather_attributes(interpolation_full, waypoint_g0, waypoint_t0):
    # Define a function to calculate the time difference in hours
    # between the arriving time at a location and the starting time of the weather forecast
    s_grids_lon_index = (interpolation_full['longitude'] >= waypoint_g0[1]).idxmax()
    forecast_name, forecast_ts = get_last_weather_forecast(waypoint_t0)
    # read in the needed weather forecast file
    wf_latest = load_weather_forcast(forecast_name[0])
    wf_previous = load_weather_forcast(forecast_name[1])
    # flip for the 1st dimension (latitude) because interpolation requires first dimension in accending order
    # slice and drop the last longitude as weather attributes are not needed for the destination
    wf_latest = np.flip(wf_latest, axis=0)[:, :-1, :, :]
    wf_previous = np.flip(wf_previous, axis=0)[:, :-1, :, :]
    time_diff0 = lambda x: (pytz.utc.localize(x['timestamp']) - forecast_ts[0]) / timedelta(hours=1)
    time_diff1 = lambda x: (pytz.utc.localize(x['timestamp']) - forecast_ts[1]) / timedelta(hours=1)

    # Apply the function to the dataframe
    interpolation_full.loc[s_grids_lon_index:, 'arriving_time_latest'] = (interpolation_full.loc[s_grids_lon_index:, :]
                                                                          .apply(time_diff0, axis=1))
    interpolation_full.loc[s_grids_lon_index:, 'arriving_time_previous'] = (
        interpolation_full.loc[s_grids_lon_index:, :]
        .apply(time_diff1, axis=1))
    interpolation_full.loc[s_grids_lon_index:, ['tf_latest', 'tf_previous']] = (
        interpolation_full.loc[s_grids_lon_index:, :]
        .apply(hour2timeframe, axis=1))
    # interpolation of weather conditions forecasted in the latest weather forecast and the previous weather forecast
    interpolation_full.loc[s_grids_lon_index:, 'latest_weather'] = interpolation_full.loc[s_grids_lon_index:, :].apply(
        lambda row:
        interp_row(row, wf_latest, 'tf_latest'), axis=1)
    interpolation_full.loc[s_grids_lon_index:, 'previous_weather'] = interpolation_full.loc[s_grids_lon_index:,
                                                                     :].apply(lambda row:
                                                                              interp_row(row, wf_previous,
                                                                                         'tf_previous'), axis=1)
    converted_weather = interpolation_full.loc[s_grids_lon_index:, :].apply(lambda row: convert_weather_array(
        row['latest_weather'], row['cog']) - convert_weather_array(row['previous_weather'], row['cog']), axis=1)

    # Concatenate the converted weather arrays into a single flattened array
    flattened_weather = np.concatenate(converted_weather.to_numpy())

    # Zero-pad the flattened array if necessary
    if len(flattened_weather) < 59 * 8:
        padding = np.zeros((59 * 8 - len(flattened_weather),))
        flattened_weather = np.concatenate((padding, flattened_weather))
    return flattened_weather

def load_weather_forcast(forecast_name):
    return np.load(file_path + 'gfs_NP_' + forecast_name + '.npy')

def get_last_weather_forecast(waypoint_t0):
    '''return the forecast files names and the timestamps of the forecast'''
    # only keep the year month day hour minute seconds
    dt = waypoint_t0
    forecast_hour = (dt.hour // 6) * 6
    last_forecast = pytz.utc.localize(datetime(dt.year, dt.month, dt.day, forecast_hour))
    previous_forecast = last_forecast - timedelta(hours=6)
    # the most recent forecast and the previous one
    last_forecast_name = f'{last_forecast.year:02d}{last_forecast.month:02d}{last_forecast.day:02d}{last_forecast.hour:02d}'
    previous_forecast_name = f'{previous_forecast.year:02d}{previous_forecast.month:02d}{previous_forecast.day:02d}{previous_forecast.hour:02d}'
    return (last_forecast_name, previous_forecast_name), (last_forecast, previous_forecast)

def hour2timeframe(row):
    '''The timeframe is from 0 to 209, but the real hour is from 0 to 384.
    From hour 120 onwards, every three hours per timeframe.
    The function is to find out the positions in the timeframes according to the hour'''
    if row['arriving_time_latest'] > 120:
        tf_latest = (row['arriving_time_latest'] - 120) / 3 + 120
    else:
        tf_latest = row['arriving_time_latest']

    if row['arriving_time_previous'] > 120:
        tf_previous = (row['arriving_time_previous'] - 120) / 3 + 120
    else:
        tf_previous = row['arriving_time_previous']
    return pd.Series({'tf_latest': tf_latest, 'tf_previous': tf_previous})

def interp_row(row, wf, tsn):
    '''
    The function aims to retrieve and interpolate the weather conditions at the given position according
    to its spatial coordinates and arriving time (hour)
    input:
    wf is the weather forecast array, wf_latest or wf_previous
    tsn = 'tf_latest' or 'tf_previous' according to wf
    output: weather conditions for wind, wave and swell (8)
    '''
    # Define the interpolation points as a tuple of arrays
    interp_points = (row['latitude'], row['longitude'], row[tsn], range(wf.shape[3]))

    # Interpolate the weather variable values at the interpolation points
    var_values = interpn((np.flip(LAT_LIST), LONG_LIST, range(wf.shape[2]), range(wf.shape[3])),
                         wf, interp_points, method='linear')
    return var_values

def convert_weather_array(weather_array, cog):
    # Define a function that converts a 1D numpy array of weather variables to the desired format
    # Make a copy of the input array of weather conditions
    converted_array = np.copy(weather_array)

    # Define the indices of the directions of wind, wave and swell
    direction_indices = [1, 4, 7]

    # Subtract the cog from the direction and take the absolute value and convert if larger than 180 degree
    for idx in direction_indices:
        diff = abs(weather_array[idx] - cog)
        if diff <= 180:
            converted_array[idx] = diff
        else:
            converted_array[idx] = 360 - diff
    return converted_array

