from datetime import datetime
from airflow.decorators import dag, task

@dag(
    description='Preprocess data and get prediction.',
    start_date=datetime(2023, 7, 20),
    schedule=None,
)
def WR_pipeline_dag():
    @task(
        task_id='load_data',
    )
    def load_data():
        print('load data')
        from weather_routing.load_data import load_data

        data = load_data()
        return data


    @task(
        task_id='preprocess'
    )
    def preprocess(data):
        print('preprocess')
        from weather_routing.preprocess import run_preprocessing

        weather_points, points = run_preprocessing(data)

        waypoints, waypoint_t0, waypoint_g0 = points

        return {
            'weather_points': weather_points.tolist(), 
            'waypoints': waypoints,
            'waypoint_t0': waypoint_t0.isoformat(),
            'waypoint_g0': waypoint_g0,
        }


    @task(
        task_id='make_prediction'
    )
    def make_prediction(values):
        print('infer prediction')
        from weather_routing.run_model import run_model

        weather_points = values['weather_points']

        res = run_model(weather_points)

        print('res', res)


    data = load_data()
    values = preprocess(data)
    make_prediction(values)

dag = WR_pipeline_dag()
