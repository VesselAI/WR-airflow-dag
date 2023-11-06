def WR_pipeline_dag():
    def load_data():
        print('load data')
        from load_data import load_data

        data, waypoint_t0 = load_data()
        return data, waypoint_t0

    def preprocess(data, waypoint_t0):
        print('preprocess')
        from preprocess import run_preprocessing

        weather_points, points = run_preprocessing(data, waypoint_t0)

        waypoints, waypoint_t0, waypoint_g0 = points

        return {
            'weather_points': weather_points.tolist(), 
            'waypoints': waypoints,
            'waypoint_t0': waypoint_t0.isoformat(),
            'waypoint_g0': waypoint_g0,
        }

    def make_prediction(values):
        print('infer prediction')
        from run_model import run_model

        weather_points = values['weather_points']

        res = run_model(weather_points)

        print('res', res)


    data, waypoint_t0 = load_data()
    values = preprocess(data, waypoint_t0)
    make_prediction(values)

WR_pipeline_dag()
