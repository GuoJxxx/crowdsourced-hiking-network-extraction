import math


METERS_PER_DEGREE_LATITUDE = 111070.34306591158
METERS_PER_DEGREE_LONGITUDE = 83044.98918812413


def fast_distance(a_lat, a_lon, b_lat, b_lon):
    if a_lat == b_lat and a_lon == b_lon:
        return 0.0
    y_distance = METERS_PER_DEGREE_LATITUDE * (a_lat - b_lat)
    x_distance = METERS_PER_DEGREE_LONGITUDE * (a_lon - b_lon)
    return math.sqrt(y_distance * y_distance + x_distance * x_distance)

