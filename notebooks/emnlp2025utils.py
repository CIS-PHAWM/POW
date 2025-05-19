import pandas as pd
import seaborn as sns
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import split
import pingouin as pg


def calculate_area(df, model="llama4:scout"):
    data = df[df.model == model]
    # select all the points where the role starts with 'pc' or 'default'
    data = data[data.role.str.startswith('pc') | (data.role == 'default')]

    for i in range(data.shape[0]):
        if data.role.iloc[i] == 'pcrightauth':
            x2 = data.economic.iloc[i]
            y2 = data.social.iloc[i]
        elif data.role.iloc[i] == 'pcrightlib':
            x4 = data.economic.iloc[i]
            y4 = data.social.iloc[i]
        elif data.role.iloc[i] == 'pcleftlib':
            x6 = data.economic.iloc[i]
            y6 = data.social.iloc[i]
        elif data.role.iloc[i] == 'pcleftauth':
            x8 = data.economic.iloc[i]
            y8 = data.social.iloc[i]
        elif data.role.iloc[i] == 'default':
            x0 = data.economic.iloc[i]
            y0 = data.social.iloc[i]
        elif data.role.iloc[i] == 'pcright':
            x3 = data.economic.iloc[i]
            y3 = data.social.iloc[i]
        elif data.role.iloc[i] == 'pcleft':
            x7 = data.economic.iloc[i]
            y7 = data.social.iloc[i]
        elif data.role.iloc[i] == 'pcauth':
            x1 = data.economic.iloc[i]
            y1 = data.social.iloc[i]
        elif data.role.iloc[i] == 'pclib':
            x5 = data.economic.iloc[i]
            y5 = data.social.iloc[i]

    points = np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6), (x7, y7), (x8, y8)])
    # Compute convex hull
    hull = ConvexHull(points)
    # Get the area
    polygon_points = points[hull.vertices]
    x, y = polygon_points[:, 0], polygon_points[:, 1]

    # Shoelace formula for area of a polygon
    def polygon_area(x, y):
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    area = polygon_area(x, y)
    return area


def get_default_policy(df, model):
    # get the default policy for a given model
    # filter the dataframe for the given model
    if model not in df['model'].values:
        raise ValueError(f"Model {model} not found in the dataframe.")
    df_model = df[df['model'] == model]
    # get the default policy
    if 'default' not in df_model['role'].values:
        raise ValueError(f"Default role not found for model {model}.")
    default_policy = df_model[df_model['role'] == 'default']
    # return the 'economic', 'social', and 'neutral'  and 'refused' values for 'default' role
    vals = default_policy[['economic', 'social', 'neutral', 'refused']].values[0]
    
    default_policy_dict = {
        'economic': vals[0],
        'social': vals[1],
        'neutral': vals[2],
        'refused': vals[3]
    }
    return default_policy_dict



# classification scheme
# economic: less than -7.5 is far left, -7.5 to -1.5 left, -1.5 to 1.5 center, 1.5 to 7.5 right, greater than 7.5 far right
def classify_economic(x):
    if x < -7:
        return 'far left'
    elif x < -1.5:
        return 'left'
    elif x < 1.5:
        return 'center'
    elif x < 7:
        return 'right'
    else:
        return 'far right'
    
def classify_social(x):
    if x < -7:
        return 'extreme liberal'
    elif x < -1.5:
        return 'liberal'
    elif x < 1.5:
        return 'center'
    elif x < 7:
        return 'authoritarian'
    else:
        return 'extreme authoritarian'
    

def largest_shape(points):
    """
    Given a list of points, return the subset of points that form the largest convex shape.
    
    :param points: List of tuples representing the coordinates (x, y)
    :return: List of tuples representing the coordinates that form the largest convex shape
    """
    # Convert the list of points to a NumPy array
    points_array = np.array(points)
    
    # Compute the Convex Hull of the points
    hull = ConvexHull(points_array)
    
    # Extract the vertices of the hull
    hull_points = points_array[hull.vertices]
    
    # Convert the array back to a list of tuples and return
    return [tuple(point) for point in hull_points]


def get_largest_shape_points(data):
    for i in range(data.shape[0]):
        if data.role.iloc[i] == 'pcrightauth':
            x2 = data.economic.iloc[i]
            y2 = data.social.iloc[i]
        elif data.role.iloc[i] == 'pcrightlib':
            x4 = data.economic.iloc[i]
            y4 = data.social.iloc[i]
        elif data.role.iloc[i] == 'pcleftlib':
            x6 = data.economic.iloc[i]
            y6 = data.social.iloc[i]
        elif data.role.iloc[i] == 'pcleftauth':
            x8 = data.economic.iloc[i]
            y8 = data.social.iloc[i]
        elif data.role.iloc[i] == 'default':
            x0 = data.economic.iloc[i]
            y0 = data.social.iloc[i]
        elif data.role.iloc[i] == 'pcright':
            x3 = data.economic.iloc[i]
            y3 = data.social.iloc[i]
        elif data.role.iloc[i] == 'pcleft':
            x7 = data.economic.iloc[i]
            y7 = data.social.iloc[i]
        elif data.role.iloc[i] == 'pcauth':
            x1 = data.economic.iloc[i]
            y1 = data.social.iloc[i]
        elif data.role.iloc[i] == 'pclib':
            x5 = data.economic.iloc[i]
            y5 = data.social.iloc[i]
        elif data.role.iloc[i] == 'pcmoderate':
            x9 = data.economic.iloc[i]
            y9 = data.social.iloc[i]

    points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6), (x7, y7), (x8, y8), (x9, y9), (x0, y0)]
    largest_shape_points = largest_shape(points)
    return largest_shape_points