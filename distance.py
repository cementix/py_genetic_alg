import math

def city_distance(city1, city2):
    x1, y1 = city1['x'], city1['y']
    x2, y2 = city2['x'], city2['y']
    
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2) 