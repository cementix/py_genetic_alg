from tsp_parse import parse_tsp_df
from utils import city_distance, calculate_fitness
import random

df1 = parse_tsp_df("./tsps/berlin52.tsp")
df2 = parse_tsp_df("./tsps/berlin11_modified.tsp")

# print(len(df1), len(df2))
# print(df1.dtypes)
# print(df1)

city1 = df1.loc[df1.id == 1].iloc[0]
city2 = df1.loc[df1.id == 2].iloc[0]

dist = city_distance(city1, city2)
print("distance between city 1 and 2:", dist)


def create_random_solution(df):
    cities = df["id"].tolist()
    random.shuffle(cities)
    return cities


solution = create_random_solution(df2)
print("random solution:", solution)

ids = df2["id"].tolist()
if sorted(solution) == sorted(ids):
    print("all cities included, no duplicates")
else:
    print("something wrong (missing or duplicate ids)")

test_solution = create_random_solution(df2)
fitness = calculate_fitness(df2, test_solution)
print("test solution:", test_solution)
print("fitness:", fitness)

if fitness > 0:
    print("fitness seems valid (positive value)")
else:
    print("fitness error: distance must be positive")
