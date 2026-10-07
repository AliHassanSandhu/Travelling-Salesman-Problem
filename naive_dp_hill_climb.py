import external_input as ex_input
from calculate_time import calculate_execution_time
import genetic_aalgorithm_tsp as GA
from naive_tsp import NaiveTSP
from dynamic_tsp import DynamicTSP
from hillclimbing_tsp import HillClimbingTSP

TSP_FILE = "usa13509.tsp"
NAIVE_CITIES = 11
DP_CITIES = 20
HILL_CITIES = 50

@calculate_execution_time
def run_naive():
    naive = NaiveTSP(number_of_cities=NAIVE_CITIES, tsp_file=TSP_FILE)
    return naive.run()

print("\n--- Running Naive TSP ---")
naive_res = run_naive()
# Handling potential variation in wrapper return structures
if isinstance(naive_res, tuple) and len(naive_res) == 2 and isinstance(naive_res[0], tuple):
    (naive_route, naive_cost), naive_time = naive_res
else:
    naive_route, naive_cost = naive_res[0], naive_res[1]
    naive_time = "N/A"

with open("naive_output.txt", "w") as f:
    f.write(f"Naive Best Route: {naive_route}")
    f.write(f"Naive Best Cost : {naive_cost}")
    f.write(f"Execution Time  : {naive_time}")

print(f"Naive Best Route: {naive_route}")
print(f"Naive Best Cost : {naive_cost}")
print(f"Execution Time  : {naive_time}")


@calculate_execution_time
def run_dp():
    dp = DynamicTSP(number_of_cities=DP_CITIES, tsp_file=TSP_FILE)
    return dp.run()

print("\n--- Running Dynamic Programming TSP ---")
dp_res = run_dp()
if isinstance(dp_res, tuple) and len(dp_res) == 2 and isinstance(dp_res[0], tuple):
    (dp_route, dp_cost), dp_time = dp_res
else:
    dp_route, dp_cost = dp_res[0], dp_res[1]
    dp_time = "N/A"

with open("dp_output.txt", "w") as f:
    f.write(f"DP Best Route   : {dp_route}")
    f.write(f"DP Best Cost    : {dp_cost}")
    f.write(f"Execution Time  : {dp_time}")

print(f"DP Best Route   : {dp_route}")
print(f"DP Best Cost    : {dp_cost}")
print(f"Execution Time  : {dp_time}")



@calculate_execution_time
def run_hill_climbing():
    hill = HillClimbingTSP(number_of_cities=HILL_CITIES, tsp_file=TSP_FILE, seed=42)
    return hill.run()

print("\n--- Running Hill Climbing TSP ---")
hill_res = run_hill_climbing()
if isinstance(hill_res, tuple) and len(hill_res) == 2 and isinstance(hill_res[0], tuple):
    (hill_route, hill_cost), hill_time = hill_res
else:
    hill_route, hill_cost = hill_res[0], hill_res[1]
    hill_time = "N/A"

with open("hill_output.txt", "w") as f:
    f.write(f"Hill Climbing Route: {hill_route}")
    f.write(f"Hill Climbing Cost : {hill_cost}")
    f.write(f"Execution Time     : {hill_time}")

print(f"Hill Climbing Route: {hill_route}")
print(f"Hill Climbing Cost : {hill_cost}")
print(f"Execution Time     : {hill_time}")


