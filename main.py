# import genetic_aalgorithm_tsp as GA

# import matplotlib.pyplot as plt
# import external_input as ex_input





# arg_names = [
#     "number_of_cities",
#     "population_size",
#     "mutation_rate",
#     "no_of_generations",
#     "elitism"
# ]

# args = dict(
#     zip(
#         arg_names,
#         ex_input.get_external_input(
#             [15, 10, 0.1, 1000, 2]
#         )
#     )
# )


# number_of_cities = (
#     args["number_of_cities"]
#     if type(args["number_of_cities"]) == int
#     else 15
# )

# population_size = (
#     args["population_size"]
#     if type(args["population_size"]) == int
#     else 10
# )

# mutation_rate = (
#     args["mutation_rate"]
#     if type(args["mutation_rate"]) == float
#     else 0.1
# )

# no_of_generations = (
#     args["no_of_generations"]
#     if type(args["no_of_generations"]) == int
#     else 1000
# )

# elitism = (
#     args["elitism"]
#     if type(args["elitism"]) == int
#     else 2
# )




# tsp = GA.Genitic_Algorithm(
#     number_of_cities=number_of_cities,
#     population_size=population_size,
#     mutation_rate=mutation_rate,
#     no_of_generations=no_of_generations,
#     elitism=elitism
# )




# (
#     best_route,
#     best_fitness,
#     best_fit_hist,
#     avg_fit_hist,
#     best_cost_hist
# ) = tsp.run()


# print(
#     "\n================ Optimization Finished ================"
# )

# print(
#     f"Optimal Route Found: {best_route}"
# )

# print(
#     f"Best Fitness Value: {best_fitness:.6f}"
# )

# print(
#     f"Calculated Path Cost: "
#     f"{1.0 / best_fitness:.2f}"
# )


# tsp.plot_convergence(
#     best_fit_hist,
#     avg_fit_hist,
#     best_cost_hist
# )


# from naive_tsp import NaiveTSP


# number_of_cities = 10

# naive = NaiveTSP(
#     number_of_cities=number_of_cities,
#     tsp_file="usa13509.tsp"
# )

# best_route, best_cost = naive.run()

# print(
#     "Naive Best Route:",
#     best_route
# )

# print(
#     "Naive Best Cost:",
#     best_cost
# )



# from dynamic_tsp import DynamicTSP


# number_of_cities = 15

# dp = DynamicTSP(
#     number_of_cities=number_of_cities,
#     tsp_file="usa13509.tsp"
# )

# best_route, best_cost = dp.run()

# print(
#     "DP Optimal Route:",
#     best_route
# )

# print(
#     "DP Optimal Cost:",
#     best_cost
# )


from hillclimbing_tsp import HillClimbingTSP


number_of_cities = 50

hill = HillClimbingTSP(
    number_of_cities=number_of_cities,
    tsp_file="usa13509.tsp",
    seed=42
)

best_route, best_cost = hill.run()

print(
    "Hill Climbing Route:",
    best_route
)

print(
    "Hill Climbing Cost:",
    best_cost
)