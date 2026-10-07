import numpy as np
import itertools
import atsp_read
import external_input as ex_input


class NaiveTSP:

    def __init__(
        self,
        number_of_cities,
        tsp_file="usa13509.tsp"
    ):

        self.number_of_cities = number_of_cities

        self.distance_matrix = atsp_read.read_tsplib(
            tsp_file,
            number_of_cities
        )

    def calculate_cost(self, route):

        total_cost = 0

        for i in range(
            len(route) - 1
        ):

            total_cost += self.distance_matrix[
                route[i],
                route[i + 1]
            ]

        return total_cost

    def run(self):

        start_city = 0

        intermediate_cities = range(
            1,
            self.number_of_cities
        )

        best_route = None
        best_cost = np.inf

        # Generate every possible permutation
        for permutation in itertools.permutations(
            intermediate_cities
        ):

            route = (
                [start_city]
                + list(permutation)
                + [start_city]
            )

            cost = self.calculate_cost(
                route
            )

            if cost < best_cost:

                best_cost = cost
                best_route = route

        return (
            np.array(best_route),
            best_cost
        )