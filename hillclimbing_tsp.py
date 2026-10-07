import numpy as np
import atsp_read


class HillClimbingTSP:

    def __init__(
        self,
        number_of_cities,
        tsp_file="usa13509.tsp",
        seed=42
    ):

        self.number_of_cities = number_of_cities

        self.distance_matrix = atsp_read.read_tsplib(
            tsp_file,
            number_of_cities
        )

        self.rng = np.random.default_rng(seed)

    def create_initial_route(self):

        # City 0 is fixed as the starting/ending city
        intermediate_cities = np.arange(
            1,
            self.number_of_cities
        )

        permutation = self.rng.permutation(
            intermediate_cities
        )

        route = np.concatenate(
            (
                [0],
                permutation,
                [0]
            )
        )

        return route

    def calculate_cost(self, route):

        total_cost = 0

        for i in range(
            self.number_of_cities
        ):

            edge_cost = self.distance_matrix[
                route[i],
                route[i + 1]
            ]

            if np.isinf(edge_cost):

                return np.inf

            total_cost += edge_cost

        return total_cost

    def get_neighbors(self, route):

        neighbors = []

        # Only swap intermediate cities.
        # City 0 remains fixed.
        for i in range(
            1,
            self.number_of_cities
        ):

            for j in range(
                i + 1,
                self.number_of_cities
            ):

                neighbor = route.copy()

                neighbor[i], neighbor[j] = (
                    neighbor[j],
                    neighbor[i]
                )

                neighbors.append(
                    neighbor
                )

        return neighbors

    def run(self):

        # ----------------------------------------
        # Step 1: Generate initial solution
        # ----------------------------------------

        current_route = (
            self.create_initial_route()
        )

        current_cost = (
            self.calculate_cost(
                current_route
            )
        )

        # Keep track of the best solution
        best_route = current_route.copy()
        best_cost = current_cost

        # ----------------------------------------
        # Step 2: Iteratively improve solution
        # ----------------------------------------

        while True:

            neighbors = self.get_neighbors(
                current_route
            )

            # Find the best neighboring route
            next_route = min(
                neighbors,
                key=self.calculate_cost
            )

            next_cost = (
                self.calculate_cost(
                    next_route
                )
            )

            # ------------------------------------
            # Step 3: Stop if no improvement
            # ------------------------------------

            if next_cost >= current_cost:

                break

            # Move to better solution
            current_route = next_route
            current_cost = next_cost

            # Update best solution
            if current_cost < best_cost:

                best_route = current_route.copy()
                best_cost = current_cost

        return (
            best_route,
            best_cost
        )