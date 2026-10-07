import numpy as np


def _geo_distance_matrix(coordinates):
    """
    Calculate TSPLIB GEO distances.
    """
    PI = np.pi
    RRR = 6378.388

    lat = coordinates[:, 0]
    lon = coordinates[:, 1]

    def convert(value):
        deg = np.floor(value)
        minute = value - deg
        return PI * (deg + 5.0 * minute / 3.0) / 180.0

    lat = np.array([convert(v) for v in lat])
    lon = np.array([convert(v) for v in lon])

    q1 = np.cos(lon[:, None] - lon[None, :])
    q2 = np.cos(lat[:, None] - lat[None, :])
    q3 = np.cos(lat[:, None] + lat[None, :])

    dij = RRR * np.arccos(0.5 * ((1 + q1) * q2 - (1 - q1) * q3)) + 1
    return np.floor(dij)


def read_tsplib(filename, no_of_cities=None):
    """
    Read a TSPLIB TSP/ATSP file and return a distance matrix.
    """
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    header = {}
    for line in lines:
        if line in ["NODE_COORD_SECTION", "EDGE_WEIGHT_SECTION"]:
            break

        if ":" in line:
            key, value = line.split(":", 1)
            header[key.strip().upper()] = value.strip()

    dimension = int(header["DIMENSION"])
    edge_weight_type = header.get("EDGE_WEIGHT_TYPE", "").upper()
    edge_weight_format = header.get("EDGE_WEIGHT_FORMAT", "").upper()

    # ---------------------------------------------------------
    # Determine target number of cities
    # ---------------------------------------------------------
    if no_of_cities is None:
        no_of_cities = dimension

    if no_of_cities > dimension:
        raise ValueError(
            f"no_of_cities ({no_of_cities}) cannot be greater "
            f"than DIMENSION ({dimension})"
        )

    # ---------------------------------------------------------
    # EXPLICIT distance matrix
    # ---------------------------------------------------------
    if edge_weight_type == "EXPLICIT":
        try:
            start = lines.index("EDGE_WEIGHT_SECTION") + 1
        except ValueError:
            raise ValueError("EDGE_WEIGHT_SECTION not found")

        values = []
        for line in lines[start:]:
            if line in ["EOF", "DISPLAY_DATA_SECTION", "NODE_COORD_SECTION", "DEMAND_SECTION"]:
                break
            values.extend(map(float, line.split()))

        # Build matrix directly for target dimensions
        matrix = np.zeros((no_of_cities, no_of_cities))

        if edge_weight_format == "FULL_MATRIX":
            full_matrix = np.array(values).reshape(dimension, dimension)
            matrix = full_matrix[:no_of_cities, :no_of_cities]

        elif edge_weight_format == "UPPER_ROW":
            index = 0
            for i in range(dimension):
                for j in range(i + 1, dimension):
                    if i < no_of_cities and j < no_of_cities:
                        matrix[i, j] = values[index]
                        matrix[j, i] = values[index]
                    index += 1

        elif edge_weight_format == "LOWER_ROW":
            index = 0
            for i in range(dimension):
                for j in range(i):
                    if i < no_of_cities and j < no_of_cities:
                        matrix[i, j] = values[index]
                        matrix[j, i] = values[index]
                    index += 1

        elif edge_weight_format == "UPPER_DIAG_ROW":
            index = 0
            for i in range(dimension):
                for j in range(i, dimension):
                    if i < no_of_cities and j < no_of_cities:
                        matrix[i, j] = values[index]
                        matrix[j, i] = values[index]
                    index += 1

        elif edge_weight_format == "LOWER_DIAG_ROW":
            index = 0
            for i in range(dimension):
                for j in range(i + 1):
                    if i < no_of_cities and j < no_of_cities:
                        matrix[i, j] = values[index]
                        matrix[j, i] = values[index]
                    index += 1
        else:
            raise ValueError(f"Unsupported EDGE_WEIGHT_FORMAT: {edge_weight_format}")

    # ---------------------------------------------------------
    # Coordinate-based TSP
    # ---------------------------------------------------------
    else:
        try:
            start = lines.index("NODE_COORD_SECTION") + 1
        except ValueError:
            raise ValueError("No EDGE_WEIGHT_SECTION or NODE_COORD_SECTION found")

        coordinates = []
        for line in lines[start:]:
            if line in ["EOF", "DISPLAY_DATA_SECTION"]:
                break

            parts = line.split()
            if len(parts) < 3:
                continue

            x = float(parts[1])
            y = float(parts[2])

            if len(parts) >= 4:
                z = float(parts[3])
                coordinates.append([x, y, z])
            else:
                coordinates.append([x, y])

            # Stop parsing early if we already have the required cities
            if len(coordinates) == no_of_cities:
                break

        coordinates = np.array(coordinates)

        if edge_weight_type == "EUC_2D":
            x = coordinates[:, 0]
            y = coordinates[:, 1]
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            matrix = np.floor(np.sqrt(dx**2 + dy**2) + 0.5)

        elif edge_weight_type == "CEIL_2D":
            x = coordinates[:, 0]
            y = coordinates[:, 1]
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            matrix = np.ceil(np.sqrt(dx**2 + dy**2))

        elif edge_weight_type == "EUC_3D":
            x = coordinates[:, 0]
            y = coordinates[:, 1]
            z = coordinates[:, 2]
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dz = z[:, None] - z[None, :]
            matrix = np.floor(np.sqrt(dx**2 + dy**2 + dz**2) + 0.5)

        elif edge_weight_type == "ATT":
            x = coordinates[:, 0]
            y = coordinates[:, 1]
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            rij = np.sqrt((dx**2 + dy**2) / 10.0)
            tij = np.floor(rij)
            matrix = np.where(tij < rij, tij + 1, tij)

        elif edge_weight_type == "GEO":
            matrix = _geo_distance_matrix(coordinates)

        else:
            raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE: {edge_weight_type}")

    return np.rint(matrix).astype(int)