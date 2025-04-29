import math
import h3


def extract_landmarks(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    landmarks = {}
    i = 0
    j = 0

    lines_list = []
    for line in lines[2:]:
        lines_list.append(line.strip())
        if ":" in line:
            break

    landmark_name = ','.join(lines_list)
    landmark_names = [name.strip() for name in landmark_name.split(':')[0].split(",")]
    while i < len(lines):
        line = lines[i]
        if "---------    ---------    --------" in line:
            i += 1
            coords_line = lines[i]
            coords = coords_line.split()
            if len(coords) == 3:
                latitude, longitude, altitude = map(float, coords)
                landmarks[landmark_names[j]] = {"latitude": latitude, "longitude": longitude, "altitude": altitude}
                j += 1
        i += 1
    return landmarks


def to_cartesian(lat, lon, alt=0):
    R = 6371
    lat, lon = math.radians(lat), math.radians(lon)
    x = (R + alt) * math.cos(lat) * math.cos(lon)
    y = (R + alt) * math.cos(lat) * math.sin(lon)
    z = (R + alt) * math.sin(lat)
    return x, y, z


def max_distance(elevation_angle, orbital_height):
    elevation_angle = math.radians(elevation_angle)
    earth_radius = 6371
    distance = math.sqrt(
        orbital_height ** 2 + 2 * orbital_height * earth_radius + (math.cos(elevation_angle) * earth_radius) ** 2
        ) - math.cos(elevation_angle) * earth_radius
    return distance


def to_h3_index(lat, lon, resolution=0):
    return h3.geo_to_h3(lat, lon, resolution)


def get_h3_neighbors(h3_index):
    return h3.k_ring(h3_index, 1)


def get_connections_h3(ground_users, satellites, elevation_angle):
    connections = {}
    cell_satellites = {}

    for sat_name, sat_position in satellites.items():
        h3_index = to_h3_index(sat_position["latitude"], sat_position["longitude"])
        if h3_index not in cell_satellites:
            cell_satellites[h3_index] = []
        cell_satellites[h3_index].append(
            (sat_name, to_cartesian(sat_position["latitude"], sat_position["longitude"], sat_position["altitude"])))

    max_dist = max_distance(elevation_angle, satellites[next(iter(satellites))]["altitude"])
    for user_name, user_position in ground_users.items():
        user_h3_index = to_h3_index(user_position["latitude"], user_position["longitude"])
        user_cartesian = to_cartesian(user_position["latitude"], user_position["longitude"],
                                      user_position.get("altitude", 0))

        reachable_satellites = set()
        for neighbor_h3_index in get_h3_neighbors(user_h3_index):
            if neighbor_h3_index in cell_satellites:
                for sat_name, sat_cartesian in cell_satellites[neighbor_h3_index]:
                    actual_distance = math.sqrt(
                        (sat_cartesian[0] - user_cartesian[0]) ** 2 +
                        (sat_cartesian[1] - user_cartesian[1]) ** 2 +
                        (sat_cartesian[2] - user_cartesian[2]) ** 2)
                    if actual_distance <= max_dist:
                        reachable_satellites.add(sat_name)

        connections[user_name] = list(reachable_satellites)

    return connections
