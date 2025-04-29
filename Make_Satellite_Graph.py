from skyfield.api import EarthSatellite

class SatelliteTracker:
    def __init__(self, tle_filepath):
        with open(tle_filepath) as f:
            tle_data = f.read()

        tle_lines = tle_data.splitlines()
        self.satellites = [EarthSatellite(tle_lines[i + 1], tle_lines[i + 2], tle_lines[i]) for i in
                           range(0, len(tle_lines), 3)]

    def generate_satellite_dict(self, time):
        sat_dict = {}
        for sat in self.satellites:
            geocentric = sat.at(time)
            eci_position = geocentric.position.km
            subpoint = geocentric.subpoint()
            lat = subpoint.latitude.degrees
            lon = subpoint.longitude.degrees
            alt = subpoint.elevation.km
            sat_name = sat.name
            orbit_altitude, orbit_number, sat_number = [int(s) for s in sat_name.split('_')[1:]]
            sat_dict[sat_name] = [eci_position,orbit_altitude , orbit_number, sat_number,lat,lon,alt]
        return sat_dict

    def get_max_orbit_number(self):
        max_orbit_number = 0
        for sat in self.satellites:
            sat_name = sat.name
            orbit_number = int(sat_name.split('_')[2])
            max_orbit_number = max(max_orbit_number, orbit_number)
        return max_orbit_number

    def get_max_satellite_number(self):
        max_satellite_number = 0
        for sat in self.satellites:
            sat_name = sat.name
            satellite_number = int(sat_name.split('_')[3])
            max_satellite_number = max(max_satellite_number, satellite_number)
        return max_satellite_number

    def generate_satellite_LLA_dict(self, time):
        sat_LLA_dict = {}
        for sat in self.satellites:
            geocentric = sat.at(time)
            subpoint = geocentric.subpoint()
            lat = subpoint.latitude.degrees
            lon = subpoint.longitude.degrees
            alt = subpoint.elevation.km

            sat_LLA_dict[sat.name]={"latitude": lat, "longitude": lon, "altitude": alt}
        return sat_LLA_dict

import networkx as nx

class SatelliteGraph:
    def __init__(self):
        pass

    def _distance(self, pos1, pos2):
        return sum((a - b) ** 2 for a, b in zip(pos1, pos2)) ** 0.5

    def build_graph_with_fixed_edges(self, satellite_tracker, time, pole=False):
        satellite_dict = satellite_tracker.generate_satellite_dict(time)
        graph = nx.Graph()
        graph.add_nodes_from(satellite_dict.keys())
        for sat_name, position in satellite_dict.items():
            graph.nodes[sat_name]['pos'] = position[0]
            graph.nodes[sat_name]['sequence_num'] = position[1:4]
            graph.nodes[sat_name]['pos_0'] = position[4:]
        max_orbit_number = satellite_tracker.get_max_orbit_number()
        max_satellite_number = satellite_tracker.get_max_satellite_number()

        for sat_name, sat_data in satellite_dict.items():
            same_orbit_neighbors = [f"Satellite_{sat_data[1]}_{sat_data[2]}_{sat_data[3] - 1 if sat_data[3] != 1 else max_satellite_number}",
                                    f"Satellite_{sat_data[1]}_{sat_data[2]}_{(sat_data[3] % max_satellite_number) + 1}"]
            for neighbor in same_orbit_neighbors:
                if neighbor in satellite_dict:
                    graph.add_edge(sat_name, neighbor,pos_a=graph.nodes[sat_name]['pos'], pos_b=graph.nodes[neighbor]['pos'])

        for sat_name, sat_data in satellite_dict.items():
            if pole:
                next_orbit_number = sat_data[2] + 1
            else:
                next_orbit_number = (sat_data[2] % max_orbit_number) + 1
            if not pole:
                next_orbit_satellite =f"Satellite_{sat_data[1]}_{next_orbit_number}_{sat_data[3] if next_orbit_number%2==0 else (sat_data[3] - 1 if sat_data[3] != 1 else max_satellite_number)}"
                graph.add_edge(sat_name, next_orbit_satellite,pos_a=graph.nodes[sat_name]['pos'], pos_b=graph.nodes[next_orbit_satellite]['pos'])

            elif next_orbit_number<= max_orbit_number:
                next_orbit_satellite = f"Satellite_{sat_data[1]}_{next_orbit_number}_{sat_data[3] if next_orbit_number%2==0 else (sat_data[3] - 1 if sat_data[3] != 1 else max_satellite_number)}"
                if abs(graph.nodes[sat_name]['pos'][2])<6000 and abs(graph.nodes[next_orbit_satellite]['pos'][2])<6000:
                    graph.add_edge(sat_name, next_orbit_satellite,pos_a=graph.nodes[sat_name]['pos'], pos_b=graph.nodes[next_orbit_satellite]['pos'])

        return graph
