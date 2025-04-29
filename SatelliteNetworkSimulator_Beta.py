import simpy
import networkx as nx
import numpy as np
import logging
import random
from datetime import datetime


def add_suffix_to_graph(G, suffix):
    G_new = type(G)()
    for node in G.nodes():
        G_new.add_node(f"{node}{suffix}")
    for u, v, data in G.edges(data=True):
        G_new.add_edge(f"{u}{suffix}", f"{v}{suffix}", **data)
    return G_new

class Logger():
    def __init__(self,detail,save_log,verbose,num=None):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.detail=detail
        self.save_log=save_log
        self.num=num
        if save_log:
            if num:
                if detail:
                    self.log_file = f"simulation_{num}_{current_time}_detail.log"
                else:
                    self.log_file = f"simulation_{num}_{current_time}.log"
            else:
                if detail:
                    self.log_file = f"simulation_{current_time}_detail.log"
                else:
                    self.log_file = f"simulation_{current_time}.log"
            if num:
                self.logger = logging.getLogger(f'SimulationLogger_{num}')
            else:
                self.logger = logging.getLogger(f'SimulationLogger')
            self.logger.setLevel(logging.INFO)
            handler = logging.FileHandler(self.log_file)
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)
        self.verbose=verbose

    def log(self, message,detail=False):
        if not detail or (self.detail and detail):
            if self.verbose:
                print(message)
            if self.save_log:
                self.logger.info(message)

class Packet():
    def __init__(self,source,destination,creation_time,size):
        self.source=source
        self.destination=destination
        self.creation_time=creation_time
        self.computing_waiting_time=0
        self.hops = 0
        self.size=size
        self.information=[]
        self.routing=None
        self.computing_node=None

    def extra_information(self,information):
        self.information=information

class Propagator():
    def __init__(self,env,graph,logger,satellites,statics_data={},global_graph=False):
        self.env=env
        self.logger=logger
        self.propagation_speed=3e5
        self.global_graph=global_graph
        self.graph = graph
        self.node_names=list(graph.nodes)
        self.node_positions = {node: graph.nodes[node]['pos'] for node in graph.nodes}
        self.node_neighbors={node: list(graph.neighbors(node)) for node in self.node_names}
        self.propagation_delays = {}
        self.calculate_delays()
        self.satellites=satellites
        self.statics_data=statics_data

    def _distance(self, node1, node2):
        a, b = self.node_positions[node1], self.node_positions[node2]
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    def calculate_delays(self):
        self.propagation_delays = {}
        for node1, node2 in self.node_neighbors.items():
            for neighbor in node2:
                distance = self._distance(node1, neighbor)
                propagation_delay = distance / self.propagation_speed
                self.propagation_delays[(node1, neighbor)] = propagation_delay
                self.propagation_delays[(neighbor, node1)] = propagation_delay
                if self.global_graph:
                    self.graph[node1][neighbor]['propagation_weight']= propagation_delay
                    self.graph[node1][neighbor]['propagation_weight'] = propagation_delay
        if self.global_graph:
            for edge in self.graph.edges():
                node1, node2 = edge
                if edge in self.propagation_delays:
                    self.graph[node1][node2]['missing']=0
                else:
                    self.graph[node1][node2]['missing'] =1

    def update(self,graph):
        self.node_names=list(graph.nodes)
        self.node_positions = {node: graph.nodes[node]['pos'] for node in graph.nodes}
        self.node_neighbors={node: list(graph.neighbors(node)) for node in self.node_names}
        self.calculate_delays()

    def propagate(self,node,next_hop,packet):
        if (node, next_hop) in self.propagation_delays:
            yield self.env.timeout(self.propagation_delays[(node, next_hop)])
            if next_hop in self.node_names:
                success = self.satellites[next_hop].push_forward(packet)
                if success:
                    self.logger.log(f"Time {self.env.now:.3f}: {next_hop}: Packet {(packet.source,packet.destination)} received by router. Transmission length: {self.satellites[next_hop].current_queue_length}.",detail=True)
                else:
                    if 'Lost_relay' in self.statics_data:
                        self.statics_data['Lost_relay'] += 1
                    self.logger.log(f"Time {self.env.now:.3f}: {next_hop}: Routing queue is full, discarding packet {(packet.source,packet.destination)}.")
            else:
                if 'Lost_relay' in self.statics_data:
                    self.statics_data['Lost_relay'] += 1
                self.logger.log(f"Time {self.env.now:.3f}: {next_hop} is missed, dropped 1 packet.")
        else:
            if 'Lost_relay' in self.statics_data:
                self.statics_data['Lost_relay'] += 1
            self.logger.log(f"Time {self.env.now:.3f}: connection {(node, next_hop)} is missed, dropped 1 packet.")

    def send_state(self, node, neighbor,value):
        if (node, neighbor) in self.propagation_delays:
            yield self.env.timeout(self.propagation_delays[(node, neighbor)])
            if neighbor in self.node_names:
                if node in self.satellites[neighbor].neighbors:
                    self.satellites[neighbor].neighbor_states[node]=value
                    self.satellites[neighbor].last_heartbeat[node]=self.env.now
                else:
                    self.satellites[neighbor].add_neighbor(neighbor)
                    self.satellites[neighbor].neighbor_states[node] = value
            else:
                self.logger.log(f"Time {self.env.now:.3f}: {neighbor} is missed, state update failed.")
        else:
            self.logger.log(f"Time {self.env.now:.3f}: connection {(node, neighbor)} is missed, state update failed.")

    def send_adjacency_table(self, node, neighbor,table):
        if (node, neighbor) in self.propagation_delays:
            yield self.env.timeout(self.propagation_delays[(node, neighbor)])
            if neighbor in self.node_names:
                if node in self.satellites[neighbor].neighbors:
                    self.satellites[neighbor].update_adjacency_table(table)
                else:
                    self.satellites[neighbor].add_neighbor(neighbor)
                    self.satellites[neighbor].update_adjacency_table(table)
            else:
                self.logger.log(f"Time {self.env.now:.3f}: {neighbor} is missed, state update failed.")
        else:
            self.logger.log(f"Time {self.env.now:.3f}: connection {(node, neighbor)} is missed, state update failed.")

class Satellite():
    def __init__(self,env,name,neighbors,queue_length,transmission_rate,state_update_period,logger,statics_data={},processing_time=1e-9,heartbeat_timeout=0.5):
        self.name=name
        self.neighbors= neighbors
        self.env=env
        self.queue_length=queue_length
        self.transmission_rate=transmission_rate
        self.state_update_period=state_update_period
        self.logger=logger
        self.transmission_queue = {neighbor: simpy.Store(self.env) for neighbor in self.neighbors}
        self.transmission_length ={neighbor: 0 for neighbor in self.neighbors}
        self.forward_queue = simpy.Store(self.env)
        self.current_queue_length=0
        self.active=True
        self.routing_tables={}
        self.neighbor_states={neighbor: 0 for neighbor in self.neighbors}
        self.propagator=None
        self.statics_data=statics_data
        self.processing_time=processing_time
        self.heartbeat_timeout = heartbeat_timeout
        self.last_heartbeat = {neighbor: env.now for neighbor in self.neighbors}
        self.hops={}
        self.adjacency_table = {self.name: (self.neighbors, self.env.now)}

    def set_propagator(self,propagator):
        self.propagator=propagator

    def push_forward(self,packet):
        if self.current_queue_length < self.queue_length:
            self.forward_queue.put(packet)
            return True
        else:
            return False
    def push_transmission(self,neighbor,packet):
        if self.current_queue_length < self.queue_length:
            self.current_queue_length += 1
            self.transmission_length[neighbor]+=1
            self.transmission_queue[neighbor].put(packet)
            return True
        else:
            return False
    def pop_transmission(self,neighbor):
        packet = yield self.transmission_queue[neighbor].get()
        self.current_queue_length-=1
        self.transmission_length[neighbor]-=1
        return packet

    def forward_packet(self):
        while self.active:
            packet = yield self.forward_queue.get()
            packet.hops += 1
            source, destination = packet.source, packet.destination
            yield self.env.timeout(self.processing_time)
            if not self.active:
                if 'Lost_relay' in self.statics_data:
                    self.statics_data['Lost_relay'] += 1
                self.logger.log(f"Time {self.env.now:.3f}: {self.name} is missed, dropped 1 packet")
                break
            if destination != self.name:
                if destination in self.routing_tables:
                    next_hop = random.choice(self.routing_tables[destination][0])
                    if next_hop in self.neighbors:
                        success =self.push_transmission(next_hop, packet)
                        if not success:
                            self.logger.log(f"Time {self.env.now:.3f}: {packet} is blocked because of congestion.")
                    else:
                        if 'Lost_relay' in self.statics_data:
                            self.statics_data['Lost_relay'] += 1
                        self.logger.log(f"Time {self.env.now:.3f}: {next_hop} is missed, dropped 1 packet")
                else:
                    if 'Lost_relay' in self.statics_data:
                        self.statics_data['Lost_relay'] += 1
                    self.logger.log(f"Time {self.env.now:.3f}: {destination} is missed, dropped 1 packet")
            else:
                if 'Reached' in self.statics_data:
                    self.statics_data['Reached'] += 1
                if 'Total_hops' in self.statics_data:
                    self.statics_data['Total_hops'] += packet.hops
                if 'Total_delay' in self.statics_data:
                    self.statics_data['Total_delay'] += self.env.now - packet.creation_time
                self.logger.log(f"Time {self.env.now:.3f}: Packet {(source, destination)} reached its destination {self.name}.")

    def transmit_packet(self,neighbor):
        while self.active:
            packet = yield self.env.process(self.pop_transmission(neighbor))
            yield self.env.timeout(packet.size / self.transmission_rate)
            if neighbor not in self.neighbors or not self.active:
                if 'Lost_relay' in self.statics_data:
                    self.statics_data['Lost_relay'] += 1
                self.logger.log(f"Time {self.env.now:.3f}: transmission stopped, dropped 1 packet")
                break
            self.logger.log(f"Time {self.env.now:.3f}: {self.name}: Packet {(packet.source,packet.destination)} departed. Transmission length: {self.current_queue_length}",detail=True)
            self.env.process(self.propagator.propagate(self.name,neighbor, packet))


    def update_adjacency_dict_for_bfs(self):
        new_dict = self.adjacency_table.copy()
        for node, (neighbors, _) in self.adjacency_table.items():
            for neighbor in neighbors:
                if node not in self.adjacency_table[neighbor][0]:
                    if self.adjacency_table[node][1] > self.adjacency_table[neighbor][1]:
                        new_dict[neighbor] = (new_dict[neighbor][0] + [node], new_dict[neighbor][1])
                    else:
                        new_dict[node][0].remove(neighbor)
        return new_dict

    def build_routing_table(self):
        result_dict = {}
        queue = [(neighbor, [self.name, neighbor], 1) for neighbor in self.adjacency_table[self.name][0]]
        while queue:
            (node, path, hops) = queue.pop(0)
            if node not in result_dict:
                result_dict[node] = ([path[1]], hops)
                queue.extend((neighbor, path + [neighbor], hops + 1) for neighbor in self.adjacency_table[node][0] if
                             neighbor not in path)
            elif result_dict[node][1] == hops:
                result_dict[node][0].append(path[1])
        self.routing_tables = result_dict

    def add_neighbor(self,neighbor):
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)
            self.transmission_queue[neighbor] = simpy.Store(self.env)
            self.transmission_length[neighbor] = 0
            self.neighbor_states[neighbor] = 0
            self.last_heartbeat[neighbor] = self.env.now
            self.adjacency_table [self.name]=(self.neighbors, self.env.now)
            self.adjacency_table_exchanger()
            self.env.process(self.monitor_single_neighbor(neighbor))
            self.env.process(self.transmit_packet(neighbor))

    def del_neighbor(self,neighbor):
        if self.active:
            if neighbor in self.neighbors:
                while self.transmission_queue[neighbor].items:
                    packet = yield self.env.process(self.pop_transmission(neighbor))
                    success= self.push_forward(packet)
                    if not success:
                        if 'Lost_relay' in self.statics_data:
                            self.statics_data['Lost_relay'] += 1
                    self.logger.log(f"Time {self.env.now:.3f}: {packet} is dropped because of satellite missing.")
                self.neighbors.remove(neighbor)
                del self.transmission_queue[neighbor]
                del self.transmission_length[neighbor]
                del self.neighbor_states[neighbor]
                self.adjacency_table[self.name]=(self.neighbors, self.env.now)
                self.update_adjacency_dict_for_bfs()
                self.build_routing_table()
                self.adjacency_table_exchanger()
                return True
            else:
                return False
        else:
            return False
    def state_exchanger(self):
        while self.active:
            yield self.env.timeout(self.state_update_period)
            if not self.active:
                break
            for neighbor in self.neighbors:
                self.env.process(self.propagator.send_state(self.name, neighbor,self.current_queue_length))

    def adjacency_table_exchanger(self):
        for neighbor in self.neighbors:
            self.env.process(self.propagator.send_adjacency_table(self.name,neighbor,self.adjacency_table))

    def update_adjacency_dict(self, new_dict):
        updated = False
        for key, value in new_dict.items():
            if key not in self.adjacency_table:
                self.adjacency_table[key] = value
                updated = True
            else:
                _, old_time = self.adjacency_table[key]
                _, new_time = value
                if new_time > old_time:
                    self.adjacency_table[key] = value
                    updated = True
        return updated

    def update_adjacency_table(self, table):
        if self.update_adjacency_dict(table):
            self.update_adjacency_dict_for_bfs()
            self.build_routing_table()
            self.adjacency_table_exchanger()

    def monitor_single_neighbor(self, neighbor):
        while self.active:
            timeout_duration = self.heartbeat_timeout - (self.env.now - self.last_heartbeat[neighbor])
            if timeout_duration <= 0.01:
                yield self.env.process(self.del_neighbor(neighbor))
                break
            else:
                yield self.env.timeout(timeout_duration)

    def self_missing(self):
        self.active = False
        for neighbor in self.neighbors:
            while self.transmission_queue[neighbor].items:
                packet = yield self.env.process(self.pop_transmission(neighbor))
                if 'Lost_relay' in self.statics_data:
                    self.statics_data['Lost_relay'] += 1
                self.logger.log(f"Time {self.env.now:.3f}: {packet} is dropped because of satellite missing.")
            while self.forward_queue.items:
                packet = yield self.forward_queue.get()
                if 'Lost_relay' in self.statics_data:
                    self.statics_data['Lost_relay'] += 1
                self.logger.log(f"Time {self.env.now:.3f}: {packet} is dropped because of satellite missing.")

    def all_start(self):
        self.env.process(self.forward_packet())
        for neighbor in self.neighbors:
            self.env.process(self.transmit_packet(neighbor))
            self.env.process(self.monitor_single_neighbor(neighbor))
        self.env.process(self.state_exchanger())

class SatelliteNetworkSimulator:
    def __init__(self, graph,landmarks,mean_interarrival_time,queue_length,transmission_rate,packet_size,state_update_period,logger):
        self.env = simpy.Environment()
        self.graph =graph
        self.queue_length=queue_length
        self.logger=logger
        self.transmission_rate=transmission_rate
        self.state_update_period=state_update_period
        self.statics_data = {'Total': 0, 'Reached': 0, 'Lost_upload': 0, 'Lost_relay': 0, 'Total_delay': 0, 'Total_hops': 0}
        self.satellite_names=[node for node in self.graph.nodes]
        self.satellites={node : Satellite(self.env,node,list(self.graph.neighbors(node)),queue_length,transmission_rate,state_update_period,logger,self.statics_data) for node in self.graph.nodes}
        self.propagator = Propagator(self.env, graph, logger, self.satellites,self.statics_data)
        self.landmarks=landmarks
        self.mean_interarrival_time=mean_interarrival_time
        self.size = packet_size
        for satellite in self.satellites:
            self.satellites[satellite].adjacency_table=self.extract_adjacency_dict()
            self.satellites[satellite].set_propagator(self.propagator)
            self.satellites[satellite].build_routing_table()

    def extract_adjacency_dict(self):
        adjacency_dict = {}
        for node in self.satellite_names:
            neighbors = list(self.graph.neighbors(node))
            adjacency_dict[node] = (neighbors, self.env.now)
        return adjacency_dict

    def generate_traffic(self, landmark):
        def has_common_elements(list1, list2):
            set1 = set(list1)
            set2 = set(list2)
            common_elements = set1.intersection(set2)
            return len(common_elements) > 0

        while landmark in self.landmarks:
            interarrival_time = random.expovariate(1.0 / self.mean_interarrival_time)
            yield self.env.timeout(interarrival_time)
            if not landmark in self.landmarks:
                break
            if self.landmarks[landmark]:
                sources = self.landmarks[landmark]
            else:
                self.logger.log(f"Time {self.env.now:.3f}: {landmark} has no connections, packets failed to generate.")
                continue
            destination_landmark = landmark
            while destination_landmark == landmark:
                temp_landmark = random.choice(list(self.landmarks))
                temp_destinations = self.landmarks[temp_landmark]
                if self.landmarks[temp_landmark] and not has_common_elements(sources, temp_destinations):
                    destination_landmark = temp_landmark
                    destinations = temp_destinations
            min_hops=np.inf
            min_hops_pairs=[]
            for source in sources:
                for destination in destinations:
                    if destination in self.satellites[source].routing_tables:
                        hops= self.satellites[source].routing_tables[destination][1]
                    else:
                        hops = np.inf
                    if hops < min_hops:
                        min_hops = hops
                        min_hops_pairs = [(source,destination)]
                    elif hops == min_hops:
                        min_hops_pairs.append((source,destination))
            if min_hops_pairs:
                source,destination=random.choice(min_hops_pairs)
            else:
                self.logger.log(f"Time {self.env.now:.3f}: connection between {(landmark,destination_landmark)} is missed, packet failed to generate.")
                continue
            packet = Packet(source,destination,self.env.now,self.size)
            if 'Total' in self.statics_data:
                self.statics_data['Total'] += 1
            self.logger.log(f"Time {self.env.now:.3f}: {source}: Packet generated: {(source,destination)}.")
            if source in self.satellite_names:
                success = self.satellites[source].push_forward(packet)
                if success:
                    self.logger.log(f"Time {self.env.now:.3f}: {source}: Packet {(packet.source, packet.destination)} received by router. Transmission length: {self.satellites[source].current_queue_length}.",detail=True)
                else:
                    if 'Lost_upload' in self.statics_data:
                        self.statics_data['Lost_upload'] += 1
                    self.logger.log(f"Time {self.env.now:.3f}: {source}: Routing queue is full, discarding packet {(packet.source, packet.destination)}.")
            else:
                self.logger.log(f"Time {self.env.now:.3f}: {source} is missed, packet failed to generate.")
                continue

    def get_system_state(self):
        total_queue_usage = {}
        for node in self.satellite_names:
            total_usage = self.satellites[node].current_queue_length
            average_usage = total_usage / self.queue_length
            total_queue_usage[node] = average_usage
        return total_queue_usage
    def upgrade_all(self,graph,landmarks):
        old_landmarks=set(self.landmarks.keys())
        new_landmarks=set(landmarks.keys())
        self.landmarks=landmarks
        new_nodes = set(graph.nodes())
        old_nodes = set(self.graph.nodes())
        new_edges = set(graph.edges())
        old_edges = set(self.graph.edges())
        self.satellite_names=[node for node in graph]
        self.graph=graph
        self.propagator.update(graph)
        for node in new_nodes - old_nodes:
            self.satellites[node]=Satellite(self.env, node, list(self.graph.neighbors(node)), self.queue_length, self.transmission_rate,self.state_update_period, self.logger, self.statics_data)
            self.satellites[node].set_propagator(self.propagator)
            self.satellites[node].all_start()
            self.satellites[node].adjacency_table_exchanger()
        for node in old_nodes - new_nodes:
            self.env.process(self.satellites[node].self_missing())
            del self.satellites[node]
        for edge in new_edges - old_edges:
            node, neighbor = edge
            self.satellites[node].add_neighbor(neighbor)
            self.satellites[neighbor].add_neighbor(node)
        for landmark in new_landmarks-old_landmarks:
            self.env.process(self.generate_traffic(landmark))

    def clear_statics(self):
        for statics in self.statics_data:
            self.statics_data[statics]=0

    def run(self, duration):
        if self.env.now==0:
            for landmark in self.landmarks:
                self.env.process(self.generate_traffic(landmark))
            for satellite in self.satellites:
                self.satellites[satellite].all_start()
        self.env.run(until=self.env.now+duration)
        #print(self.statics_data)










