from SatelliteNetworkSimulator_Computing import SatelliteNetworkSimulator_OnbardComputing
from SatelliteNetworkSimulation import SatelliteSimulation
from Make_Satellite_Graph import SatelliteTracker,SatelliteGraph
from skyfield.api import load
from Read_Ground_Imformation import extract_landmarks, get_connections_h3
from SatelliteNetworkSimulator_Beta import Logger
from Draw_Graph_Quiker import SatelliteVisualizer_geo
import random
import copy
import os

class SatelliteEnv(SatelliteSimulation):
    def __init__(self,mode,select_mode,q_net,epsilon,reward_factors,device,mission_possibility,poisson_rate,packet_frequency,computing_demand_factor,computing_demand_factor_2,size_after_computing_factor,size_after_computing_1,
                 begin_time, end_time, time_stride, tle_filepath, SOD_file_path, mean_interval_time,memory,
                 computing_ability, transmission_rate,downlink_rate,downstream_delays, packet_size_range, state_update_period, print_cycle,del_cycle, visualize=False,
                 print_info=False, save_log=False, show_detail=False,random_edges_del=0,random_nodes_del=0,update_cycle=1,save_training_data=None,
                 elevation_angle=45, pole=False):
        self.tracker = SatelliteTracker(tle_filepath)
        self.coordinates = extract_landmarks(SOD_file_path)
        self.graph_builder = SatelliteGraph()
        self.begin_time = begin_time
        self.end_time = end_time
        self.time_stride = time_stride
        self.mean_interval_time = mean_interval_time
        self.visualizer = SatelliteVisualizer_geo(edge_color=False) if visualize else None
        self.logger = Logger(detail=show_detail, save_log=save_log, verbose=print_info)
        self.ts = load.timescale()
        self.transmission_rate = transmission_rate
        self.downstream_delays=downstream_delays
        self.downlink_rate=downlink_rate
        self.packet_size_range = packet_size_range
        self.state_update_period = state_update_period
        self.random_edges_del = random_edges_del
        self.random_nodes_del = random_nodes_del
        self.elevation_angle = elevation_angle
        self.pole = pole
        self.statics = []
        self.time_acc = 0.0
        self.mode=mode
        self.select_mode=select_mode
        self.q_net=q_net
        self.epsilon=epsilon
        self.reward_factors = reward_factors
        self.device=device
        self.poisson_rate=poisson_rate
        self.packet_frequency=packet_frequency
        self.computing_demand_factor=computing_demand_factor
        self.computing_demand_factor_2=computing_demand_factor_2
        self.size_after_computing_1=size_after_computing_1
        self.size_after_computing_factor=size_after_computing_factor
        self.memory=memory
        self.computing_ability=computing_ability
        self.mission_possibility=mission_possibility

        self.update_cycle = update_cycle
        self.current_cycle = 0.0
        self.del_cycle = del_cycle
        self.del_update=True
        self.current_del_cycle = 0.0
        self.last_removed_nodes = set()
        self.last_removed_edges = []

        self.print_cycle = print_cycle
        self.current_print_cycle = 0.0
        self.iteration_counter = 0
        self.print_cycle_iterations = int(print_cycle / time_stride)
        self.save_training_data=save_training_data

        self.step_num=0
        self.rewards=[]

        self.current_graph = None
        self.connections = None

        self.reset(self.begin_time)
        self.current_time = self.begin_time

    def remove_random_edges(self,G, n,update=False):
        if n > G.number_of_edges():
            raise ValueError("Cannot remove more edges than exist in the graph")
        if update:
            self.last_removed_edges = random.sample(list(G.edges()), n)
        G.remove_edges_from(self.last_removed_edges)

        return G

    def remove_random_nodes(self,G, n,update=False):
        if n > G.number_of_nodes():
            raise ValueError("Cannot remove more nodes than exist in the graph")
        if update:
            self.last_removed_nodes = random.sample(list(G.nodes()), n)
        G.remove_nodes_from(self.last_removed_edges)

        return G

    def step(self,epsilon):
        self.step_num+=1

        self.current_del_cycle+=self.time_stride
        if self.del_cycle >= self.del_cycle:
            self.del_cycle += -self.del_cycle
            self.del_update=True
        self.current_cycle+=self.time_stride
        if self.current_cycle >= self.update_cycle:
            self.current_cycle += -self.update_cycle
            coordinates_s = self.tracker.generate_satellite_LLA_dict(self.time_from_str(self.current_time))
            connections = get_connections_h3(self.coordinates, coordinates_s, self.elevation_angle)
            old_nodes = set(self.simulator.graph.nodes())
            current_graph = self.graph_builder.build_graph_with_fixed_edges(self.tracker, self.time_from_str(self.current_time),pole=False)

            self.remove_random_nodes(current_graph, self.random_nodes_del,self.del_update)
            self.remove_random_edges(current_graph, self.random_edges_del,self.del_update)

            self.del_update = False
            new_nodes = set(current_graph.nodes())
            lost_nodes = old_nodes - new_nodes
            for landmark, satellites in connections.items():
                for lost_node in lost_nodes:
                    if lost_node in satellites:
                        connections[landmark].remove(lost_node)
            self.current_graph=current_graph
            self.connections=connections

        self.simulator.upgrade_all(self.current_graph, self.connections)
        for satellite in self.simulator.satellites:
            self.simulator.satellites[satellite].epsilon=epsilon

        self.simulator.run(self.time_stride)
        experiences=self.simulator.propagator.experiences
        self.simulator.propagator.experiences = []
        self.rewards.extend(self.simulator.propagator.final_rewards)
        self.iteration_counter += 1
        if self.iteration_counter >= self.print_cycle_iterations:
            self.iteration_counter = 0
            self.print_and_save_accumulated_data()
            self.current_print_cycle = 0.0
            self.rewards=[]

        self.time_acc += self.time_stride
        if self.time_acc >= 1.0:
            self.current_time = self.add_time_to_str(self.current_time, (0, int(self.time_acc)))
            self.time_acc -= int(self.time_acc)

        return experiences

    def reset(self,begin_time):
        self.statics= []
        self.begin_time=begin_time
        init_time = self.time_from_str(begin_time)
        current_graph = self.graph_builder.build_graph_with_fixed_edges(self.tracker, init_time, pole=self.pole)
        self.num_nodes = len(current_graph.nodes())
        coordinates_s = self.tracker.generate_satellite_LLA_dict(init_time)
        connections = get_connections_h3(self.coordinates, coordinates_s, self.elevation_angle)
        self.simulator = SatelliteNetworkSimulator_OnbardComputing(
            mode=self.mode,
            select_mode=self.select_mode,
            q_net=self.q_net,
            reward_factors = self.reward_factors,
            epsilon= self.epsilon,
            device= self.device,
            mission_possibility=self.mission_possibility,
            poisson_rate= self.poisson_rate,
            packet_frequency= self.packet_frequency,
            computing_demand_factor= self.computing_demand_factor,
            computing_demand_factor_2=self.computing_demand_factor_2,
            size_after_computing_factor= self.size_after_computing_factor,
            size_after_computing_1=self.size_after_computing_1,
            graph=current_graph,
            landmarks=connections,
            mean_interval_time=self.mean_interval_time,
            memory=self.memory,
            computing_ability=self.computing_ability,
            transmission_rate=self.transmission_rate,
            downstream_delays=self.downstream_delays,
            downlink_rate=self.downlink_rate,
            packet_size_range=self.packet_size_range,
            state_update_period=self.state_update_period,
            logger=self.logger)
        self.current_time = self.begin_time
        self.time_acc = 0.0
        self.current_cycle =0.0
        self.current_print_cycle=0.0
        self.current_del_cycle = 0.0
        self.iteration_counter = 0
        self.del_update = True
        old_nodes = set(self.simulator.graph.nodes())
        current_graph = self.graph_builder.build_graph_with_fixed_edges(self.tracker,self.time_from_str(self.current_time), pole=False)

        self.remove_random_nodes(current_graph, self.random_nodes_del, True)
        self.remove_random_edges(current_graph, self.random_edges_del, True)
        new_nodes = set(current_graph.nodes())
        lost_nodes = old_nodes - new_nodes
        for landmark, satellites in connections.items():
            for lost_node in lost_nodes:
                if lost_node in satellites:
                    connections[landmark].remove(lost_node)
        self.current_graph = current_graph
        self.connections = connections

    def render(self):
        self.visualize(self.simulator.graph, self.current_time, self.simulator)

    def visualize(self,current_graph,current_time,simulator):
        system_state=simulator.get_system_state()

        G_draw=current_graph.copy()
        self.update_node_colors(G_draw, system_state)

        for landmark, pos_value in self.coordinates.items():
            G_draw.add_node(landmark)
            G_draw.nodes[landmark]['pos_0'] = [pos_value["latitude"],pos_value ["longitude"], pos_value["altitude"]]
            G_draw.nodes[landmark]['color'] = 'purple'
        self.visualizer.draw_graph(G_draw)

    def show_satellite_computing_time(self):
        satellite_computing_times={}
        for satellite in self.simulator.satellites:
            satellite_computing_times[satellite]=self.simulator.satellites[satellite].computing_time
        self.print_and_save(str(satellite_computing_times))

    def print_and_save(self, message):
        print(message)
        if self.save_training_data:
            file_path = os.path.join('./training_process_data', self.save_training_data)
            with open(file_path, 'a') as file:
                file.write(message + '\n')

    def print_and_save_accumulated_data(self):
        self.print_and_save(f"====== step {self.step_num} ======")
        self.print_and_save(f"====== {self.current_time} ======")
        current_statics_snapshot = copy.deepcopy(self.simulator.statics_datas)
        if self.statics:
            current_statics = {k: current_statics_snapshot[k] - self.statics[-1][k] for k in current_statics_snapshot}
        else:
            current_statics = current_statics_snapshot
        self.statics.append(current_statics_snapshot)

        d = current_statics
        packet_loss_rates = (d['Lost_relay_0'] + d['Lost_relay_1'] + d['Lost_upload']) / (d['Lost_relay_0'] + d['Lost_relay_1'] + d['Lost_upload'] + d['Reached_0'] + d['Reached_1']) if d['Lost_relay_0'] + d['Lost_relay_1'] + d['Lost_upload'] + d['Reached_0'] + d['Reached_1'] > 0 else None
        average_delays = (d['Total_delay_0'] + d['Total_delay_1']) / (d['Reached_0'] + d['Reached_1']) if d['Reached_0'] + d['Reached_1'] > 0 else None
        average_hops = (d['Total_hops_0'] + d['Total_hops_1']) / (d['Reached_0'] + d['Reached_1']) if d['Reached_0'] + d['Reached_1'] > 0 else None
        average_computing_ratio = d['Is_computing'] / self.num_nodes / (self.print_cycle_iterations)
        average_computing_waiting_time = (d['Computing_waiting_time']) / (d['Reached_0'] + d['Reached_1']) if d['Reached_0'] + d['Reached_1'] > 0 else None

        self.print_and_save(f"current_statics: {current_statics}")
        self.print_and_save(f"Packet loss rate: {'{:.2%}'.format(packet_loss_rates) if packet_loss_rates is not None else 'None'}")
        self.print_and_save(f"Average delay for successful transmissions: {'{:.3f} seconds'.format(average_delays) if average_delays is not None else 'None'}")
        self.print_and_save(f"Average hop count for successful transmissions: {'{:.3f} hops'.format(average_hops) if average_hops is not None else 'None'}")
        self.print_and_save(f"Proportion of satellites in computation: {'{:.2%}'.format(average_computing_ratio) if average_computing_ratio is not None else 'None'}")
        self.print_and_save(f"Average waiting time for computing: {'{:.3f} seconds'.format(average_computing_waiting_time) if average_computing_waiting_time is not None else 'None'}")

        rewards = sum(self.rewards) / len(self.rewards) if len(self.rewards) > 0 else None
        self.print_and_save(f"Average ending reward: {rewards if rewards is not None else 'None'}")
