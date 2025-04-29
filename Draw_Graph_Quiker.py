import networkx as nx
import plotly.graph_objects as go
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

class SatelliteVisualizer:
    def __init__(self, edge_color=False):
        self.edge_color = edge_color

    def draw_graph(self, graph):
        pos = nx.get_node_attributes(graph, 'pos')

        Xn=[pos[k][0] for k in pos]
        Yn=[pos[k][1] for k in pos]
        Zn=[pos[k][2] for k in pos]

        node_color = nx.get_node_attributes(graph, 'color')
        node_trace = go.Scatter3d(x=Xn,
                                  y=Yn,
                                  z=Zn,
                                  mode='markers',
                                  marker=dict(size=6,
                                              color=list(node_color.values()),
                                              colorscale='Viridis',
                                              opacity=1))

        data=[node_trace]

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = 6335 * np.outer(np.cos(u), np.sin(v))
        y = 6335 * np.outer(np.sin(u), np.sin(v))
        z = 6335 * np.outer(np.ones(np.size(u)), np.cos(v))
        earth = go.Surface(x=x, y=y, z=z,colorscale=[(0, 'green'), (1, 'green')], showscale=False)

        data.append(earth)

        for e in graph.edges(data=True):
            Xe=[pos[e[0]][0], pos[e[1]][0]]
            Ye=[pos[e[0]][1], pos[e[1]][1]]
            Ze=[pos[e[0]][2], pos[e[1]][2]]

            color_node1 = graph.nodes[e[0]].get('color', 'rgb(125,125,125)')
            color_node2 = graph.nodes[e[1]].get('color', 'rgb(125,125,125)')
            if self.edge_color and color_node1 == color_node2:
                line_color = color_node1
            else:
                line_color = 'rgb(125,125,125)'

            edge_trace = go.Scatter3d(x=Xe,
                                      y=Ye,
                                      z=Ze,
                                      mode='lines',
                                      line=dict(color=line_color, width=1),
                                      hoverinfo='none')
            data.append(edge_trace)

        fig = go.Figure(data=data)
        fig.show()

class SatelliteVisualizer_geo:
    def __init__(self, edge_color=False):
        self.edge_color = edge_color
        self.world = gpd.read_file(r'D:\Project\PythonProject\Satellite_simulation_beta\ne_data\ne_50m_admin_0_countries_lakes.dbf')

    def lat_lon_alt_to_cartesian(self,lat, lon, alt):
        R = 6371
        h = alt
        phi = np.deg2rad(90 - lat)
        theta = np.deg2rad(lon)

        x = (R + h) * np.sin(phi) * np.cos(theta)
        y = (R + h) * np.sin(phi) * np.sin(theta)
        z = (R + h) * np.cos(phi)

        return x, y, z

    def draw_graph(self, graph):
        Xn, Yn, Zn = {}, {}, {}

        for node, attr in graph.nodes(data=True):
            lat, lon, alt = attr['pos_0']
            x, y, z = self.lat_lon_alt_to_cartesian(lat, lon, alt)
            Xn[node]=x
            Yn[node]=y
            Zn[node]=z

        node_color = nx.get_node_attributes(graph, 'color')
        node_trace = go.Scatter3d(x=list(Xn.values()),
                                  y=list(Yn.values()),
                                  z=list(Zn.values()),
                                  mode='markers',
                                  marker=dict(size=6,
                                              color=list(node_color.values()),
                                              colorscale='Viridis',
                                              opacity=1))

        data = [node_trace]

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = 6371 * np.outer(np.cos(u), np.sin(v))
        y = 6371 * np.outer(np.sin(u), np.sin(v))
        z = 6371 * np.outer(np.ones(np.size(u)), np.cos(v))
        earth = go.Surface(x=x, y=y, z=z,colorscale=[(0, 'lightblue'), (1, 'lightblue')], showscale=False)

        data.append(earth)

        for _, row in self.world.iterrows():
            geom = row['geometry']
            if isinstance(geom, Polygon):
                lons,lats = geom.exterior.xy
                Xc, Yc, Zc = [], [], []
                for lat, lon in zip(lats, lons):
                    x, y, z = self.lat_lon_alt_to_cartesian(lat, lon, 0)
                    Xc.append(x)
                    Yc.append(y)
                    Zc.append(z)
                data.append(go.Scatter3d(x=Xc, y=Yc, z=Zc, mode='lines', line=dict(width=1, color="black")))
            elif isinstance(geom, MultiPolygon):
                for poly in geom.geoms:
                    lons,lats = poly.exterior.xy
                    Xc,Yc,Zc=[],[],[]
                    for lat, lon in zip(lats, lons):
                        x, y, z = self.lat_lon_alt_to_cartesian(lat, lon, 0)
                        Xc.append(x)
                        Yc.append(y)
                        Zc.append(z)
                    data.append(go.Scatter3d(x=Xc, y=Yc, z=Zc, mode='lines', line=dict(width=1, color="black")))

        for e in graph.edges(data=True):
            Xe=[Xn[e[0]], Xn[e[1]]]
            Ye=[Yn[e[0]], Yn[e[1]]]
            Ze=[Zn[e[0]], Zn[e[1]]]

            color_node1 = graph.nodes[e[0]].get('color', 'rgb(125,125,125)')
            color_node2 = graph.nodes[e[1]].get('color', 'rgb(125,125,125)')
            if self.edge_color and color_node1 == color_node2:
                line_color = color_node1
            else:
                line_color = 'black'

            edge_trace = go.Scatter3d(x=Xe,
                                      y=Ye,
                                      z=Ze,
                                      mode='lines',
                                      line=dict(color=line_color, width=1),
                                      hoverinfo='none')
            data.append(edge_trace)

        fig = go.Figure(data=data)
        fig.update_layout(
            scene=dict(
                xaxis=dict(showbackground=False, visible=False),
                yaxis=dict(showbackground=False, visible=False),
                zaxis=dict(showbackground=False, visible=False)
            )
        )
        fig.show()
