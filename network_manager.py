import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class NetworkManager:
    def __init__(self):
        self.G = nx.Graph()  # Initialize an empty graph
        self.H = nx.Graph()  # Initialize an empty graph
        self.Z = nx.Graph()  # Initialize an empty graph
        print("Initialized an empty graph.")

    # Add an agent to the network
    def add_agent(self, agent_id, agent_type):
        self.G.add_node(agent_id, type=agent_type)
        if agent_type == 'human':
            self.H.add_node(agent_id, type=agent_type)
        elif agent_type == 'zombie':
            self.Z.add_node(agent_id, type=agent_type)

    def remove_agent(self, agent_id):
        if self.G.has_node(agent_id):
            self.G.nodes[agent_id]['status'] = 'dead'
            # print(f"Changed status of agent {agent_id} to 'dead'.")
        else:
            print(f"Node {agent_id} does not exist in the graph.")

    def add_interaction(self, agent1_id, agent2_id, interaction_type):
        self.G.add_edge(agent1_id, agent2_id, interaction=interaction_type)
        print(f"Added interaction between {agent1_id} and {agent2_id}.")

    def perform_analysis(self):
        # Calculate degree centrality
        degree_centrality = nx.degree_centrality(self.G)

        # Calculate betweenness centrality
        betweenness_centrality = nx.betweenness_centrality(self.G)

        # Calculate closeness centrality
        closeness_centrality = nx.closeness_centrality(self.G)

        # Calculate eigenvector centrality
        eigenvector_centrality = nx.eigenvector_centrality(self.G)

        # Calculate clustering coefficient
        clustering_coefficient = nx.clustering(self.G)

        return {
            'degree_centrality': degree_centrality,
            'betweenness_centrality': betweenness_centrality,
            'closeness_centrality': closeness_centrality,
            'eigenvector_centrality': eigenvector_centrality,
            'clustering_coefficient': clustering_coefficient
        }

    # def gather_statistics(self):
    #     analysis_results = self.perform_analysis()
    #
    #     for metric, values in analysis_results.items():
    #         print(f"{metric}:")
    #         for node, value in values.items():
    #             print(f"Node {node}: {value}")
    #         print("\n")
    #
    #     return analysis_results

    def add_edge(self, a1, a2):
        if a1 in self.G and 'type' in self.G.nodes[a1] and a2 in self.G and 'type' in self.G.nodes[
            a2]:
            self.G.add_edge(a1, a2)
            agent1_type = self.G.nodes[a1]['type']
            agent2_type = self.G.nodes[a2]['type']
            if agent1_type == 'human' and agent2_type == 'human':
                self.H.add_edge(a1, a2)
            elif agent1_type == 'zombie' and agent2_type == 'zombie':
                self.Z.add_edge(a1, a2)
        else:
            print(f"Nodes {a1} and/or {a2} do not exist in the graph or do not have a 'type' attribute.")

    def remove_edge(self, node1, node2):
        self.G.remove_edge(node1, node2)
        print(f"Removed edge between {node1} and {node2}.")


    def visualize_network(self, network_type):
        # Select the appropriate graph
        if network_type == 'human':
            G = self.H
        elif network_type == 'zombie':
            G = self.Z
        else:
            G = self.G

        # Generate random 3D positions for each node
        pos = {node: (np.random.rand(), np.random.rand(), np.random.rand()) for node in G.nodes()}

        # Set up 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Extract node positions
        xs, ys, zs = [], [], []
        for node, (x, y, z) in pos.items():
            xs.append(x)
            ys.append(y)
            zs.append(z)

        # Plot nodes
        ax.scatter(xs, ys, zs)

        # Plot edges
        for edge in G.edges():
            x, y, z = zip(*[pos[node] for node in edge])
            ax.plot(x, y, z, color='black')  # Customize color as needed

        # Customize the axes and display
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.show()


