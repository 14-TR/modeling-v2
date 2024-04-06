import networkx as nx
import matplotlib.pyplot as plt

class NetworkManager:
    def __init__(self):
        self.G = nx.Graph()  # Initialize an empty graph

    def add_agent(self, agent_id, agent_type):
        self.G.add_node(agent_id, type=agent_type)

    def remove_agent(self, agent_id):
        self.G.remove_node(agent_id)

    def add_interaction(self, agent1_id, agent2_id, interaction_type):
        self.G.add_edge(agent1_id, agent2_id, interaction=interaction_type)

    def perform_analysis(self):
        # Example: Calculate degree centrality
        centrality = nx.degree_centrality(self.G)
        return centrality

    def visualize_network(self):
        nx.draw(self.G, with_labels=True)
        plt.show()
