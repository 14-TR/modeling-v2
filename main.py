import os
import pandas as pd
from datetime import datetime
from abm import Simulation, ml, rl, gl
from config import log_path
from network_manager import NetworkManager


def main():
    # Initialize the simulation
    sim = Simulation()

    # Generate a unique folder name using the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    new_log_path = os.path.join(log_path, timestamp)

    # Create the new folder
    os.makedirs(new_log_path, exist_ok=True)

    # Initialize an empty DataFrame for the encounter logs
    enc_df = pd.DataFrame(columns=["Epoch", "Day", "Entity 1", "Entity 2", "Interaction Type"])

    # Run the simulation and get the metrics dictionary and encounter logs
    metrics_list, encounter_logs = sim.run()

    # Convert the metrics dictionary to a DataFrame
    metrics_df = pd.DataFrame(metrics_list)

    # Gather network analysis statistics
    # network_statistics = sim.network_manager.gather_statistics()

    # Convert the network statistics dictionary to a DataFrame
    # network_statistics_df = pd.DataFrame(network_statistics)

    # Append the network statistics to the metrics DataFrame
    # metrics_df = pd.concat([metrics_df, network_statistics_df], axis=1)

    # Append the encounter logs for each epoch to the DataFrame
    for logs in encounter_logs:
        enc_df = enc_df._append(pd.DataFrame([str(record).split(',') for record in logs],
                                             columns=["Epoch", "Day", "Entity 1", "Entity 2", "Interaction Type"]))

    # Create dataframes for each log and write them to CSV files in the new folder
    move_df = pd.DataFrame([str(record).split(',') for record in ml.logs],
                           columns=["Epoch", "Day", "Entity", "Old X", "Old Y", "Old Z", "New X", "New Y", "New Z"])
    move_df.to_csv(os.path.join(new_log_path, "move_log.csv"), index=False)

    res_df = pd.DataFrame([str(record).split(',') for record in rl.logs],
                          columns=["Epoch", "Day", "Entity", "Resource Change", "Current Resources", "Reason"])
    res_df.to_csv(os.path.join(new_log_path, "res_log.csv"), index=False)

    grp_df = pd.DataFrame([str(record).split(',') for record in gl.logs],
                          columns=["Epoch", "Day", "Group", "Entity", "Action", "Reason"])
    grp_df.to_csv(os.path.join(new_log_path, "grp_log.csv"), index=False)

    # Write the metrics DataFrame to a CSV file
    metrics_df.to_csv(os.path.join(new_log_path, "metrics_log.csv"), index=False)

    # Write the encounter logs DataFrame to a CSV file
    enc_df.to_csv(os.path.join(new_log_path, "enc_log.csv"), index=False)

    # visualize the network
    sim.network_manager.visualize_network('human')
    sim.network_manager.visualize_network('zombie')

    # sim.network_manager.visualize_network('all')

    # general_graph_nodes = list(sim.network_manager.G.nodes())
    # human_graph_nodes = list(sim.network_manager.H.nodes())
    # zombie_graph_nodes = list(sim.network_manager.Z.nodes())
    #
    # print("Nodes in the general graph: ", general_graph_nodes)
    # print("Nodes in the human graph: ", human_graph_nodes)
    # print("Nodes in the zombie graph: ", zombie_graph_nodes)

    # general_graph_edges = len(list(sim.network_manager.G.edges()))
    human_graph_edges = len(list(sim.network_manager.H.edges()))
    zombie_graph_edges = len(list(sim.network_manager.Z.edges()))

    # print("Edges in the general graph: ", general_graph_edges)
    print("Edges in the human graph: ", human_graph_edges)
    print("Edges in the zombie graph: ", zombie_graph_edges)

    return metrics_df, move_df, enc_df, res_df, grp_df


if __name__ == '__main__':
    metrics_df, move_df, enc_df, res_df, grp_df = main()
