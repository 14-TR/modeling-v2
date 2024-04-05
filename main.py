import os
import pprint

import pandas as pd
from datetime import datetime

from globals import global_entities
from sim import Simulation
from log import ml, el, rl, gl
from config import log_path
# from globals import reset_simulation



def main():
    # Initialize the simulation
    # reset_simulation()
    sim = Simulation()

    # Run the simulation and get the metrics dictionary
    metrics_dict = sim.run()

    # Convert the metrics dictionary to a DataFrame
    metrics_df = pd.DataFrame(metrics_dict, index=[0])

    # Generate a unique folder name using the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    new_log_path = os.path.join(log_path, timestamp)

    # Create the new folder
    os.makedirs(new_log_path, exist_ok=True)

    # Create dataframes for each log and write them to CSV files in the new folder
    move_df = pd.DataFrame([str(record).split(',') for record in ml.logs],
                           columns=["Epoch", "Day", "Entity", "Old X", "Old Y", "Old Z", "New X", "New Y", "New Z"])
    move_df.to_csv(os.path.join(new_log_path, "move_log.csv"), index=False)

    enc_df = pd.DataFrame([str(record).split(',') for record in el.logs],
                          columns=["Epoch", "Day", "Entity 1", "Entity 2", "Interaction Type"])
    enc_df.to_csv(os.path.join(new_log_path, "enc_log.csv"), index=False)

    res_df = pd.DataFrame([str(record).split(',') for record in rl.logs],
                          columns=["Epoch", "Day", "Entity", "Resource Change", "Current Resources", "Reason"])
    res_df.to_csv(os.path.join(new_log_path, "res_log.csv"), index=False)

    grp_df = pd.DataFrame([str(record).split(',') for record in gl.logs],
                          columns=["Epoch", "Day", "Group", "Entity", "Action", "Reason"])
    grp_df.to_csv(os.path.join(new_log_path, "grp_log.csv"), index=False)

    # Write the metrics DataFrame to a CSV file
    metrics_df.to_csv(os.path.join(new_log_path, "metrics_log.csv"), index=False)


    # for entity_type, entities in global_entities.items():
    #     global_entities[entity_type] = [str(entity) for entity in entities if
    #                                     entity.day == max(entity.day for entity in entities)]
    #
    # global_entities_df = pd.DataFrame.from_dict(global_entities, orient='index').transpose()
    # pd.set_option('display.max_columns', None)  # Show all columns
    # pd.set_option('display.max_rows', None)  # Show all rows
    # pd.set_option('display.width', None)  # No max width
    # pd.set_option('display.max_colwidth', None)  # Show full width of showing strings
    # print(global_entities_df)
    # the number of zombies from global_entities dict
    # print(len(global_entities['zombies']))
    return metrics_df, move_df, enc_df, res_df, grp_df


if __name__ == '__main__':
    metrics_df, move_df, enc_df, res_df, grp_df = main()