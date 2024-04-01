import os
import pandas as pd

from sim import Simulation
from log import ml, el, rl, gl
from config import log_path

def main():
    # Initialize the simulation
    sim = Simulation()

    # Run the simulation
    sim.run()

    # Create dataframes for each log and write them to CSV files
    move_df = pd.DataFrame([str(record).split(',') for record in ml.logs], columns=["Epoch", "Day", "Entity", "Old X", "Old Y", "Old Z", "New X", "New Y", "New Z"])
    move_df.to_csv(os.path.join(log_path, "move_log.csv"), index=False)

    enc_df = pd.DataFrame([str(record).split(',') for record in el.logs], columns=["Epoch", "Day", "Entity 1", "Entity 2", "Interaction Type"])
    enc_df.to_csv(os.path.join(log_path, "enc_log.csv"), index=False)

    res_df = pd.DataFrame([str(record).split(',') for record in rl.logs], columns=["Epoch", "Day", "Entity", "Resource Change","Current Resources", "Reason"])
    res_df.to_csv(os.path.join(log_path, "res_log.csv"), index=False)

    grp_df = pd.DataFrame([str(record).split(',') for record in gl.logs], columns=["Epoch", "Day", "Group", "Entity", "Action", "Reason"])
    grp_df.to_csv(os.path.join(log_path, "grp_log.csv"), index=False)

if __name__ == '__main__':
    main()