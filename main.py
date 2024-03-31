from sim import Simulation
from util import write_logs_to_csv
from log import ml, el, rl, gl

# Initialize the simulation
sim = Simulation()

# Run the simulation
sim.run()

# Write the logs to CSV files
write_logs_to_csv(ml, "move")
write_logs_to_csv(el, "enc")
write_logs_to_csv(rl, "res")
write_logs_to_csv(gl, "grp")