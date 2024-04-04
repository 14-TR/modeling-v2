import datetime
import random
from config import log_path
import csv
import os

from log import MoveRecord, EncRecord, ResRecord, GrpRecord


class IDGenerator:
    def __init__(self):
        self.characters = "123456789ABCDEFG"
        self.used_ids = set()

    def gen_id(self, entity_type):
        while True:
            new_id = ''.join(random.choices(self.characters, k=6))
            if new_id not in self.used_ids:
                self.used_ids.add(new_id)
                return new_id + "_" +entity_type


# Create a global instance of IDGenerator
id_generator = IDGenerator()


def write_logs_to_csv(log, log_type):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # Get the current date and time
    dir_path = os.path.join(log_path, timestamp)  # Append the timestamp to the log_path
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_path = os.path.join(dir_path, log_type + "_log.csv")

    # Open the file in append mode. This will create the file if it doesn't exist.
    with open(file_path, 'a+', newline='') as file:
        writer = csv.writer(file)
        if log_type == "move":
            writer.writerow(
                ["Epoch", "Day", "Entity", "Old Location X", "Old Location Y", "Old Z", "New Location X", "New Location Y", "New Z"])
            for record in log.logs:
                writer.writerow(
                    str(record).split(','))  # Convert the record object to a string and split it into a list
        elif log_type == "enc":
            writer.writerow(["Epoch", "Day", "Entity 1", "Entity 2", "Interaction Type"])
            for record in log.logs:
                writer.writerow(
                    str(record).split(','))  # Convert the record object to a string and split it into a list
        elif log_type == "res":
            writer.writerow(["Epoch", "Day", "Entity", "Resource Change", "Reason"])
            for record in log.logs:
                writer.writerow(
                    str(record).split(','))  # Convert the record object to a string and split it into a list
        elif log_type == "grp":
            writer.writerow(["Epoch", "Day", "Group", "Entity", "Action", "Reason"])
            for record in log.logs:
                writer.writerow(str(record).split(','))  # Convert the record object to a string and split it into a list
        elif log_type == "metrics":
            log.write_to_csv(log_path, log_type + "_log.csv")
