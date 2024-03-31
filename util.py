import random


class IDGenerator:
	def __init__(self):
		self.characters = "123456789ABCDEFG"
		self.used_ids = set()

	def gen_id(self):
		while True:
			new_id = ''.join(random.choices(self.characters, k = 6))
			if new_id not in self.used_ids:
				self.used_ids.add(new_id)
				return new_id


# Create a global instance of IDGenerator
id_generator = IDGenerator()

from config import log_path
import csv


def write_logs_to_csv(log, log_type):
	file_path = log_path + log_type + "_log.csv"
	with open(file_path, 'w', newline = '') as file:
		writer = csv.writer(file)
		if log_type == "move":
			writer.writerow([ "Entity", "Old Location", "New Location" ])
			for record in log.logs:
				writer.writerow([ record.entity, record.old_loc, record.new_loc ])
		elif log_type == "enc":
			# Add the appropriate headers and data for the EncLog
			pass
		elif log_type == "res":
			# Add the appropriate headers and data for the ResLog
			pass
		elif log_type == "grp":
			# Add the appropriate headers and data for the GrpLog
			pass
