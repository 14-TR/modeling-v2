# Description: This file contains the classes for logging the encounters between the player and the other entities in
# the game. The EncRecord class is used to store the details of each encounter, such as the entities involved,
# the action taken, and the resources exchanged. The EncLog class is a singleton class that logs all the encounters
# that occur during the game. The log() method is used to add a new encounter record to the log,
# and the display_logs() method is used to print all the encounter records to the console.

from env import Grid

# ==================================================================


class EncRecord:
    def __init__(self, ent, other, action):
        self.ent = ent
        self.other = other
        self.action = action
        self.ent_loc = ent.loc
        self.other_loc = other.loc
        self.dist_to_res_pnt = ent.grid.get_distance_to_nearest_res_pnt(ent.loc['x'], ent.loc['y'])

    def __str__(self):
        return f"{self.human} {self.action} {self.other}, distance to nearest resource point: {self.dist_to_res_pnt}"


# ==================================================================

#create a class for logging EncRecords, singleton pattern


class EncLog:
	_instance = None

	def __new__(cls):
		if cls._instance is None:
			cls._instance = super(EncLog, cls).__new__(cls)
			cls._instance.logs = []
		return cls._instance

	def log(self, enc_record):
		self.logs.append(enc_record)

	def display_logs(self):
		for log in self.logs:
			print(log)

# ==================================================================
# move record class to log the movement of entities, singleton pattern, from x y, to x y , being id


class MoveRecord:
	def __init__(self, entity, old_loc, new_loc):
		self.entity = entity
		self.old_loc = old_loc
		self.new_loc = new_loc

	def __str__(self):
		return f"{self.entity} moved from {self.old_loc} to {self.new_loc}"

# ==================================================================
# MoveLog class to log the movement of entities singleton pattern


class MoveLog:
	_instance = None

	def __new__(cls):
		if cls._instance is None:
			cls._instance = super(MoveLog, cls).__new__(cls)
			cls._instance.logs = []
		return cls._instance

	def log(self, entity, new_loc):
		self.logs.append((entity, new_loc))

	def display_logs(self):
		for log in self.logs:
			print(f"{log[0]} moved to {log[1]}")

# ==================================================================

# ResRecord class to log resource change of ents, singleton pattern, being id, res change amount, reason for change


class ResRecord:
	def __init__(self, entity, res_change, reason):
		self.entity = entity
		self.res_change = res_change
		self.reason = reason

	def __str__(self):
		return f"{self.entity} changed resources by {self.res_change} due to {self.reason}"

# ==================================================================

# ResLog class to log resource change of ents, singleton pattern


class ResLog:
	_instance = None

	def __new__(cls):
		if cls._instance is None:
			cls._instance = super(ResLog, cls).__new__(cls)
			cls._instance.logs = []
		return cls._instance

	def log(self, res_record):
		self.logs.append(res_record)

	def display_logs(self):
		for log in self.logs:
			print(log)

# ==================================================================


class GrpRecord:
    def __init__(self, group, member, action, reason):
        self.group = group
        self.member = member
        self.action = action
        self.reason = reason

    def __str__(self):
        return f"Member {self.member.id} was {self.action} to group {self.group.id}"

# ==================================================================


class GrpLog:
	_instance = None

	def __new__(cls):
		if cls._instance is None:
			cls._instance = super(GrpLog, cls).__new__(cls)
			cls._instance.logs = []
		return cls._instance

	def log(self, group_record):
		self.logs.append(group_record)

	def display_logs(self):
		for log in self.logs:
			print(log)

# ==================================================================


ml = MoveLog()
rl = ResLog()
el = EncLog()
gl = GrpLog()
