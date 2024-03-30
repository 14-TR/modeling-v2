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
