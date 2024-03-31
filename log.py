from env import DayTracker, Epoch


class EncRecord:
    def __init__(self, ent, other, action):
        self.epoch = Epoch.get_current_epoch()
        self.day = DayTracker.get_current_day()
        self.ent = ent.id
        self.other = other.id
        self.action = action
        self.x = ent.loc['x']
        self.y = ent.loc['y']
        self.z = ent.grid.get_elev_at(ent.loc['x'], ent.loc['y'])
        self.res_dist = ent.grid.get_distance_to_nearest_res_pnt(ent.loc['x'], ent.loc['y'])

    def __str__(self):
        return f"Epoch: {self.epoch}, Day: {self.day}, Entity: {self.ent}, Other: {self.other}, Action: {self.action}, X: {self.x}, Y: {self.y}, Z: {self.z}, Resource Distance: {self.res_dist}"


class MoveRecord:
    def __init__(self, ent, old_loc, new_loc):
        self.epoch = Epoch.get_current_epoch()
        self.day = DayTracker.get_current_day()
        self.entity = ent.id
        self.start_x = old_loc['x']
        self.start_y = old_loc['y']
        self.end_x = new_loc['x']
        self.end_y = new_loc['y']

    def __str__(self):
        return f"Epoch: {self.epoch}, Day: {self.day}, Entity: {self.entity}, Start Location: ({self.start_x}, {self.start_y}), End Location: ({self.end_x}, {self.end_y})"


class ResRecord:
    def __init__(self, ent, res_change=0, reason=None):
        self.epoch = Epoch.get_current_epoch()
        self.day = DayTracker.get_current_day()
        self.entity = ent.id
        self.res_change = res_change
        self.current_res = ent.att['res']
        self.reason = reason

    def __str__(self):
        return f"Epoch: {self.epoch}, Day: {self.day}, Entity: {self.entity}, Resource Change: {self.res_change}, Current Resources: {self.current_res}, Reason: {self.reason}"


class GrpRecord:
    def __init__(self, grp, member, action, reason):
        self.epoch = Epoch.get_current_epoch()
        self.day = DayTracker.get_current_day()
        self.group = grp.id
        self.member = member
        self.action = action
        self.reason = reason

    def __str__(self):
        return f"Epoch: {self.epoch}, Day: {self.day}, Group: {self.group}, Member: {self.member}, Action: {self.action}, Reason: {self.reason}"


class EncLog:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EncLog, cls).__new__(cls)
            cls._instance.logs = []
        return cls._instance

    def log(self, ent, other, action):
        record = EncRecord(ent, other, action)
        self.logs.append(str(record))

    def display_logs(self):
        for log in self.logs:
            print(log)


class MoveLog:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MoveLog, cls).__new__(cls)
            cls._instance.logs = []
        return cls._instance

    def log(self, entity, old_loc, new_loc):
        record = MoveRecord(entity, old_loc, new_loc)
        self.logs.append(str(record))

    def display_logs(self):
        for log in self.logs:
            print(log)


class ResLog:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ResLog, cls).__new__(cls)
            cls._instance.logs = []
        return cls._instance

    def log(self, entity, res_change, reason):
        record = ResRecord(entity, res_change, reason)
        self.logs.append(str(record))

    def display_logs(self):
        for log in self.logs:
            print(log)


class GrpLog:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GrpLog, cls).__new__(cls)
            cls._instance.logs = []
        return cls._instance

    def log(self, group, member, action, reason):
        record = GrpRecord(group, member, action, reason)
        self.logs.append(str(record))

    def display_logs(self):
        for log in self.logs:
            print(log)


ml = MoveLog()
rl = ResLog()
el = EncLog()
gl = GrpLog()