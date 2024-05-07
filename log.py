import pandas as pd

from env import DayTracker, Epoch


class EncRecord:
    def __init__(self, ent, other, action, new_loc_x, new_loc_y, new_loc_z):
        self.epoch = Epoch.get_current_epoch()
        self.day = DayTracker.get_current_day()
        self.ent_id = ent.id
        # self.ent_loc = ent.loc
        self.other_id = other.id
        # self.other_loc = other.loc
        self.action = action
        self.new_loc_x = new_loc_x
        self.new_loc_y = new_loc_y
        self.new_loc_z = new_loc_z

    def __dict__(self):
        return {'Epoch': self.epoch, 'Day': self.day, 'Entity 1': self.ent_id, 'Entity 2': self.other_id, 'Interaction Type': self.action, 'New X': self.new_loc_x, 'New Y': self.new_loc_y, 'New Z': self.new_loc_z}

    def __str__(self):
        return f"{self.epoch},{self.day},{self.ent_id},{self.other_id},{self.action},{self.new_loc_x},{self.new_loc_y},{self.new_loc_z}"


class MoveRecord:
    def __init__(self, ent, old_loc_x, old_loc_y, old_loc_z, new_loc_x, new_loc_y, new_loc_z):
        self.epoch = Epoch.get_current_epoch()
        self.day = DayTracker.get_current_day()
        self.entity_id = ent.id
        self.old_loc_x = old_loc_x
        self.old_loc_y = old_loc_y
        self.old_loc_z = old_loc_z
        self.new_loc_x = new_loc_x
        self.new_loc_y = new_loc_y
        self.new_loc_z = new_loc_z

    def __str__(self):
        return f"{self.epoch},{self.day},{self.entity_id},{self.old_loc_x},{self.old_loc_y},{self.old_loc_z},{self.new_loc_x},{self.new_loc_y},{self.new_loc_z}"


class ResRecord:
    def __init__(self, ent, res_change=0, reason=None):
        self.epoch = Epoch.get_current_epoch()
        self.day = DayTracker.get_current_day()
        self.entity_id = ent.id
        # self.entity_loc = ent.loc
        self.res_change = res_change
        self.current_res = ent.att['res']
        self.reason = reason

    def __str__(self):
        return f"{self.epoch},{self.day},{self.entity_id},{self.res_change},{self.current_res},{self.reason}"


class GrpRecord:
    def __init__(self, grp, member, action, reason):
        self.epoch = Epoch.get_current_epoch()
        self.day = DayTracker.get_current_day()
        self.group_id = grp.id
        self.member_id = member
        self.action = action
        self.reason = reason

    def __str__(self):
        return f"{self.epoch},{self.day},{self.group_id},{self.member_id},{self.action},{self.reason}"


class EncLog:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EncLog, cls).__new__(cls)
            cls._instance.logs = []
        return cls._instance

    def log(self, ent, other, action, new_loc_x, new_loc_y, new_loc_z):
        record = EncRecord(ent, other, action, new_loc_x, new_loc_y, new_loc_z)
        self.logs.append(record)

    def display_logs(self):
        for log in self.logs:
            print(str(log))


class MoveLog:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MoveLog, cls).__new__(cls)
            cls._instance.logs = []
        return cls._instance

    def log(self, entity, old_loc_x, old_loc_y, old_loc_z, new_loc_x, new_loc_y, new_loc_z):
        record = MoveRecord(entity, old_loc_x, old_loc_y, old_loc_z, new_loc_x, new_loc_y, new_loc_z)
        self.logs.append(record)

    def display_logs(self):
        for log in self.logs:
            print(str(log))


class ResLog:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ResLog, cls).__new__(cls)
            cls._instance.logs = []
        return cls._instance

    def log(self, entity, res_change, reason):
        record = ResRecord(entity, res_change, reason)
        self.logs.append(record)

    def display_logs(self):
        for log in self.logs:
            print(str(log))


class GrpLog:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GrpLog, cls).__new__(cls)
            cls._instance.logs = []
        return cls._instance

    def log(self, group, member, action, reason):
        record = GrpRecord(group, member, action, reason)
        self.logs.append(record)

    def display_logs(self):
        for log in self.logs:
            print(str(log))


# class MetLog:
#     _instance = None
#
#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super(MetLog, cls).__new__(cls)
#             cls._instance.__init__()
#         return cls._instance
#
#     def __init__(self):
#         self.metrics = pd.DataFrame(columns=['Epoch', 'Total_Days', 'Ending_Num_Humans', 'Ending_Num_Zombies', 'Peak_Zombies', 'Peak_Groups', 'Love', 'War', 'Rob', 'Esc', 'Kill', 'Infect'])
#         self.logs = []
#
#     def log_metrics(self, epoch, total_days, ending_num_humans, ending_num_zombies, peak_zombies, peak_groups, love, war, rob, esc, kill, infect):
#         self.metrics = self.metrics.append({'Epoch': epoch, 'Total_Days': total_days, 'Ending_Num_Humans': ending_num_humans, 'Ending_Num_Zombies': ending_num_zombies, 'Peak_Zombies': peak_zombies, 'Peak_Groups': peak_groups, 'Love': love, 'War': war, 'Rob': rob, 'Esc': esc, 'Kill': kill, 'Infect': infect}, ignore_index=True)
#
#     def write_to_csv(self, log_path, filename):
#         self.metrics.to_csv(log_path + filename, index=False)


ml = MoveLog()
rl = ResLog()
el = EncLog()
gl = GrpLog()
# metl = MetLog()

