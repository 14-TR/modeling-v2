import random

log_path = r"C:\Users\TR\Desktop\results"
# log_path = r"C:\Users\tingram\Desktop\results"

inf_rate = 2

start_res = random.randint(5, 10)

start_ttd = random.randint(5, 10)

ttd_rate = .5

res_lose_rate = 2

max_res_gain = 5

size = 100
grid_size = (size, size)
w, h = size, size
vi = 0.025
vj = 0.025
z = 4

num_humans = 100

num_zombies = 5

epochs = 1

days = 50

loser_survival_rate = 0.25  # The loser keeps % of their resources

loser_death_rate = 0.5  # chance that the loser is killed