import random

# log_path = r"C:\Users\TR\Desktop\results"
log_path = r"C:\Users\tingram\Desktop\results"

inf_rate = 2

start_res = float(random.randint(10, 20))

start_ttd = float(random.randint(10, 20))

ttd_rate = .5

res_lose_rate = 1
hunger = res_lose_rate*2

max_res_gain = 5

size = 100
grid_size = (size, size)
w, h = size, size
vi = 0.025
vj = 0.025
z = .5


num_humans = 100

num_zombies = 5

epochs = 1

days = 365

loser_survival_rate = 0.25  # The loser keeps % of their resources

loser_death_rate = 0.5  # chance that the loser is killed
