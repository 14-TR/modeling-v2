import random

log_path = r"C:\Users\TR\Desktop\results"
# log_path = r"C:\Users\tingram\Desktop\results"

inf_rate = 2

def get_random_res():
    return random.randint(10, 20)

start_res = get_random_res()

start_ttd = random.randint(10, 20)

ttd_rate = .5

res_lose_rate = .5

max_res_gain = 5

size = 10
grid_size = (size, size)
w, h = size, size
vi = 0.025
vj = 0.025
z = 4

num_humans = 10

num_zombies = 1

epochs = 10

days = 30

loser_survival_rate = 0.25  # The loser keeps % of their resources

loser_death_rate = 0.5  # chance that the loser is killed