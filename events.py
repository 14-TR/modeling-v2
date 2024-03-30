import random
from config import inf_rate


def interact(ent1, ent2):
    if ent1.is_zombie:
        ent1.zombie_to_human(ent2)
    else:
        if ent2.is_zombie:
            ent1.human_to_zombie(ent2)
        else:
            ent1.human_to_human(ent2)

def update_status(entity):
    if entity.is_zombie:
        entity.half_life -= 1.5
        if entity.lifespan <= 0:
            entity.is_active = False
    else:
        if entity.ressources > 0:
            entity.resources -= .5
        else:
            entity.is_zombie = True
            entity.resources = 0


def love_encounter(human, other):
    amount = (human.resources + other.resources) / 2
    human.resources = other.resources = amount
    #  encounter and resource change logging


def war_encounter(human, other):
    if human.war_xp > other.war_xp:
        human.war_xp += .5
        human.resources = other.resources
        other.resources = 0
        other.is_zombie = True
        #  log war and res change
    elif human.war_xp < other.war_xp:
        other.war_xp += .5
        other.resources = human.resources
        human.resources = 0
        human.is_zombie = True
        #  log war and res change
    else:
        theft_encounter(human, other)


def theft_encounter(human, other):
    result = random.choices(population=[human,
                                        other],
                            weights=[human.theft_xp + 0.1,
                                     other.theft_xp + 0.1
                                     ])
    if result == human:
        amount = other.resources / 2
        human.theft_xp += .5
        human.resources += amount
        other.resources -= amount
        # log theft and res change
    elif result == other:
        amount = human.resources / 2
        other.war_xp += .5
        other.resources += amount
        human.resources -= amount
        # log theft and res change
    else:
        war_encounter(human, other)


def kill_zombie_encounter(human, zombie):
    human.war_xp += .5
    human.resources += 2
    zombie.is_active = False
    # log


def infect_human_encounter(human, zombie):
    human.is_zombie = True
    human.half_life = 10
    zombie.half_life += 3
    # log


def human_to_human(human, other):
    outcome = random.choices(population=['love',
                                         'war',
                                         'theft',
                                         'run'],
                             weights=[abs(human.love_xp + other.love_xp + 0.1),
                                      abs(human.war_xp + other.love_xp + 0.1),
                                      abs(human.theft_xp + other.theft_xp + 0.1),
                                      abs(human.run_xp + other.run_xp + 0.1)
                                      ])
    if outcome == 'love':
        love_encounter(human, other)
    elif outcome == 'war':
        war_encounter(human, other)
    elif outcome == 'theft':
        theft_encounter(human, other)
    elif outcome == 'run':
        human.run_xp += .5
        pass


def human_to_zombie(human, zombie):
    outcome = random.choices(population=['kill',
                                         'inf',
                                         'esc'],
                             weights=[abs(human['war_xp'] + 0.1),
                                      abs(inf_rate + 0.1),
                                      abs(human['esc_xp'] + 0.1)
                                      ])
    if outcome == 'kill':
        kill_zombie_encounter(human, zombie)
        human['war_xp'] += .5
    elif outcome == 'inf':
        infect_human_encounter(human, zombie)
    elif outcome == 'esc':
        human['esc_xp'] += .5
        pass


def zombie_to_human(human, zombie):
    outcome = random.choices(['kill','esc'], [abs(zombie['war_xp'] + 0.1),
                                        abs(zombie['esc_xp'] + 0.1)
                                        ])