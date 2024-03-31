import random
from config import inf_rate
from ents import Group


def interact(ent1, ent2):
    if not ent1.is_adjacent(ent2):
        return

    if ent1.is_z or ent2.is_z:
        if not ent1.is_z:
            human, zombie = ent1, ent2
        else:
            human, zombie = ent2, ent1

        if human.loc['x'] - 2 <= zombie.loc['x'] <= human.loc['x'] + 2 and human.loc['y'] - 2 <= zombie.loc['y'] <= human.loc['y'] + 2:
            human_to_zombie(human, zombie)
    else:
        if ent1.loc['x'] - 2 <= ent2.loc['x'] <= ent1.loc['x'] + 2 and ent1.loc['y'] - 2 <= ent2.loc['y'] <= ent1.loc['y'] + 2:
            human_to_human(ent1, ent2)


def update_status(entity):
    if entity.is_zombie:
        entity.att['ttd'] -= 1.5
        if entity.att['ttd'] <= 0:
            entity.is_active = False
    else:
        if entity.att['res'] > 0:
            entity.att['res'] -= .5
        else:
            entity.is_zombie = True
            entity.att['res'] = 0


def love_encounter(human, other):
    amount = (human.att['res'] + other.att['res']) / 2
    human.att['res'] = other.att['res'] = amount
    human.enc['luv'] += 1
    human.xp['luv_xp'] += .5
    other.enc['luv'] += 1
    other.xp['luv_xp'] += .5

    # creates a group between the two entities if they are not already group mates
    for group in human.grp:
        if other.id in group.members:
            break
    else:
        group = Group()
        group.members.add_member(human.id)
        group.members.add_member(other.id)
        human.grp[group.id] = group
        other.grp[group.id] = group

    # add each others id to their network as a friend
    human.net['friend'][other.id] = other
    other.net['friend'][human.id] = human

    #  encounter and resource change logging placeholder


def war_encounter(human, other):
    from config import loser_survival_rate, loser_death_rate

    # Determine the winner and loser based on war_xp
    winner, loser = (human, other) if human.xp['war'] > other.xp['war'] else (other, human)

    # Update war_xp and resources
    winner.xp['war'] += .5
    winner.att['res'] += loser.att['res'] * (1 - loser_survival_rate)
    loser.att['res'] *= loser_survival_rate

    # Check if loser is dead and handle accordingly
    if loser.att['res'] <= 0 or random.random() < loser_death_rate:
        loser.att['res'] = 0
        loser.is_zombie = True
        for group in loser.grp.values():
            group.remove_member(loser)
    else:
        # Add each other to their network as foes
        winner.net['foes'][loser.id] = loser
        loser.net['foes'][winner.id] = winner


def theft_encounter(human, other):
    # Determine the winner and loser based on theft_xp
    winner, loser = (human, other) if human.xp['theft'] > other.xp['theft'] else (other, human)

    # Update theft_xp and resources
    winner.xp['rob'] += .5
    winner.att['res'] += loser.att['res']
    loser.att['res'] = 0

    # Check if loser is dead and handle accordingly
    if loser.att['res'] <= 0:
        loser.is_zombie = True
        for group in loser.grp.values():
            group.remove_member(loser)
    else:
        # Add each other to their network as foes
        winner.net['foes'][loser.id] = loser
        loser.net['foes'][winner.id] = winner

    # encounter and resource change logging placeholder


def kill_zombie_encounter(human, zombie):
    human.xp['war'] += .5
    human.att['res'] += 2
    zombie.is_active = False
    # log


def infect_human_encounter(human, zombie):
    human.is_zombie = True
    human.att['ttd'] = 10
    zombie.att['ttd'] += 2
    # log


def human_to_human(human, other):
    outcome = random.choices(population=['love', 'war', 'rob', 'run'],
                             weights=[abs(human.xp['luv'] + other.xp['luv'] + 0.1),
                                      abs(human.xp['war'] + other.xp['war'] + 0.1),
                                      abs(human.xp['rob'] + other.xp['rob'] + 0.1),
                                      abs(human.xp['esc'] + other.xp['esc'] + 0.1)
                                      ])
    if outcome[0] == 'love':
        love_encounter(human, other)
    elif outcome[0] == 'war':
        war_encounter(human, other)
    elif outcome[0] == 'rob':
        theft_encounter(human, other)
    elif outcome[0] == 'esc':
        human.xp['esc'] += .5


def human_to_zombie(human, zombie):
    outcome = random.choices(population=['kill',
                                         'inf',
                                         'esc'],
                             weights=[abs(human.xp['war'] + 0.1),
                                      abs(inf_rate + 0.1),
                                      abs(human.xp['esc'] + 0.1)
                                      ])
    if outcome[0] == 'kill':
        kill_zombie_encounter(human, zombie)
        human.xp['war'] += .5
    elif outcome[0] == 'inf':
        infect_human_encounter(human, zombie)
    elif outcome[0] == 'esc':
        human.xp['esc'] += .5


def zombie_to_human(zombie, other):
    inf_event = random.choices([True, False],
                               [inf_rate, (1 - inf_rate) + other.xp['esc'] + 0.1])
    if inf_event[0]:
        infect_human_encounter(other, zombie)
    else:
        other.xp['esc'] += .5