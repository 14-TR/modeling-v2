import random

from config import inf_rate, res_lose_rate, ttd_rate
from ents import Group, entities
from log import EncRecord, ResRecord, GrpRecord, el, rl, gl



def interact(simulation,ent1, ent2):
    if not ent1.is_adjacent(ent2):
        return

    # Check if both entities are zombies
    if ent1.is_z and ent2.is_z:
        return  # Skip interaction if both entities are zombies

    if ent1.is_z or ent2.is_z:
        if not ent1.is_z:
            human, zombie = ent1, ent2
        else:
            human, zombie = ent2, ent1

        if human.loc['x'] - 2 <= zombie.loc['x'] <= human.loc['x'] + 2 and human.loc['y'] - 2 <= zombie.loc['y'] <= human.loc['y'] + 2:
            human_to_zombie(human, zombie)
    else:
        if ent1.loc['x'] - 2 <= ent2.loc['x'] <= ent1.loc['x'] + 2 and ent1.loc['y'] - 2 <= ent2.loc['y'] <= ent1.loc['y'] + 2:
            human_to_human(simulation, ent1, ent2)


def update_status(entity):
    if entity.is_z:
        entity.att['ttd'] -= ttd_rate
        if entity.att['ttd'] <= 0:
            entity.is_active = False
    else:
        if entity.att['res'] > 0 and not entity.is_z:
            entity.att['res'] -= res_lose_rate
        else:
            entity.is_z = True
            entity.att['res'] = 0


def love_encounter(human, other):
    amount = (human.att['res'] + other.att['res']) / 2
    human.att['res'] = other.att['res'] = amount
    human.enc['luv'] += 1
    human.xp['luv'] += .5
    other.enc['luv'] += 1
    other.xp['luv'] += .5

    # creates a group between the two entities if they are not already group mates
    for group_id in human.grp.keys():
        if group_id in entities:  # Check if the Group object exists
            group = entities[group_id]  # Get the Group object
            if other.id in group.members:
                break
    else:
        group = Group("human")
        group.members.append(human.id)
        group.members.append(other.id)
        human.grp[group.id] = group
        other.grp[group.id] = group

    # add each others id to their network as a friend
    human.net['friend'][other.id] = other
    other.net['friend'][human.id] = human

    #  encounter and resource change logging placeholder
    er = EncRecord(human, other, 'love')
    el.logs.append(er)

    rr = ResRecord(human, amount, 'love')
    rl.logs.append(rr)


def war_encounter(simulation, human, other):
    from config import loser_survival_rate, loser_death_rate

    # Determine the winner and loser based on war_xp
    winner, loser = (human, other) if human.xp['war'] > other.xp['war'] else (other, human)

    # Update war_xp and resources
    winner.xp['war'] += .5
    winner.enc['war'] += 1

    er = EncRecord(human, other, 'war')
    el.logs.append(er)

    winner.att['res'] += loser.att['res'] * (1 - loser_survival_rate)

    rr = ResRecord(human, loser.att['res'] * (1 - loser_survival_rate), 'war')
    rl.logs.append(rr)

    loser.att['res'] *= loser_survival_rate
    loser.enc['war'] += 1

    rr = ResRecord(other, -loser.att['res']*loser_survival_rate, 'war')
    rl.logs.append(rr)

    # Check if loser is dead and handle accordingly
    if loser.att['res'] <= 0 or random.random() < loser_death_rate:
        loser.att['res'] = 0
        loser.turn_into_zombie()
        simulation.zombies.append(loser)
        simulation.humans.remove(loser)
        for group in loser.grp.values():
            group.remove_member(human)
            gr = GrpRecord(group, loser.id, 'remove', 'war')
            gl.logs.append(gr)
    else:
        # Add each other to their network as foes
        winner.net['foe'][loser.id] = loser
        rr = ResRecord(human, loser.att[ 'res' ], 'war')
        rl.logs.append(rr)

        loser.net['foe'][winner.id] = winner
        rr = ResRecord(other, loser.att[ 'res' ], 'war')
        rl.logs.append(rr)


def theft_encounter(human, other):
    # Determine the winner and loser based on theft_xp
    winner, loser = (human, other) if human.xp['rob'] > other.xp['rob'] else (other, human)

    # Update theft_xp and resources
    winner.xp['rob'] += .5
    er = EncRecord(winner, loser, 'rob')
    el.logs.append(er)

    winner.att['res'] += loser.att['res'] * (loser.xp['rob']/winner.xp['rob'])
    rr = ResRecord(winner, loser.att['res'] * (loser.xp['rob']/winner.xp['rob']), 'rob')
    rl.logs.append(rr)

    loser.att['res'] -= loser.att['res'] * (loser.xp['rob']/winner.xp['rob'])
    rr = ResRecord(loser, -loser.att['res'] * (loser.xp['rob']/winner.xp['rob']), 'rob')
    rl.logs.append(rr)

    # Check if loser is dead and handle accordingly
    if loser.att['res'] <= 0:
        loser.is_zombie = True
        for group in loser.grp.values():
            group.remove_member(loser)
            gr = GrpRecord(group, loser.id, 'remove', 'rob')
            gl.logs.append(gr)
    else:
        # Add each other to their network as foes
        winner.net['foe'][loser.id] = loser
        loser.net['foe'][winner.id] = winner

    # encounter and resource change logging placeholder


def kill_zombie_encounter(human, zombie):

    human.xp['war'] += .5

    er = EncRecord(human, zombie, 'kill')
    el.logs.append(er)

    human.att['res'] += 2

    rr = ResRecord(human, 2, 'kill')
    rl.logs.append(rr)

    zombie.is_active = False

    for group in zombie.grp.values():
        group.remove_member(zombie)
        gr = GrpRecord(group, zombie.id, 'remove', 'kill')
        gl.logs.append(gr)

    # log


def infect_human_encounter(human, zombie):

    # Create a new zombie group and add the infected human to it
    zombie_group = Group("zombie")
    zombie_group.add_member(human)
    gr = GrpRecord(zombie_group, human.id, 'add', 'infect')
    gl.logs.append(gr)

    human.att['ttd'] = 10
    er = EncRecord(zombie, human, 'infect')
    el.logs.append(er)
    zombie.att['ttd'] += 2
    zombie.enc['inf'] += 1

    # Remove the human from all his current groups
    for group in human.grp.values():
        group.remove_member(human)
        gr = GrpRecord(group, human.id, 'remove', 'infect')
        gl.logs.append(gr)

    human.turn_into_zombie()


def human_to_human(simulation, human, other):
    outcome = random.choices(population=['love', 'war', 'rob', 'run'],
                             weights=[abs(human.xp['luv'] + other.xp['luv'] + 0.1),
                                      abs(human.xp['war'] + other.xp['war'] + 0.1),
                                      abs(human.xp['rob'] + other.xp['rob'] + 0.1),
                                      abs(human.xp['esc'] + other.xp['esc'] + 0.1)
                                      ])
    if outcome[0] == 'love':
        love_encounter(human, other)
    elif outcome[0] == 'war':
        war_encounter(simulation, human, other)
    elif outcome[0] == 'rob':
        theft_encounter(human, other)
    elif outcome[0] == 'esc':
        human.xp['esc'] += .5
        other.xp['esc'] += .25
        er = EncRecord(human, other, 'esc')
        el.logs.append(er)


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
        er = EncRecord(human, zombie, 'esc')
        el.logs.append(er)


def zombie_to_human(zombie, other):
    inf_event = random.choices([True, False],
                               [inf_rate, (1 - inf_rate) + other.xp['esc'] + 0.1])
    if inf_event[0]:
        infect_human_encounter(other, zombie)
    else:
        other.xp['esc'] += .5
        er = EncRecord(other, zombie, 'esc')
        el.logs.append(er)