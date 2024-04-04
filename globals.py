global_entities = {'humans': [], 'zombies': [], 'groups': [], 'removed': []}


def print_removed_entities():
    removed_entities_str = [str(entity) for entity in global_entities['removed']]
    return removed_entities_str
