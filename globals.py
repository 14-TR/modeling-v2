import pandas as pd
from ents import entities

# Create a dictionary to store the entities by type
global_entities = {'humans': [], 'zombies': [], 'groups': [], 'removed': []}

# Iterate over all entities
for entity in entities.values():
    # Add the entity to the corresponding list in the global_entities dictionary
    if entity.is_h:
        global_entities['humans'].append(str(entity))
    elif entity.is_z:
        global_entities['zombies'].append(str(entity))
    elif not entity.is_active:
        global_entities['removed'].append(str(entity))
    # Add more conditions here if there are more types of entities

# Convert the global_entities dictionary to a DataFrame
# global_entities_df = pd.DataFrame.from_dict(global_entities, orient='index').transpose()
#
# # Set the display options
# pd.set_option('display.max_columns', None)  # Show all columns
# pd.set_option('display.max_rows', None)  # Show all rows
# pd.set_option('display.width', None)  # No max width
# pd.set_option('display.max_colwidth', None)  # Show full width of showing strings

# Print the DataFrame
# print(global_entities_df)
# def print_removed_entities():
#     removed_entities_str = [str(entity) for entity in global_entities['removed']]
#     return removed_entities_str
