import json
from os.path import expanduser

# Input file path
input_file = f'{expanduser("~")}/elastic-spot-ml/sailor/providers/gcp/multizone_bandwidths.json'

# Load data
with open(input_file, 'r') as f:
    data = json.load(f)

# Extract the zones (keys at the top level)
zones = list(data.keys())


def extract_region(zone):
    parts = zone.split('-')
    return '-'.join(parts[:2])


# Build the cost matrix
cost_matrix = {}
for sender_zone in zones:
    sender_region = extract_region(sender_zone)
    cost_matrix[sender_zone] = {}
    for receiver_zone in zones:
        receiver_region = extract_region(receiver_zone)
        if sender_zone == receiver_zone:
            cost = 0.0
        elif sender_region == receiver_region:
            cost = 0.01
        else:
            cost = 0.02

        cost_matrix[sender_zone][receiver_zone] = cost

# Write to output JSON
with open('communication_cost.json', 'w') as outfile:
    json.dump(cost_matrix, outfile, indent=2)