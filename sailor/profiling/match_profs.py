import sys
import json
import numpy as np
import copy
import os

def print_amp(path_planner):
    data = np.load(path_planner)
    print(data)

def print_oobleck(path_planner):
    with open(path_planner, 'r') as f:
        planner_dict = json.load(f)

    fwd = []
    bwd = []
    for layer, data in planner_dict['data'].items():
        fwd.append(data['forward'])
        bwd.append(data['backward'])

def print_sailor(path_planner):
    with open(path_planner, 'r') as f:
        planner_dict = json.load(f)

    fwd = {}
    bwd = {}

    # per bs
    for bs, data in planner_dict.items():
        data_fwd = [data["FwdE"], data["FwdT"], data["FwdH"]]
        data_bwd = [data["BwdE"], data["BwdT"], data["BwdH"]]

        fwd[bs] = data_fwd
        bwd[bs] = data_bwd

    print(fwd, bwd)


def print_varuna(path_planner):

    with open(path_planner, 'r') as f:
        planner_dict = json.load(f)

    fwd = {}
    bwd = {}

    # per layer
    for layer, data in planner_dict.items():
        fwd_bs = {}
        bwd_bs = {}
        for bs in data.keys():
            fwd_bs[bs] = data[bs]['forward']
            bwd_bs[bs] = data[bs]['backward']

        print(layer, fwd_bs, bwd_bs)


def match_varuna(sim_dict, path_planner, num_layers):
    with open(path_planner, 'r') as f:
        planner_dict = json.load(f)

    # No TMP here
    new_dict = {}
    for layer, data in planner_dict.items():
        to_remove = []
        for bs in data.keys():
            if bs in sim_dict:
                sim_dict_tmp1 = sim_dict[bs]["1"]
                if layer == str(0):
                    data[bs]['forward'] = sim_dict_tmp1[0][0]
                    data[bs]['backward'] = sim_dict_tmp1[0][1]
                elif layer == str(num_layers+1):
                    data[bs]['forward'] = sim_dict_tmp1[2][0]
                    data[bs]['backward'] = sim_dict_tmp1[2][1]
                else:
                    data[bs]['forward'] = sim_dict_tmp1[1][0]
                    data[bs]['backward'] = sim_dict_tmp1[1][1]
            else:
                to_remove.append(bs)
        for bs in to_remove:
            data.pop(bs)
        new_dict[layer] = data

    with open(path_planner, 'w') as f:
        json.dump(new_dict, f, indent=2)


def match_oobleck(sim_dict, path_planner, num_layers):
    with open(path_planner, 'r') as f:
        planner_dict = json.load(f)

    sim_dict_tmp1 = sim_dict["1"]["1"]
    for layer, data in planner_dict['data'].items():
        if layer == str(0):
            data['forward'] = sim_dict_tmp1[0][0]
            data['backward'] = sim_dict_tmp1[0][1]
        elif layer==str(num_layers+1):
            data['forward'] = sim_dict_tmp1[2][0]
            data['backward'] = sim_dict_tmp1[2][1]
        else:
            data['forward'] = sim_dict_tmp1[1][0]
            data['backward'] = sim_dict_tmp1[1][1]
        planner_dict['data'][layer] = data

    print(planner_dict['data'])
    with open(path_planner, 'w') as f:
        json.dump(planner_dict, f, indent=2)

def match_amp(sim_dict, path_planner, num_layers):

    sim_dict_bs1 = sim_dict["1"]
    for tmp in sim_dict_bs1.keys():
        sim_list = [sim_dict_bs1[tmp][0][0]] + num_layers*[sim_dict_bs1[tmp][1][0]] + [sim_dict_bs1[tmp][2][0]]

        path_planner_tmp = os.path.dirname(os.path.realpath(path_planner)) + f"/profile_{tmp}.npy"

        with open(path_planner_tmp, 'wb') as f:
            np.save(f, np.asarray(sim_list))


def match_piper(sim_dict, path_planner, num_layers):
    with open(path_planner, 'r') as f:
        planner_dict = json.load(f)

    new_nodes = []
    for node in planner_dict["nodes"]:
        sim_dict_bs1 = sim_dict["1"]
        for tmp in sim_dict_bs1.keys():
            node["TMPCs"][tmp] = copy.deepcopy(node["TMPCs"]["1"])
            if node["id"] == 0:
                node["TMPCs"][tmp][0]["timePerSample"] = sim_dict_bs1[tmp][0][0] + sim_dict_bs1[tmp][0][1]
            elif node["id"] == num_layers+1:
                node["TMPCs"][tmp][0]["timePerSample"] = sim_dict_bs1[tmp][2][0] + sim_dict_bs1[tmp][2][1]
            else:
                node["TMPCs"][tmp][0]["timePerSample"] = sim_dict_bs1[tmp][1][0] + sim_dict_bs1[tmp][1][1]
        new_nodes.append(node)

    print(new_nodes)
    planner_dict["nodes"] = new_nodes
    with open(path_planner, 'w') as f:
        json.dump(planner_dict, f, indent=2)


def match_sailor(sim_dict, path_planner, num_layers):
    with open(path_planner, 'r') as f:
        planner_dict = json.load(f)

    print(planner_dict)

    for bs, data in planner_dict.items():
        data_copy = copy.deepcopy(data)
        data_new = {}
        for tmp in sim_dict[bs].keys():
            data_copy_tmp = copy.deepcopy(data_copy)
            data_copy_tmp['FwdE'] = sim_dict[bs][tmp][0][0]
            data_copy_tmp['BwdE'] = sim_dict[bs][tmp][0][1]

            data_copy_tmp['FwdT'] = sim_dict[bs][tmp][1][0]
            data_copy_tmp['BwdT'] = sim_dict[bs][tmp][1][1]

            data_copy_tmp['FwdH'] = sim_dict[bs][tmp][2][0]
            data_copy_tmp['BwdH'] = sim_dict[bs][tmp][2][1]

            data_copy_tmp['Update'] = sim_dict[bs][tmp][3]
            data_new[tmp] = data_copy_tmp

        planner_dict[bs] = data_new

    with open(path_planner, 'w') as f:
        json.dump(planner_dict, f, indent=2)

def match_planner(sim_dict, planner, path_planner, num_layers):
    if planner == 'Varuna':
        match_varuna(sim_dict, path_planner, num_layers)
    elif planner == 'Oobleck':
        match_oobleck(sim_dict, path_planner, num_layers)
    elif planner == 'AMP':
        match_amp(sim_dict, path_planner, num_layers)
    elif planner == 'Piper':
        match_piper(sim_dict, path_planner, num_layers)
    elif planner == 'Sailor':
        match_sailor(sim_dict, path_planner, num_layers)

if __name__ == "__main__":
    model = sys.argv[1]
    gpu_type = sys.argv[2]
    num_layers = int(sys.argv[3])
    path_sim = 'Planner/simulations/profiles_tmp.json'
    with open(path_sim, 'r') as f:
        path_dict = json.load(f)

    path_dict = path_dict[model][gpu_type]
    #print(path_dict)
    for planner in ['AMP']:
        if planner=='AMP':
            path_planner = f"Planner/baselines/{planner}/profiles/{model}/{gpu_type}/profile.npy"
        elif planner=='Sailor':
            path_planner = f"Planner/sailor_planner/profiles/{model}/{gpu_type}/profile.json"
        else:
            path_planner = f"Planner/baselines/{planner}/profiles/{model}/{gpu_type}/profile.json"
        #print_varuna(path_planner)
        match_planner(path_dict, planner, path_planner, num_layers)
