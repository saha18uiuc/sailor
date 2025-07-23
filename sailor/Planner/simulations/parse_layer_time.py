import json

def get_time(input_dict, model, gpu, mbs, tp):
    fwd = input_dict[model][gpu][str(mbs)][str(tp)]["2"][0]
    bwd = input_dict[model][gpu][str(mbs)][str(tp)]["2"][1]
    return fwd+bwd

if __name__=="__main__":
    input_path = "profiles_tmp.json"
    with open(input_path, 'r') as f:
        input_dict = json.load(f)

    model = 'OPT-350'
    gpu = 'A100-40'
    mbs_values = [1,2,4]
    tp_values = [1,2,4]
    for mbs in mbs_values:
        for tp in tp_values:
            fwd_bwd = get_time(input_dict, model, gpu, mbs, tp)
            print(mbs, tp, fwd_bwd)