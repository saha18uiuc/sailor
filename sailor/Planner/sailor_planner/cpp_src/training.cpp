#include "training.hpp"

struct TrainingInfo* convertLLMInput(
    vector<map<pair<int, int>, map<int, vector<double>>>>& profile_input,
    Json::Value model_mem_info,
    Json::Value train_config_info,
    int float_size
) {
    auto optimizer = train_config_info["optimizer"].asString();
    auto global_batch_size = train_config_info["global_batch_size"].asInt();
    int num_transformer_layers = train_config_info["num_layers"].asInt();

    // the key is gpu_type
    vector<map<pair<int, int>, struct ProfilingInfo>> prof_info_all_gpus;
    vector<set<int>> mbs_set;
    vector<map<int, vector<int>>> mbs_tmp_map;

    int num_layers = 0;
    int idx = 0;

    for (auto gpu_info : profile_input) {

        mbs_set.push_back({});
        mbs_tmp_map.push_back({});

        map<pair<int, int>, struct ProfilingInfo> prof_info;
        for (auto it: gpu_info) {
            // it is a pair of (mbs, tmp)

            vector<double> fwd_times = {};
            vector<double> bwd_times = {};
            vector<double> exec_times = {};
            double update_time = 0.0;

            for (auto layer_info : it.second) {
                auto info = layer_info.second;
                fwd_times.push_back(info[0]);
                bwd_times.push_back(info[1]);
                update_time = max(update_time, info[2]);
            }

            num_layers = fwd_times.size();
            for (int i=0; i < num_layers; i++) {
                exec_times.push_back(fwd_times[i]+bwd_times[i]);
            }

            prof_info[it.first] = ProfilingInfo(
                fwd_times,
                bwd_times,
                exec_times,
                update_time
            );

            auto mbs = it.first.first;
            auto tmp = it.first.second;

            mbs_set[idx].insert(mbs);
            if (mbs_tmp_map[idx].find(mbs) == mbs_tmp_map[idx].end())
                mbs_tmp_map[idx][mbs] = {};
            mbs_tmp_map[idx][mbs].push_back(tmp);
        }
        prof_info_all_gpus.push_back(prof_info);
        idx++;

    }

    // tp up to 8
    vector<vector<size_t>> activation_params(9, vector<size_t>(num_layers, 0));
    vector<vector<size_t>> out_params(9, vector<size_t>(num_layers, 0));
    vector<vector<size_t>> layer_params(9, vector<size_t>(num_layers, 0));


    for (auto it = model_mem_info.begin(); it != model_mem_info.end(); ++it) {
        string key = it.key().asString();
        int tmp = stoi(key);
        for (auto it_tmp = model_mem_info[key].begin(); it_tmp != model_mem_info[key].end(); ++it_tmp) {
            string key_tmp = it_tmp.key().asString();
            int layer = stoi(key_tmp);
            //printf("%d, %d\n",tmp, layer);

            activation_params[tmp][layer] = model_mem_info[key][key_tmp]["act_mem_floats"].asUInt64();
            out_params[tmp][layer] = model_mem_info[key][key_tmp]["act_output_floats"].asUInt64();
            layer_params[tmp][layer] = model_mem_info[key][key_tmp]["params_floats"].asUInt64();
        }
    }

    struct ModelInfo model_info(
        prof_info_all_gpus,
        mbs_tmp_map,
        activation_params,
        out_params,
        layer_params,
        num_layers
    );

    struct TrainingInfo* train_info = new TrainingInfo(
        model_info,
        optimizer,
        global_batch_size,
        float_size,
        mbs_set
    );


    return train_info;
}
