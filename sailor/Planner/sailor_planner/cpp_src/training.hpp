#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <jsoncpp/json/json.h>
#include <iostream>
#include <set>

using namespace std;

struct ProfilingInfo {
    vector<double> fwd_times;
    vector<double> bwd_times;
    vector<double> exec_times;
    double update;

    ProfilingInfo():
        fwd_times({}), bwd_times({}), exec_times({}), update(0.0) {};

    ProfilingInfo(
        vector<double> fwd_times_arg,
        vector<double> bwd_times_arg,
        vector<double> exec_times_arg,
        double update_arg):
            fwd_times(fwd_times_arg), bwd_times(bwd_times_arg),
            exec_times(exec_times_arg), update(update_arg) {};
};

struct ModelInfo {
    vector<map<pair<int,int>, struct ProfilingInfo>> profiles_per_gpu;
    vector<map<int, vector<int>>> mbs_tmp_map_per_gpu;
    vector<vector<size_t>> activation_params;
    vector<vector<size_t>> out_params;
    vector<vector<size_t>> layer_params;
    int num_layers;

    ModelInfo():
        profiles_per_gpu({}), mbs_tmp_map_per_gpu({}), activation_params({}), out_params({}), layer_params({}), num_layers(0) {};

    ModelInfo(
        vector<map<pair<int,int>, struct ProfilingInfo>> profiles_arg,
        vector<map<int, vector<int>>> mbs_tmp_map_arg,
        vector<vector<size_t>> activation_params_arg,
        vector<vector<size_t>> out_params_arg,
        vector<vector<size_t>> layer_params_arg,
        int num_layers_arg):
            profiles_per_gpu(profiles_arg), mbs_tmp_map_per_gpu(mbs_tmp_map_arg), activation_params(activation_params_arg),
            out_params(out_params_arg), layer_params(layer_params_arg), num_layers(num_layers_arg) {};
};

struct TrainingInfo {
    struct ModelInfo model;
    string optimizer;
    int global_batch_size;
    int bytes_per_parameter;
    vector<set<int>> mbs_set;

    TrainingInfo():
        model(ModelInfo()), optimizer(""), global_batch_size(0), bytes_per_parameter(0), mbs_set({}) {};

    TrainingInfo(
        struct ModelInfo model_arg,
        string optimizer_arg,
        int global_batch_size_arg,
        int bytes_per_parameter_arg,
        vector<set<int>> mbs_set_arg):
            model(model_arg), optimizer(optimizer_arg),
            global_batch_size(global_batch_size_arg),
            bytes_per_parameter(bytes_per_parameter_arg),
            mbs_set(mbs_set_arg) {};
};

struct TrainingInfo* convertLLMInput(
    vector<map<pair<int, int>, map<int, vector<double>>>>& profile_input,
    Json::Value model_mem_info,
    Json::Value train_config_info,
    int float_size
);
