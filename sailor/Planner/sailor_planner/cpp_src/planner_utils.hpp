#include <vector>
#include <numeric>
#include <climits>
#include <bits/stdc++.h>
#include <cmath>
#include <algorithm>

#include "training.hpp"
#include "utils/network_utils.hpp"

#define SPEEDUP_THRESHOLD 1.5
#define MEMORY_BUCKET_DEEPSPEED_SIZE 500000000

#ifndef NETWORK_COEFFS_TYPE
#define NETWORK_COEFFS_TYPE unordered_map<string, unordered_map<int, unordered_map<int, unordered_map<string, unordered_map<int, unordered_map<int, pair<vector<double>, double>>>>>>>
#endif

using namespace std;

struct StageConfig
{
    int dp;
    unordered_map<string, int> num_gpus_used;
    vector<tuple<int, int, int>> dp_pairs; // dp pairs, each in the form of (GPU_j, TP_j, zone_j)

    StageConfig() : dp(0), num_gpus_used({}), dp_pairs({}) {};

    StageConfig(
        int dp_arg,
        unordered_map<string, int> num_gpus_used_arg,
        vector<tuple<int, int, int>> dp_pairs_arg) : dp(dp_arg), num_gpus_used(num_gpus_used_arg), dp_pairs(dp_pairs_arg) {};

    vector<tuple<int, int, int>> get_dp_pairs() { return dp_pairs; }
};

struct PipelineInfo
{
    double straggler;
    double comp_time;
    double inter_stage_comm;
    double ar_sync;
    double update;
    double Tpp;
    double first_stage_comp;
    double ar_cost;

    PipelineInfo() : straggler(0.0), comp_time(0.0), inter_stage_comm(0.0), ar_sync(0.0), update(0.0), Tpp(0.0), first_stage_comp(0.0), ar_cost(0.0) {};

    PipelineInfo(
        double straggler_arg,
        double all_comp_time_arg,
        double inter_stage_comm_arg,
        double ar_sync_arg,
        double update_arg,
        double Tpp_arg,
        double first_stage_comp_arg,
        double ar_cost_arg) : straggler(straggler_arg), comp_time(all_comp_time_arg), inter_stage_comm(inter_stage_comm_arg), ar_sync(ar_sync_arg), update(update_arg), Tpp(Tpp_arg), first_stage_comp(first_stage_comp_arg), ar_cost(ar_cost_arg) {};
};

vector<vector<int>> get_stages(int num_layers, int pp);
double get_cost_gpu_type(int num_gpus_used, string gpu_type);

vector<vector<int>> find_tmp_degrees(
    vector<vector<int>> &stages,
    struct TrainingInfo *training_info,
    vector<vector<map<pair<int, int>, int>>> &max_tmps_vector_per_gpu,
    vector<vector<map<tuple<int, int, int>, int>>> min_tmps_vector_per_gpu,
    vector<vector<int>> &possible_tmps,
    int mbs,
    int num_available_gpus,
    int float_size,
    bool homog
);

bool check_stage_fits(
    vector<vector<int>> &stages,
    map<pair<int, int>, size_t> &params_all_stages,
    map<pair<int, int>, double> &activation_per_stage,
    int stage_idx,
    int mbs,
    int tmp,
    string gpu_type,
    int float_size);

struct PipelineInfo simulate_time_single_stage(
    vector<int> &stage,
    vector<pair<int, int>> &tp_degrees,
    set<string> &zones,
    vector<string> &id_to_zone,
    struct TrainingInfo *training_info,
    int mbs,
    NETWORK_COEFFS_TYPE &network_coeff,
    map<pair<string, string>, vector<vector<vector<map<pair<int, int>, double>>>>> &ar_times_bottleneck,
    double stage_params_size,
    int dp);

double get_ar_time_with_buckets(
    size_t tensor_size,
    double bucket_size,
    int num_workers,
    vector<double> &network_coeff,
    unordered_map<size_t, double> known_times = {});

double find_activation_time(
    int stage_idx,
    int num_stages,
    int config_idx,
    int num_configs,
    double activation_size,
    vector<struct StageConfig> &configs_per_stage,
    NETWORK_COEFFS_TYPE &network_coeff,
    unordered_map<size_t, double> &known_times);

pair<double, double> merge_stages_get_time(
    struct PipelineInfo *pp_info1,
    struct PipelineInfo *pp_info2,
    struct TrainingInfo *training_info,
    int min_dp,
    int mbs,
    int float_size,
    int stage_idx,
    int num_stages,
    vector<vector<int>> &stages,
    vector<struct StageConfig> &configs_per_stage,
    NETWORK_COEFFS_TYPE &network_coeff,
    unordered_map<size_t, double> &known_times,
    unordered_map<string, unordered_map<string, double>> &comm_cost,
    vector<string>& id_to_zone
);

string extract_region_from_zone(const string &zone);
vector<pair<string, vector<string>>> get_regions_list(
    const unordered_map<string, vector<pair<string, vector<int>>>> &given_resources,
    unordered_map<string, vector<int>> &region_gpu_count,
    unordered_map<string, vector<string>> &zones_per_region,
    vector<string> regions,
    unordered_map<string, unordered_map<string, double>> &throughput,
    const NETWORK_COEFFS_TYPE &network_coeff,
    const vector<vector<int>> &tmp_degrees);

double get_max_throughput(const NETWORK_COEFFS_TYPE &network_coeff,
                          const string &zone1, int gpu_count1,
                          const string &zone2, int gpu_count2);