#include <map>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <iostream>
#include <jsoncpp/json/json.h>
#include <bits/stdc++.h>


#include "print_utils.hpp"
#ifndef NETWORK_COEFFS_TYPE
#define NETWORK_COEFFS_TYPE unordered_map<string, unordered_map<int, unordered_map<int, unordered_map<string, unordered_map<int, unordered_map<int, pair<vector<double>, double>>>>>>>
#endif

using namespace std;

vector<map<int, vector<double>>> read_network_coeffs(
    const char* network_coeff_path,
    map<string, unordered_map<string, int>> quotas_dict,
    const char* zone
);
map<string, unordered_map<string, int>> read_quotas(const char* quotas_dict_path);
unordered_map<string, unordered_map<string, double>> read_comm_cost(const char* comm_file);

Json::Value read_basic_json(const char* input_path);
map<pair<int, int>, map<int, vector<double>>> read_llm_profile(const char* profile_path);

NETWORK_COEFFS_TYPE read_full_network_coeffs(const char* network_coeff_path, vector<string> &available_gpu_types);