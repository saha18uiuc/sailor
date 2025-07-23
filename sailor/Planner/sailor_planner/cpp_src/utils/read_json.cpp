#include "read_json.hpp"

Json::Value read_basic_json(const char* input_path) {
    std::ifstream input_file(input_path, std::ifstream::binary);
    if (!input_file)
    {
        throw std::runtime_error("File was not opened successfully!");
    }

    Json::Value input;
    input_file >> input;

    return input;
}

map<string, unordered_map<string, int>> read_quotas(const char* quotas_dict_path) {
    Json::Value quotas = read_basic_json(quotas_dict_path);
    map<string, unordered_map<string, int>> quotas_map = {};

    int idx=0;
    for (auto it = quotas.begin(); it != quotas.end(); ++it) {
        string gpu_type = it.key().asString();
        quotas_map[gpu_type] = {};
        for (auto zt = quotas[gpu_type].begin(); zt != quotas[gpu_type].end(); ++zt) {
            string zone = zt.key().asString();
            int temp = (*zt).asInt();
            quotas_map[gpu_type][zone] = temp;
        }
        idx++;
    }
    return quotas_map;
}

unordered_map<string, unordered_map<string, double>> read_comm_cost(const char* comm_file) {
    Json::Value comm_input = read_basic_json(comm_file);
    unordered_map<string, unordered_map<string, double>> res = {};

    for (auto it = comm_input.begin(); it != comm_input.end(); ++it) {
        string sender_zone = it.key().asString();
        res[sender_zone] = {};
        for (auto zt = comm_input[sender_zone].begin(); zt != comm_input[sender_zone].end(); ++zt) {
            string recv_zone = zt.key().asString();
            double temp = (*zt).asDouble();
            res[sender_zone][recv_zone] = temp;
        }
    }

    return res;
}

NETWORK_COEFFS_TYPE read_full_network_coeffs(const char* network_coeff_path, vector<string> &available_gpu_types) {

    Json::Value coeffs = read_basic_json(network_coeff_path);
                // { "sender_zone" : { "sender_gpu_type" : {"sender_gpu_count" : { "recv_zone" : { "recv_gpu_type" : {"recv_gpu_count" : coeffs}}}
    NETWORK_COEFFS_TYPE network_coeffs;

    for (auto sender_zones = coeffs.begin(); sender_zones != coeffs.end(); ++sender_zones){
        string send_zone = sender_zones.key().asString();
        network_coeffs[send_zone] = {};

        for (auto sender_gpu_types = coeffs[send_zone].begin(); sender_gpu_types != coeffs[send_zone].end(); ++sender_gpu_types){
            string send_gpu_type = sender_gpu_types.key().asString();
            auto sender_gpu_idx = find(available_gpu_types.begin(), available_gpu_types.end(), send_gpu_type) - available_gpu_types.begin();
            if (sender_gpu_idx >= available_gpu_types.size())
                continue;

            network_coeffs[send_zone][sender_gpu_idx] = {};
            for (auto sender_gpu_counts = coeffs[send_zone][send_gpu_type].begin(); sender_gpu_counts != coeffs[send_zone][send_gpu_type].end(); ++sender_gpu_counts){
                string scount = sender_gpu_counts.key().asString();
                int send_gpu_count = stoi(scount);
                network_coeffs[send_zone][sender_gpu_idx][send_gpu_count] = {};
                for (auto recver_zones = coeffs[send_zone][send_gpu_type][scount].begin(); recver_zones != coeffs[send_zone][send_gpu_type][scount].end(); ++recver_zones){
                    string recv_zone = recver_zones.key().asString();
                    network_coeffs[send_zone][sender_gpu_idx][send_gpu_count][recv_zone] = {};

                    for (auto recver_gpu_types = coeffs[send_zone][send_gpu_type][scount][recv_zone].begin(); recver_gpu_types != coeffs[send_zone][send_gpu_type][scount][recv_zone].end(); ++recver_gpu_types){
                        string recv_gpu_type = recver_gpu_types.key().asString();
                        auto recv_gpu_idx = find(available_gpu_types.begin(), available_gpu_types.end(), recv_gpu_type) - available_gpu_types.begin();
                        network_coeffs[send_zone][sender_gpu_idx][send_gpu_count][recv_zone][recv_gpu_idx] = {};

                        if (recv_gpu_idx >= available_gpu_types.size())
                            continue;

                        for (auto recver_gpu_counts = coeffs[send_zone][send_gpu_type][scount][recv_zone][recv_gpu_type].begin(); recver_gpu_counts != coeffs[send_zone][send_gpu_type][scount][recv_zone][recv_gpu_type].end(); ++recver_gpu_counts){
                            string rcount = recver_gpu_counts.key().asString();
                            int recv_gpu_count = stoi(rcount);

                            network_coeffs[send_zone][sender_gpu_idx][send_gpu_count][recv_zone][recv_gpu_idx][recv_gpu_count] = {};
                            vector<double> curr_coeffs;
                            double max_tp = coeffs[send_zone][send_gpu_type][scount][recv_zone][recv_gpu_type][rcount][1].asDouble();
                            for (auto ut = coeffs[send_zone][send_gpu_type][scount][recv_zone][recv_gpu_type][rcount][0].begin(); ut != coeffs[send_zone][send_gpu_type][scount][recv_zone][recv_gpu_type][rcount][0].end(); ++ut) {
                                double t = (*ut).asDouble();
                                curr_coeffs.push_back(t);
                            } // for coeffs
                            network_coeffs[send_zone][sender_gpu_idx][send_gpu_count][recv_zone][recv_gpu_idx][recv_gpu_count] = make_pair(curr_coeffs, max_tp);
                        } // for recver_gpu_counts
                    } // for recver_gpu_types
                } // for recver_zone
            } // for sender_gpu_counts
        } // for sender_gpu_types
    } // for sender_zone
    return network_coeffs;
}

vector<map<int, vector<double>>>  read_network_coeffs(
    const char* network_coeff_path,
    map<string, unordered_map<string, int>> quotas_dict,
    const char* zone
) {
    Json::Value coeffs = read_basic_json(network_coeff_path);
    vector<map<int, vector<double>>> network_coeffs;

    int idx = 0;
    for (auto quotas_info = quotas_dict.begin(); quotas_info != quotas_dict.end(); ++quotas_info) {
        auto gpu_type = quotas_info->first;
        network_coeffs.push_back({});
        for (auto it = coeffs[zone][gpu_type].begin(); it != coeffs[zone][gpu_type].end(); ++it) {
            string tmp = it.key().asString();
            int int_tmp = stoi(tmp);
            network_coeffs[idx][int_tmp] = {};
            for (auto ut = coeffs[zone][gpu_type][tmp][zone][gpu_type][tmp][0].begin(); ut != coeffs[zone][gpu_type][tmp][zone][gpu_type][tmp][0].end(); ++ut) {
                double t = (*ut).asDouble();
                network_coeffs[idx][int_tmp].push_back(t);
            }
        }
        idx++;
    }

    return network_coeffs;
}

map<pair<int, int>, map<int, vector<double>>> read_llm_profile(const char* profile_path) {
    cout << profile_path << endl;
    Json::Value profile = read_basic_json(profile_path);

    map<pair<int, int>, map<int, vector<double>>>  profile_info;
    for (auto it = profile.begin(); it != profile.end(); ++it) {
        auto mbs = it.key().asString();
        for (auto ut = profile[mbs].begin(); ut != profile[mbs].end(); ++ut) {
            auto tmp = ut.key().asString();
            auto case_json_dict = profile[mbs][tmp];
            map<int, vector<double>> case_dict = {};

            for (auto st = case_json_dict.begin(); st != case_json_dict.end(); ++st) {
                vector<double> prof_info = {
                    case_json_dict[st.key().asString()][0].asDouble(),
                    case_json_dict[st.key().asString()][1].asDouble(),
                    case_json_dict[st.key().asString()][2].asDouble()
                };
                case_dict[stoi(st.key().asString())] = prof_info;
            }

            profile_info[make_pair(stoi(mbs), stoi(tmp))] = case_dict;
        }
    }

    return profile_info;
}
