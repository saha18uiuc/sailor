#include <stdio.h>
#include <math.h>
#include <vector>
#include <unordered_map>

using namespace std;

double get_time_on_config(
    size_t tensor_size,
    vector<double>& network_coefficients,
    unordered_map<size_t, double>& known_times
);

double estimate_send_time(
    double tensor_size,
    vector<double>& network_coefficients,
    unordered_map<size_t, double>& known_times
);

double get_network_bandwidth(size_t tensor_size, vector<double>& network_coefficients);

double estimate_send_time_heterogeneous(
    double tensor_size,
    vector<double>& network_coefficients_sender,
    vector<double>& network_coefficients_receiver,
    unordered_map<size_t, double>& known_times
);

double estimate_ar_time(
    double tensor_size,
    vector<double>& network_coefficients,
    int num_workers,
    unordered_map<size_t, double>& known_times
);


double estimate_ar_time_with_bw(double tensor_size, double bw_bytes, int num_workers);