#include "network_utils.hpp"

#include <bits/stdc++.h>
#include <cassert>

#define INV_BILLION 1.0 / 1e9

double get_time_on_config(
    size_t tensor_size,
    vector<double>& network_coefficients,
    unordered_map<size_t, double>& known_times
) {
    /**
    * Use network coeffs to compute time to send the tensor size
    * @param tensor_size: size of tensor in bytes
    * @param network_coeffs: polynomial network coeffs
    * @param known_times: caching map
    * @return estimated communication time (in seconds)
    */
    // if (known_times.find(tensor_size) != known_times.end()) {
    //     return known_times[tensor_size];
    // }
    double tensor_size_mb = log2(tensor_size) - log2(1000000.0);
    int N = network_coefficients.size();

    double bw_gb = network_coefficients[0];
    for (int i = 1; i < N; i++){
        bw_gb = bw_gb*tensor_size_mb + network_coefficients[i];
    }
    assert(bw_gb);
    double time = (tensor_size * INV_BILLION) / bw_gb;

    //known_times[tensor_size] = time - TODO
    return time;
}

double get_network_bandwidth(size_t tensor_size, vector<double>& network_coefficients) {

    double a = network_coefficients[0];
    double b = network_coefficients[1];
    double c = network_coefficients[2];

    double tensor_size_mb = log2(tensor_size) - log2(1000000.0);
    double bw_gb = network_coefficients[0];
    int N = network_coefficients.size();
    for (int i = 1; i < N; i++){
        bw_gb = bw_gb*tensor_size_mb + network_coefficients[i];
    }
    return bw_gb;
}


double estimate_send_time(
    double tensor_size,
    vector<double>& network_coefficients,
    unordered_map<size_t, double>& known_times
) {
    return get_time_on_config(
        tensor_size,
        network_coefficients,
        known_times
    );
}

double estimate_send_time_heterogeneous(
    double tensor_size,
    vector<double>& network_coefficients_sender,
    vector<double>& network_coefficients_receiver,
    unordered_map<size_t, double>& known_times
) {

    // sender and receiver might have different coefficients
    double time_sender = get_time_on_config(
        tensor_size,
        network_coefficients_sender,
        known_times
    );

    double time_receiver = get_time_on_config(
        tensor_size,
        network_coefficients_receiver,
        known_times
    );

    return max(time_sender, time_receiver);

}

double estimate_ar_time(
    double tensor_size,
    vector<double>& network_coefficients,
    int num_workers,
    unordered_map<size_t, double>& known_times
) {
    double part_time = get_time_on_config(
        tensor_size/num_workers, network_coefficients, known_times);
    return 2 * (num_workers-1) * part_time;
}


double estimate_ar_time_with_bw(double tensor_size, double bw_bytes, int num_workers)
{
    double part_time = (tensor_size/num_workers)/bw_bytes;
    return 2 * (num_workers-1) * part_time;
}