#include <vector>
#include <iostream>

using namespace std;

template <typename T> void print_1Dvector_contents(vector<T> v);
template <typename T> void print_2Dvector_contents(vector<vector<T>> v);

template <typename T>
void print_1Dvector_contents(vector<T> v) {
    cout << "[";
    for (auto el : v) {
        cout << el << ",";
    }
    cout << "]";
    cout << endl;
}

template <typename T>
void print_2Dvector_contents(vector<vector<T>> v) {
    for (auto el : v) {
        print_1Dvector_contents<T>(el);
    }
}
