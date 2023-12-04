#include "normalization.h"
#include <vector>

using namespace std;

pair<vector<double>, vector<double>> findMinMax(const vector<vector<double>> &data) {
    vector<double> minValues = data[0];
    vector<double> maxValues = data[0];
    for (long unsigned int i = 1; i < data.size(); ++i) {
        for (long unsigned int j = 0; j < data[0].size(); ++j) {
            if (data[i][j] < minValues[j]) {
                minValues[j] = data[i][j];  
            }
            if (data[i][j] > maxValues[j]) {
                maxValues[j] = data[i][j];  
            }
        }
    }
    return {minValues, maxValues};
}

void normalize(vector<vector<double>> &data, 
               const vector<double> &minValues, 
               const vector<double> &maxValues) {
    for (auto& vector : data) {
        for (size_t i = 0; i < vector.size(); ++i) {
            vector[i] = (vector[i] - minValues[i]) / (maxValues[i] - minValues[i]);
        }
    }
}
