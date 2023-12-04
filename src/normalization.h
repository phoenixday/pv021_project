#ifndef NORMALIZATION_H
#define NORMALIZATION_H

#include <vector>

using namespace std;

pair<vector<double>, vector<double>> findMinMax(const vector<vector<double>> &data);
void normalize(vector<vector<double>> &data, const vector<double> &minValues, const vector<double> &maxValues);

#endif // NORMALIZATION_H