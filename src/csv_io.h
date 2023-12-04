#ifndef CSV_IO_H
#define CSV_IO_H

#include <vector>
#include <string>

using namespace std;

vector<vector<double>> readVectors(const string &filename);
vector<int> readLabels(const string &filename);
void writePredictions(const string &filename, const vector<int> &data);

#endif // CSV_IO_H