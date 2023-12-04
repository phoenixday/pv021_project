#include "csv_io.h"
#include <vector>
#include <iostream>
#include <fstream> // for reading csv
#include <sstream> // for writing to csv

using namespace std;

vector<vector<double>> readVectors(const string &filename) {
    cout << "Reading vectors from " << filename << " ..." << endl;
    ifstream file(filename);
    string line;
    vector<vector<double>> data;
    while (getline(file, line)) {
        stringstream lineStream(line);
        string cell;
        vector<double> row;
        while (getline(lineStream, cell, ',')) {
            row.push_back(stod(cell));
        }
        data.push_back(row);
    }
    if (data.empty()) {
        cerr << filename << " is empty or could not be read." << endl;
    }
    return data;
}

vector<int> readLabels(const string &filename) {
    cout << "Reading labels from " << filename << " ..." << endl;
    ifstream file(filename);
    string line;
    vector<int> data;
    while (getline(file, line)) {
        data.push_back(stoi(line));
    }
    if (data.empty()) {
        cerr << filename << " is empty or could not be read." << endl;
    }
    return data;
}

void writePredictions(const string& filename, const vector<int>& data) {
    cout << "Writing predictions to " << filename << " ..." << endl;
    ofstream file(filename);
    for (int val : data) {
        file << val << endl;
    }
    file.close();
}