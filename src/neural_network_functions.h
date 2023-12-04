#ifndef NEURAL_NETWORK_FUNCTIONS_H
#define NEURAL_NETWORK_FUNCTIONS_H

#include <functional>
#include <vector>

using namespace std;

void passHidden(const vector<vector<double>>& batch_prev, 
                vector<vector<double>>& batch,
                const vector<double> &weights, const vector<double> &bias,
                function<double(double)> activation);
void passOutput(const vector<vector<double>>& batch_prev, 
                vector<vector<double>>& batch,
                const vector<double> &weights, const vector<double> &bias,
                function<void(vector<double> &x)> activation);
void backpropagationHidden(const vector<vector<double>>& batch, 
                           vector<vector<double>>& batch_d,
                           const vector<vector<double>>& batch_d_next, 
                           const vector<double> &next_layer_weights,
                           function<double(double)> activationDerivative);
void updateWeightsWithAdam(vector<double> &weights, vector<double> &bias,
                           const vector<vector<double>>& batch_gradients, 
                           const vector<vector<double>>& batch_inputs,
                           vector<double> &m_weights, vector<double> &v_weights, const int epoch);

#endif // NEURAL_NETWORK_FUNCTIONS_H