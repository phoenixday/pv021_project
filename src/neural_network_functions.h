#ifndef NEURAL_NETWORK_FUNCTIONS_H
#define NEURAL_NETWORK_FUNCTIONS_H

#include <functional>
#include <vector>

using namespace std;

void passHidden(const vector<double> &prev_layer, vector<double> &layer,
                const vector<double> &weights, const vector<double> &bias, 
                function<double(double)> activation);
void passOutput(const vector<double> &prev_layer, vector<double> &output,
                const vector<double> &weights, const vector<double> &bias,
                function<void(vector<double> &x)> activation);
void backpropagationHidden(const vector<double> &layer, vector<double> &d_layer, 
                           const vector<double> &d_next_layer, const vector<double> next_layer_weights,
                           function<double(double)> activationDerivative);
void updateWeightsWithAdam(vector<double> &weights, vector<double> &bias,
                           const vector<double> &gradients, const vector<double> &inputs,
                           vector<double> &m_weights, vector<double> &v_weights, const int epoch);

#endif // NEURAL_NETWORK_FUNCTIONS_H