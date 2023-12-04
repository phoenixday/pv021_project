#include "neural_network_functions.h"
#include "hyperparameters.h"
#include <functional>
#include <vector>
#include <cmath>

using namespace std;

void passHidden(const vector<vector<double>>& batch_prev, 
                vector<vector<double>>& batch,
                const vector<double> &weights, const vector<double> &bias,
                function<double(double)> activation) {
    for (size_t batch_idx = 0; batch_idx < batch_prev.size(); ++batch_idx) {
        for (size_t i = 0; i < batch[0].size(); ++i) {
            batch[batch_idx][i] = 0.0;
            for (size_t j = 0; j < batch_prev[0].size(); ++j) {
                batch[batch_idx][i] += batch_prev[batch_idx][j] * weights[i * batch_prev[0].size() + j];
            }
            batch[batch_idx][i] += bias[i];
            batch[batch_idx][i] = activation(batch[batch_idx][i]);
        }
    }
}

void passOutput(const vector<vector<double>>& batch_prev, 
                vector<vector<double>>& batch,
                const vector<double> &weights, const vector<double> &bias,
                function<void(vector<double> &x)> activation) {
    for (size_t batch_idx = 0; batch_idx < batch_prev.size(); ++batch_idx) {
        for (size_t i = 0; i < batch[0].size(); ++i) {
            batch[batch_idx][i] = 0.0;
            for (size_t j = 0; j < batch_prev[0].size(); ++j) {
                batch[batch_idx][i] += batch_prev[batch_idx][j] * weights[i * batch_prev[0].size() + j];
            }
            batch[batch_idx][i] += bias[i];
        }
        activation(batch[batch_idx]);
    }
}

void backpropagationHidden(const vector<vector<double>>& batch, 
                           vector<vector<double>>& batch_d,
                           const vector<vector<double>>& batch_d_next, 
                           const vector<double> &next_layer_weights,
                           function<double(double)> activationDerivative) {
    for (size_t batch_idx = 0; batch_idx < batch.size(); ++batch_idx) {
        for (size_t i = 0; i < batch_d[0].size(); ++i) {
            double error = 0.0;
            for (size_t j = 0; j < batch_d_next[0].size(); ++j) {
                error += batch_d_next[batch_idx][j] * next_layer_weights[j * batch_d[0].size() + i];
            }
            batch_d[batch_idx][i] = error * activationDerivative(batch[batch_idx][i]);
        }
    }
}

void updateWeightsWithAdam(vector<double> &weights, vector<double> &bias,
                           const vector<vector<double>>& batch_gradients, 
                           const vector<vector<double>>& batch_inputs,
                           vector<double> &m_weights, vector<double> &v_weights, const int epoch) {
    vector<double> gradients_accumulate(weights.size(), 0.0);
    vector<double> bias_gradients_accumulate(bias.size(), 0.0);

    // Accumulate gradients over batch
    for (size_t batch_idx = 0; batch_idx < batch_gradients.size(); ++batch_idx) {
        for (size_t i = 0; i < bias.size(); ++i) {
            bias_gradients_accumulate[i] += batch_gradients[batch_idx][i];
            for (size_t j = 0; j < batch_inputs[0].size(); ++j) {
                int idx = i * batch_inputs[0].size() + j;
                gradients_accumulate[idx] += batch_inputs[batch_idx][j] * batch_gradients[batch_idx][i];
            }
        }
    }

    // Update weights and biases using accumulated gradients
    for (size_t i = 0; i < bias.size(); ++i) {
        for (size_t j = 0; j < batch_inputs[0].size(); ++j) {
            int idx = i * batch_inputs[0].size() + j;
            m_weights[idx] = BETA1 * m_weights[idx] + (1.0 - BETA1) * gradients_accumulate[idx];
            v_weights[idx] = BETA2 * v_weights[idx] + (1.0 - BETA2) * pow(gradients_accumulate[idx], 2);
            double m_corr = m_weights[idx] / (1.0 - pow(BETA1, epoch));
            double v_corr = v_weights[idx] / (1.0 - pow(BETA2, epoch));
            weights[idx] += LEARNING_RATE * (m_corr / (sqrt(v_corr) + EPSILON) + LAMBDA * weights[idx]);
        }
        bias[i] += LEARNING_RATE * bias_gradients_accumulate[i] / batch_gradients.size();
    }
}