#include <omp.h>
#include "neural_network_functions.h"
#include "hyperparameters.h"
#include <functional>
#include <vector>
#include <cmath>

using namespace std;

void passHidden(const vector<double> &prev_layer, vector<double> &layer,
                const vector<double> &weights, const vector<double> &bias,
                function<double(double)> activation) {
    #pragma omp parallel for
    for (long unsigned int i = 0; i < layer.size(); ++i) {
        for (long unsigned int j = 0; j < prev_layer.size(); ++j) {
            layer[i] += prev_layer[j] * weights[i * prev_layer.size() + j];
        }
        layer[i] += bias[i];
        layer[i] = activation(layer[i]);
    }
}

void passOutput(const vector<double> &prev_layer, vector<double> &output,
                const vector<double> &weights, const vector<double> &bias,
                function<void(vector<double> &x)> activation) {
    #pragma omp parallel for
    for (long unsigned int i = 0; i < output.size(); ++i) {
        for (long unsigned int j = 0; j < prev_layer.size(); ++j) {
            output[i] += prev_layer[j] * weights[i * prev_layer.size() + j];
        }
        output[i] += bias[i];
    }
    activation(output);
}

void backpropagationHidden(const vector<double> &layer, vector<double> &d_layer, 
                           const vector<double> &d_next_layer, const vector<double> next_layer_weights,
                           function<double(double)> activationDerivative) {
    #pragma omp parallel for
    for (long unsigned int i = 0; i < d_layer.size(); ++i) {
        double error = 0.0;
        for (long unsigned int j = 0; j < d_next_layer.size(); ++j) {
            error += d_next_layer[j] * next_layer_weights[j * d_layer.size() + i];
        }
        d_layer[i] = error * activationDerivative(layer[i]); 
    }
}

void updateWeightsWithAdam(vector<double> &weights, vector<double> &bias,
                           const vector<double> &gradients, const vector<double> &inputs,
                           vector<double> &m_weights, vector<double> &v_weights, const int epoch) {
    #pragma omp parallel for
    for (long unsigned int i = 0; i < bias.size(); ++i) {
        for (long unsigned int j = 0; j < inputs.size(); ++j) {
            int idx = i * inputs.size() + j;
            m_weights[idx] = BETA1 * m_weights[idx] + (1.0 - BETA1) * inputs[j] * gradients[i];
            v_weights[idx] = BETA2 * v_weights[idx] + (1.0 - BETA2) * pow(inputs[j] * gradients[i], 2);
            double m_corr = m_weights[idx] / (1.0 - pow(BETA1, epoch));
            double v_corr = v_weights[idx] / (1.0 - pow(BETA2, epoch));
            weights[idx] += LEARNING_RATE * (m_corr / (sqrt(v_corr) + EPSILON) + LAMBDA * weights[idx]);
        }
        bias[i] += LEARNING_RATE * gradients[i];
    }
}