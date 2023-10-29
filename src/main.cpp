#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

using namespace std;

// Hyperparameters
const int EPOCHS = 10;
const int INPUT_SIZE = 784; // 28x28
const int HIDDEN_SIZE = 256;
const int OUTPUT_SIZE = 10;
const double LEARNING_RATE = 0.001;
const double BETA1 = 0.9;
const double BETA2 = 0.999;
const double EPSILON = 1e-8;

double relu(double x) {
    return max(0.0, x);
}

double reluDerivative(double x) {
    return (x > 0) ? 1.0 : 0.0;
}

void softmax(vector<double> &x) {
    double max_val = *max_element(x.begin(), x.end());
    double sum = 0.0;
    for (auto &val : x) {
        val = exp(val - max_val);
        sum += val;
    }
    for (auto &val : x) {
        val /= sum;
    }
}

void forwardPass(const vector<double> &inputs, const vector<double> &hidden_weights,
                 const vector<double> &hidden_bias, const vector<double> &output_weights,
                 const vector<double> &output_bias, vector<double> &hidden, vector<double> &output) {
    // Hidden layer
    for (int h = 0; h < HIDDEN_SIZE; ++h) {
        for (int j = 0; j < INPUT_SIZE; ++j) {
            hidden[h] += inputs[j] * hidden_weights[h * inputs.size() + j];
        }
        hidden[h] += hidden_bias[h];
        hidden[h] = relu(hidden[h]);
    }

    // Output layer
    for (int o = 0; o < OUTPUT_SIZE; ++o) {
        for (int h = 0; h < HIDDEN_SIZE; ++h) {
            output[o] += hidden[h] * output_weights[o * hidden.size() + h];
        }
        output[o] += output_bias[o];
    }
    softmax(output);
}

void updateWeightsWithAdam(vector<double> &weights, vector<double> &bias,
                           const vector<double> &gradients, const vector<double> &inputs,
                           vector<double> &m_weights, vector<double> &v_weights, int epoch) {
    int layerSize = bias.size();
    int inputSize = inputs.size();

    for (int h = 0; h < layerSize; ++h) {
        for (int j = 0; j < inputSize; ++j) {
            int idx = h * inputSize + j;
            m_weights[idx] = BETA1 * m_weights[idx] + (1.0 - BETA1) * inputs[j] * gradients[h];
            v_weights[idx] = BETA2 * v_weights[idx] + (1.0 - BETA2) * pow(inputs[j] * gradients[h], 2);
            double m_corr = m_weights[idx] / (1.0 - pow(BETA1, epoch));
            double v_corr = v_weights[idx] / (1.0 - pow(BETA2, epoch));
            weights[idx] += LEARNING_RATE * m_corr / (sqrt(v_corr) + EPSILON);
        }
        bias[h] += LEARNING_RATE * gradients[h];
    }
}

vector<vector<double>> readCsv(const string &filename) {
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

vector<int> readSingleColumnCsv(const string &filename) {
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

void writeCsv(const std::string& filename, const std::vector<int>& data) {
    std::ofstream file(filename);
    for (int val : data) {
        file << val << std::endl;
    }
    file.close();
}

int main() {
    vector<vector<double>> train_vectors = readCsv("data/fashion_mnist_train_vectors.csv");
    vector<int> train_labels = readSingleColumnCsv("data/fashion_mnist_train_labels.csv");
    vector<vector<double>> test_vectors = readCsv("data/fashion_mnist_test_vectors.csv");
    vector<int> test_labels = readSingleColumnCsv("data/fashion_mnist_test_labels.csv");

    // Validation split (last 10% of training data)
    int total_training = train_vectors.size();
    int validation_size = total_training / 10;
    int training_size = total_training - validation_size;

    // Initialize weights and biases
    random_device rd;
    mt19937 gen(42);
    uniform_real_distribution<> dis(-0.1, 0.1);

    vector<double> hidden_weights(INPUT_SIZE * HIDDEN_SIZE);
    vector<double> output_weights(HIDDEN_SIZE * OUTPUT_SIZE);
    vector<double> hidden_bias(HIDDEN_SIZE);
    vector<double> output_bias(OUTPUT_SIZE);

    for (auto &w : hidden_weights) w = dis(gen);
    for (auto &w : output_weights) w = dis(gen);
    for (auto &b : hidden_bias) b = dis(gen);
    for (auto &b : output_bias) b = dis(gen);

    // Pre-allocate memory for hidden and output vectors
    vector<double> hidden(HIDDEN_SIZE);
    vector<double> output(OUTPUT_SIZE);

    // Initialize Adam parameters
    vector<double> m_hidden_weights(INPUT_SIZE * HIDDEN_SIZE, 0.0);
    vector<double> v_hidden_weights(INPUT_SIZE * HIDDEN_SIZE, 0.0);
    vector<double> m_output_weights(HIDDEN_SIZE * OUTPUT_SIZE, 0.0);
    vector<double> v_output_weights(HIDDEN_SIZE * OUTPUT_SIZE, 0.0);

    vector<int> train_predictions;

    // Training
    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
        for (size_t i = 0; i < training_size; ++i) {
            // Reset values to zero for reuse
            fill(hidden.begin(), hidden.end(), 0.0);
            fill(output.begin(), output.end(), 0.0);

            // Forward pass
            vector<double> hidden(HIDDEN_SIZE, 0.0);
            vector<double> output(OUTPUT_SIZE, 0.0);
            forwardPass(train_vectors[i], hidden_weights, hidden_bias, output_weights, output_bias, hidden, output);

            // Write to train_predictions.csv
            int predicted_label = max_element(output.begin(), output.end()) - output.begin();
            if (epoch == EPOCHS - 1) train_predictions.push_back(predicted_label);

            // Compute the loss and error
            vector<double> error(OUTPUT_SIZE, 0.0);
            for (int o = 0; o < OUTPUT_SIZE; ++o) {
                error[o] = (o == train_labels[i]) ? 1.0 - output[o] : 0.0 - output[o];
            }

            // Backpropagation
            vector<double> d_output(OUTPUT_SIZE, 0.0);
            for (int o = 0; o < OUTPUT_SIZE; ++o) {
                d_output[o] = error[o]; // Since the derivative of softmax loss w.r.t its input is error[o] itself
            }

            vector<double> d_hidden(HIDDEN_SIZE, 0.0);
            for (int h = 0; h < HIDDEN_SIZE; ++h) {
                double error = 0.0;
                for (int o = 0; o < OUTPUT_SIZE; ++o) {
                    error += d_output[o] * output_weights[o * HIDDEN_SIZE + h];
                }
                d_hidden[h] = error * reluDerivative(hidden[h]); // replace sigmoidDerivative with reluDerivative
            }

            // Update weights and biases with Adam optimization
            updateWeightsWithAdam(hidden_weights, hidden_bias, d_hidden, train_vectors[i],
                                  m_hidden_weights, v_hidden_weights, epoch);

            updateWeightsWithAdam(output_weights, output_bias, d_output, hidden,
                                  m_output_weights, v_output_weights, epoch);
        }

        // Validation
        int correct_count = 0;
        for (size_t i = training_size; i < total_training; ++i) {
            // Reset values to zero for reuse
            fill(hidden.begin(), hidden.end(), 0.0);
            fill(output.begin(), output.end(), 0.0);

            // Forward pass
            vector<double> hidden(HIDDEN_SIZE, 0.0);
            vector<double> output(OUTPUT_SIZE, 0.0);
            forwardPass(train_vectors[i], hidden_weights, hidden_bias, output_weights, output_bias, hidden, output);

            // Write to train_predictions.csv
            int predicted_label = max_element(output.begin(), output.end()) - output.begin();
            if (epoch == EPOCHS - 1) train_predictions.push_back(predicted_label);
            correct_count += (predicted_label == train_labels[i]) ? 1 : 0;
        }
        double accuracy = (double)correct_count / validation_size * 100;
        cout << "Epoch " << epoch << ", Accuracy: " << accuracy << "%" << endl;
    }

    // Testing
    int correct_count = 0;
    vector<int> test_predictions;
    for (size_t i = 0; i < test_vectors.size(); ++i) {
        // Forward pass
        vector<double> hidden(HIDDEN_SIZE, 0.0);
        vector<double> output(OUTPUT_SIZE, 0.0);
        forwardPass(test_vectors[i], hidden_weights, hidden_bias, output_weights, output_bias, hidden, output);

        // Write to test_predictions.csv
        int predicted_label = max_element(output.begin(), output.end()) - output.begin();
        test_predictions.push_back(predicted_label);
        correct_count += (predicted_label == test_labels[i]) ? 1 : 0;
    }
    double accuracy = (double)correct_count / test_vectors.size() * 100;
    cout << "Final Test Accuracy: " << accuracy << "%" << endl;

    writeCsv("train_predictions.csv", train_predictions);
    writeCsv("test_predictions.csv", test_predictions);

    return 0;
}
