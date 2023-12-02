#include <iostream> // allowed
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm> // allowed
#include <random> // allowed

using namespace std;

// Hyperparameters
const int EPOCHS = 20;
const int INPUT_SIZE = 784; // 28x28
const int HIDDEN_SIZE1 = 64;
const int HIDDEN_SIZE2 = 32;
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

void forwardPass(const vector<double> &inputs, 
                 const vector<double> &hidden_weights1, const vector<double> &hidden_bias1, 
                 const vector<double> &hidden_weights2, const vector<double> &hidden_bias2, 
                 const vector<double> &output_weights, const vector<double> &output_bias, 
                 vector<double> &hidden1, vector<double> &hidden2, vector<double> &output) {
    // Hidden layer 1
    for (int h = 0; h < HIDDEN_SIZE1; ++h) {
        for (int j = 0; j < INPUT_SIZE; ++j) {
            hidden1[h] += inputs[j] * hidden_weights1[h * INPUT_SIZE + j];
        }
        hidden1[h] += hidden_bias1[h];
        hidden1[h] = relu(hidden1[h]);
    }

    // Hidden layer 2
    for (int h = 0; h < HIDDEN_SIZE2; ++h) {
        for (int j = 0; j < HIDDEN_SIZE1; ++j) {
            hidden2[h] += hidden1[j] * hidden_weights2[h * HIDDEN_SIZE1 + j];
        }
        hidden2[h] += hidden_bias2[h];
        hidden2[h] = relu(hidden2[h]);
    }

    // Output layer
    for (int o = 0; o < OUTPUT_SIZE; ++o) {
        for (int h = 0; h < HIDDEN_SIZE2; ++h) {
            output[o] += hidden2[h] * output_weights[o * HIDDEN_SIZE2 + h];
        }
        output[o] += output_bias[o];
    }
    softmax(output);
}

void updateWeightsWithAdam(vector<double> &weights, vector<double> &biases,
                           const vector<double> &gradient_weights, const vector<double> &gradient_biases,
                           vector<double> &m_weights, vector<double> &v_weights,
                           vector<double> &m_biases, vector<double> &v_biases,
                           int epoch, int layer_size, int prev_layer_size) {
    for (int i = 0; i < layer_size; ++i) {
        // Update weights
        for (int j = 0; j < prev_layer_size; ++j) {
            int idx = i * prev_layer_size + j;
            m_weights[idx] = BETA1 * m_weights[idx] + (1 - BETA1) * gradient_weights[idx];
            v_weights[idx] = BETA2 * v_weights[idx] + (1 - BETA2) * gradient_weights[idx] * gradient_weights[idx];

            double m_hat = m_weights[idx] / (1 - pow(BETA1, epoch));
            double v_hat = v_weights[idx] / (1 - pow(BETA2, epoch));

            weights[idx] -= LEARNING_RATE * m_hat / (sqrt(v_hat) + EPSILON);
        }

        // Update biases
        m_biases[i] = BETA1 * m_biases[i] + (1 - BETA1) * gradient_biases[i];
        v_biases[i] = BETA2 * v_biases[i] + (1 - BETA2) * gradient_biases[i] * gradient_biases[i];

        double m_hat_bias = m_biases[i] / (1 - pow(BETA1, epoch));
        double v_hat_bias = v_biases[i] / (1 - pow(BETA2, epoch));

        biases[i] -= LEARNING_RATE * m_hat_bias / (sqrt(v_hat_bias) + EPSILON);
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

    // He initialization for each layer
    normal_distribution<> dis_hidden1(0, sqrt(2.0 / INPUT_SIZE));
    normal_distribution<> dis_hidden2(0, sqrt(2.0 / HIDDEN_SIZE1));
    normal_distribution<> dis_output(0, sqrt(2.0 / HIDDEN_SIZE2));

    // weights
    vector<double> hidden_weights1(INPUT_SIZE * HIDDEN_SIZE1);
    vector<double> hidden_weights2(HIDDEN_SIZE1 * HIDDEN_SIZE2);
    vector<double> output_weights(HIDDEN_SIZE2 * OUTPUT_SIZE);
    for (auto &w : hidden_weights1) w = dis_hidden1(gen);
    for (auto &w : hidden_weights2) w = dis_hidden2(gen);
    for (auto &w : output_weights) w = dis_output(gen);

    // biases
    vector<double> hidden_bias1(HIDDEN_SIZE1);
    vector<double> hidden_bias2(HIDDEN_SIZE2);
    vector<double> output_bias(OUTPUT_SIZE);
    for (auto &b : hidden_bias1) b = dis_hidden1(gen);
    for (auto &b : hidden_bias2) b = dis_hidden2(gen);
    for (auto &b : output_bias) b = dis_output(gen);

    // Pre-allocate memory for hidden1 and output vectors
    vector<double> hidden1(HIDDEN_SIZE1);
    vector<double> hidden2(HIDDEN_SIZE2);
    vector<double> output(OUTPUT_SIZE);

    // Initialize Adam weights
    vector<double> m_hidden_weights1(INPUT_SIZE * HIDDEN_SIZE1, 0.0);
    vector<double> v_hidden_weights1(INPUT_SIZE * HIDDEN_SIZE1, 0.0);
    vector<double> m_hidden_weights2(HIDDEN_SIZE1 * HIDDEN_SIZE2, 0.0);
    vector<double> v_hidden_weights2(HIDDEN_SIZE1 * HIDDEN_SIZE2, 0.0);
    vector<double> m_output_weights(HIDDEN_SIZE2 * OUTPUT_SIZE, 0.0);
    vector<double> v_output_weights(HIDDEN_SIZE2 * OUTPUT_SIZE, 0.0);

    // Initialize Adam biases
    vector<double> m_hidden_biases1(HIDDEN_SIZE1, 0.0);
    vector<double> v_hidden_biases1(HIDDEN_SIZE1, 0.0);
    vector<double> m_hidden_biases2(HIDDEN_SIZE2, 0.0);
    vector<double> v_hidden_biases2(HIDDEN_SIZE2, 0.0);
    vector<double> m_output_biases(OUTPUT_SIZE, 0.0);
    vector<double> v_output_biases(OUTPUT_SIZE, 0.0);

    vector<int> train_predictions;

    // Training
    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
        for (size_t i = 0; i < training_size; ++i) {
            // Reset hidden and output layers
            fill(hidden1.begin(), hidden1.end(), 0.0);
            fill(hidden2.begin(), hidden2.end(), 0.0);
            fill(output.begin(), output.end(), 0.0);

            // Forward pass
            forwardPass(train_vectors[i], hidden_weights1, hidden_bias1, hidden_weights2, hidden_bias2, output_weights, output_bias, hidden1, hidden2, output);

            // Write to train_predictions.csv
            int predicted_label = max_element(output.begin(), output.end()) - output.begin();
            if (epoch == EPOCHS - 1) train_predictions.push_back(predicted_label);

            // Compute the loss and error
            vector<double> error_output(OUTPUT_SIZE, 0.0);
            for (int o = 0; o < OUTPUT_SIZE; ++o) {
                error_output[o] = (o == train_labels[i]) ? 1.0 - output[o] : 0.0 - output[o];
            }

            // Backpropagation
            vector<double> d_output(OUTPUT_SIZE, 0.0);
            for (int o = 0; o < OUTPUT_SIZE; ++o) {
                d_output[o] = error_output[o]; 
            }

            // Backpropagation for second hidden layer
            vector<double> d_hidden2(HIDDEN_SIZE2, 0.0);
            vector<double> gradient_output_weights(OUTPUT_SIZE * HIDDEN_SIZE2, 0.0);
            for (int h = 0; h < HIDDEN_SIZE2; ++h) {
                double error = 0.0;
                for (int o = 0; o < OUTPUT_SIZE; ++o) {
                    error += d_output[o] * output_weights[o * HIDDEN_SIZE2 + h];
                    gradient_output_weights[o * HIDDEN_SIZE2 + h] = d_output[o] * hidden2[h];
                }
                d_hidden2[h] = error * reluDerivative(hidden2[h]);
            }

            // Backpropagation for first hidden layer
            vector<double> d_hidden1(HIDDEN_SIZE1, 0.0);
            vector<double> gradient_hidden2_weights(HIDDEN_SIZE2 * HIDDEN_SIZE1, 0.0);
            for (int h = 0; h < HIDDEN_SIZE1; ++h) {
                double error = 0.0;
                for (int j = 0; j < HIDDEN_SIZE2; ++j) {
                    error += d_hidden2[j] * hidden_weights2[j * HIDDEN_SIZE1 + h];
                    gradient_hidden2_weights[j * HIDDEN_SIZE1 + h] = d_hidden2[j] * hidden1[h];
                }
                d_hidden1[h] = error * reluDerivative(hidden1[h]);
            }

            // Update weights and biases with Adam optimization
            vector<double> gradient_hidden1_weights(INPUT_SIZE * HIDDEN_SIZE1, 0.0);
            for (int h = 0; h < HIDDEN_SIZE1; ++h) {
                for (int j = 0; j < INPUT_SIZE; ++j) {
                    gradient_hidden1_weights[h * INPUT_SIZE + j] = d_hidden1[h] * train_vectors[i][j];
                }
            }

            updateWeightsWithAdam(output_weights, output_bias, gradient_output_weights, d_output,
                                m_output_weights, v_output_weights, m_output_biases, v_output_biases,
                                epoch, OUTPUT_SIZE, HIDDEN_SIZE2);

            updateWeightsWithAdam(hidden_weights2, hidden_bias2, gradient_hidden2_weights, d_hidden2,
                                m_hidden_weights2, v_hidden_weights2, m_hidden_biases2, v_hidden_biases2,
                                epoch, HIDDEN_SIZE2, HIDDEN_SIZE1);

            updateWeightsWithAdam(hidden_weights1, hidden_bias1, gradient_hidden1_weights, d_hidden1,
                                m_hidden_weights1, v_hidden_weights1, m_hidden_biases1, v_hidden_biases1,
                                epoch, HIDDEN_SIZE1, INPUT_SIZE);
        }

        // Validation
        int correct_count = 0;
        for (size_t i = training_size; i < total_training; ++i) {
            // Reset hidden and output layers
            fill(hidden1.begin(), hidden1.end(), 0.0);
            fill(hidden2.begin(), hidden2.end(), 0.0);
            fill(output.begin(), output.end(), 0.0);

            // Forward pass
            forwardPass(train_vectors[i], hidden_weights1, hidden_bias1, hidden_weights2, hidden_bias2, output_weights, output_bias, hidden1, hidden2, output);

            // Write to train_predictions.csv
            int predicted_label = max_element(output.begin(), output.end()) - output.begin();
            if (epoch == EPOCHS) train_predictions.push_back(predicted_label);
            correct_count += (predicted_label == train_labels[i]) ? 1 : 0;
        }
        double accuracy = (double)correct_count / validation_size * 100;
        cout << "Epoch " << epoch << ", Accuracy: " << accuracy << "%" << endl;
    }

    // Testing
    int correct_count = 0;
    vector<int> test_predictions;
    for (size_t i = 0; i < test_vectors.size(); ++i) {
        // Reset hidden and output layers
        fill(hidden1.begin(), hidden1.end(), 0.0);
        fill(hidden2.begin(), hidden2.end(), 0.0);
        fill(output.begin(), output.end(), 0.0);

        // Forward pass with test vectors
        forwardPass(test_vectors[i], hidden_weights1, hidden_bias1, hidden_weights2, hidden_bias2, output_weights, output_bias, hidden1, hidden2, output);

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
