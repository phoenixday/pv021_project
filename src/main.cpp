#include <iostream> // allowed
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm> // allowed
#include <random> // allowed

using namespace std;

// Hyperparameters
const int EPOCHS = 10;
const int INPUT_SIZE = 784; // 28x28
const int HIDDEN_SIZE1 = 256;
const int HIDDEN_SIZE2 = 128;
const int HIDDEN_SIZE3 = 128;
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

void passHidden(vector<double> &prev_layer, vector<double> &layer,
                const vector<double> &weights, const vector<double> &bias) {
    for (int i = 0; i < layer.size(); ++i) {
        for (int j = 0; j < prev_layer.size(); ++j) {
            layer[i] += prev_layer[j] * weights[i * prev_layer.size() + j];
        }
        layer[i] += bias[i];
        layer[i] = relu(layer[i]);
    }
}

void passOutput(vector<double> &prev_layer, vector<double> &output,
                const vector<double> &weights, const vector<double> &bias) {
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        for (int j = 0; j < prev_layer.size(); ++j) {
            output[i] += prev_layer[j] * weights[i * prev_layer.size() + j];
        }
        output[i] += bias[i];
    }
    softmax(output);
}

void backpropagationHidden(vector<double> &layer, vector<double> &d_layer, 
                           vector<double> &d_next_layer, vector<double> next_layer_weights) {
    for (int i = 0; i < d_layer.size(); ++i) {
        double error = 0.0;
        for (int j = 0; j < d_next_layer.size(); ++j) {
            error += d_next_layer[j] * next_layer_weights[j * d_layer.size() + i];
        }
        d_layer[i] = error * reluDerivative(layer[i]); 
    }
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

    // He initialization for each layer
    normal_distribution<> dis_hidden1(0, sqrt(2.0 / INPUT_SIZE));
    normal_distribution<> dis_hidden2(0, sqrt(2.0 / HIDDEN_SIZE1));
    normal_distribution<> dis_hidden3(0, sqrt(2.0 / HIDDEN_SIZE2));
    normal_distribution<> dis_output(0, sqrt(2.0 / HIDDEN_SIZE3));

    // weights
    vector<double> hidden_weights1(INPUT_SIZE * HIDDEN_SIZE1);
    vector<double> hidden_weights2(HIDDEN_SIZE1 * HIDDEN_SIZE2);
    vector<double> hidden_weights3(HIDDEN_SIZE2 * HIDDEN_SIZE3);
    vector<double> output_weights(HIDDEN_SIZE3 * OUTPUT_SIZE);
    for (auto &w : hidden_weights1) w = dis_hidden1(gen);
    for (auto &w : hidden_weights2) w = dis_hidden2(gen);
    for (auto &w : hidden_weights3) w = dis_hidden3(gen);
    for (auto &w : output_weights) w = dis_output(gen);

    // biases
    vector<double> hidden_bias1(HIDDEN_SIZE1);
    vector<double> hidden_bias2(HIDDEN_SIZE2);
    vector<double> hidden_bias3(HIDDEN_SIZE3);
    vector<double> output_bias(OUTPUT_SIZE);
    for (auto &b : hidden_bias1) b = dis_hidden1(gen);
    for (auto &b : hidden_bias2) b = dis_hidden2(gen);
    for (auto &b : hidden_bias3) b = dis_hidden3(gen);
    for (auto &b : output_bias) b = dis_output(gen);

    // Pre-allocate memory for hidden and output vectors
    vector<double> hidden1(HIDDEN_SIZE1);
    vector<double> hidden2(HIDDEN_SIZE2);
    vector<double> hidden3(HIDDEN_SIZE2);
    vector<double> output(OUTPUT_SIZE);

    // Initialize Adam weights
    vector<double> m_hidden_weights1(INPUT_SIZE * HIDDEN_SIZE1, 0.0);
    vector<double> v_hidden_weights1(INPUT_SIZE * HIDDEN_SIZE1, 0.0);
    vector<double> m_hidden_weights2(HIDDEN_SIZE1 * HIDDEN_SIZE2, 0.0);
    vector<double> v_hidden_weights2(HIDDEN_SIZE1 * HIDDEN_SIZE2, 0.0);
    vector<double> m_hidden_weights3(HIDDEN_SIZE2 * HIDDEN_SIZE3, 0.0);
    vector<double> v_hidden_weights3(HIDDEN_SIZE2 * HIDDEN_SIZE3, 0.0);
    vector<double> m_output_weights(HIDDEN_SIZE2 * OUTPUT_SIZE, 0.0);
    vector<double> v_output_weights(HIDDEN_SIZE2 * OUTPUT_SIZE, 0.0);

    vector<int> train_predictions;

    // Prepare indices for shuffling
    vector<int> indices(train_vectors.size());
    iota(indices.begin(), indices.end(), 0);

    // Training
    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
        // Shuffle the training data and labels together
        shuffle(indices.begin(), indices.end(), gen);
        
        for (size_t idx = 0; idx < training_size; ++idx) {
            int i = indices[idx];

            // Reset hidden and output layers
            fill(hidden1.begin(), hidden1.end(), 0.0);
            fill(hidden2.begin(), hidden2.end(), 0.0);
            fill(hidden3.begin(), hidden3.end(), 0.0);
            fill(output.begin(), output.end(), 0.0);

            // Forward pass
            passHidden(train_vectors[i], hidden1, hidden_weights1, hidden_bias1);
            passHidden(hidden1, hidden2, hidden_weights2, hidden_bias2);
            passHidden(hidden2, hidden3, hidden_weights3, hidden_bias3);
            passOutput(hidden3, output, output_weights, output_bias);

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

            vector<double> d_hidden3(HIDDEN_SIZE3, 0.0);
            backpropagationHidden(hidden3, d_hidden3, d_output, output_weights);

            vector<double> d_hidden2(HIDDEN_SIZE2, 0.0);
            backpropagationHidden(hidden2, d_hidden2, d_hidden3, hidden_weights3);

            vector<double> d_hidden1(HIDDEN_SIZE1, 0.0);
            backpropagationHidden(hidden1, d_hidden1, d_hidden2, hidden_weights2);

            updateWeightsWithAdam(hidden_weights1, hidden_bias1, d_hidden1, train_vectors[i],
                                m_hidden_weights1, v_hidden_weights1, epoch);

            updateWeightsWithAdam(hidden_weights2, hidden_bias2, d_hidden2, hidden1,
                                m_hidden_weights2, v_hidden_weights2, epoch);

            updateWeightsWithAdam(hidden_weights3, hidden_bias3, d_hidden3, hidden2,
                                m_hidden_weights3, v_hidden_weights3, epoch);

            updateWeightsWithAdam(output_weights, output_bias, d_output, hidden3,
                                  m_output_weights, v_output_weights, epoch);
        }

        // Validation
        int correct_count = 0;
        for (size_t i = training_size; i < total_training; ++i) {
            // Reset hidden and output layers
            fill(hidden1.begin(), hidden1.end(), 0.0);
            fill(hidden2.begin(), hidden2.end(), 0.0);
            fill(hidden3.begin(), hidden3.end(), 0.0);
            fill(output.begin(), output.end(), 0.0);

            // Forward pass
            passHidden(train_vectors[i], hidden1, hidden_weights1, hidden_bias1);
            passHidden(hidden1, hidden2, hidden_weights2, hidden_bias2);
            passHidden(hidden2, hidden3, hidden_weights3, hidden_bias3);
            passOutput(hidden3, output, output_weights, output_bias);
            
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
        fill(hidden3.begin(), hidden3.end(), 0.0);
        fill(output.begin(), output.end(), 0.0);

        // Forward pass with test vectors
        passHidden(test_vectors[i], hidden1, hidden_weights1, hidden_bias1);
        passHidden(hidden1, hidden2, hidden_weights2, hidden_bias2);
        passHidden(hidden2, hidden3, hidden_weights3, hidden_bias3);
        passOutput(hidden3, output, output_weights, output_bias);

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
