#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <map>
#include "activation_functions.h"
#include "neural_network_functions.h"
#include "normalization.h"
#include "csv_io.h"
#include "hyperparameters.h"

using namespace std;

int main() {
    vector<vector<double>> train_vectors = readVectors("data/fashion_mnist_train_vectors.csv");
    vector<int> train_labels = readLabels("data/fashion_mnist_train_labels.csv");
    vector<vector<double>> test_vectors = readVectors("data/fashion_mnist_test_vectors.csv");
    vector<int> test_labels = readLabels("data/fashion_mnist_test_labels.csv");

    // min-max normalization
    auto [minValues, maxValues] = findMinMax(train_vectors);
    normalize(train_vectors, minValues, maxValues);
    normalize(test_vectors, minValues, maxValues);

    // validation split 
    int total_training = train_vectors.size();
    int validation_size = total_training / 10;
    int training_size = total_training - validation_size;

    // hidden and output vectors
    vector<double> hidden1(HIDDEN_SIZE1);
    vector<double> hidden2(HIDDEN_SIZE2);
    vector<double> output(OUTPUT_SIZE);

    // weights and biases
    random_device rd;
    mt19937 gen(42);
    uniform_real_distribution<> dis(-0.1, 0.1);
    vector<double> hidden_weights1(INPUT_SIZE * HIDDEN_SIZE1);
    vector<double> hidden_weights2(HIDDEN_SIZE1 * HIDDEN_SIZE2);
    vector<double> output_weights(HIDDEN_SIZE2 * OUTPUT_SIZE);
    vector<double> hidden_bias1(HIDDEN_SIZE1);
    vector<double> hidden_bias2(HIDDEN_SIZE2);
    vector<double> output_bias(OUTPUT_SIZE);
    for (auto &w : hidden_weights1) w = dis(gen);
    for (auto &w : hidden_weights2) w = dis(gen);
    for (auto &w : output_weights) w = dis(gen);
    for (auto &b : hidden_bias1) b = dis(gen);
    for (auto &b : hidden_bias2) b = dis(gen);
    for (auto &b : output_bias) b = dis(gen);

    // adam weights
    vector<double> m_hidden_weights1(INPUT_SIZE * HIDDEN_SIZE1, 0.0);
    vector<double> v_hidden_weights1(INPUT_SIZE * HIDDEN_SIZE1, 0.0);
    vector<double> m_hidden_weights2(HIDDEN_SIZE1 * HIDDEN_SIZE2, 0.0);
    vector<double> v_hidden_weights2(HIDDEN_SIZE1 * HIDDEN_SIZE2, 0.0);
    vector<double> m_output_weights(HIDDEN_SIZE2 * OUTPUT_SIZE, 0.0);
    vector<double> v_output_weights(HIDDEN_SIZE2 * OUTPUT_SIZE, 0.0);

    map<int, int> train_predictions_map;

    // indices for shuffling
    vector<int> indices(train_vectors.size());
    iota(indices.begin(), indices.end(), 0);

    // TRAINING
    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
        shuffle(indices.begin(), indices.end(), gen);
        
        for (int idx = 0; idx < training_size; idx += BATCH_SIZE) {
            // Determine the size of the current batch
            int current_batch_size = min(BATCH_SIZE, training_size - idx);

            // Create and initialize batch vectors
            vector<vector<double>> batch_input(current_batch_size, vector<double>(INPUT_SIZE, 0.0));
            vector<vector<double>> batch_hidden1(current_batch_size, vector<double>(HIDDEN_SIZE1, 0.0));
            vector<vector<double>> batch_hidden2(current_batch_size, vector<double>(HIDDEN_SIZE2, 0.0));
            vector<vector<double>> batch_output(current_batch_size, vector<double>(OUTPUT_SIZE, 0.0));
            vector<vector<double>> batch_d_hidden1(current_batch_size, vector<double>(HIDDEN_SIZE1, 0.0));
            vector<vector<double>> batch_d_hidden2(current_batch_size, vector<double>(HIDDEN_SIZE2, 0.0));
            vector<vector<double>> batch_error_output(current_batch_size, vector<double>(OUTPUT_SIZE, 0.0));

            // Prepare the batch
            for (int b = 0; b < batch_input.size(); ++b) {
                batch_input[b] = train_vectors[indices[idx + b]];
            }

            // Forward pass for batch
            passHidden(batch_input, batch_hidden1, hidden_weights1, hidden_bias1, relu);
            passHidden(batch_hidden1, batch_hidden2, hidden_weights2, hidden_bias2, relu);
            passOutput(batch_hidden2, batch_output, output_weights, output_bias, softmax);

            // Error computation and backpropagation for batch
            for (int b = 0; b < batch_input.size(); ++b) {
                int i = indices[idx + b];
                int true_label = train_labels[i];
                for (int o = 0; o < OUTPUT_SIZE; ++o) {
                    batch_error_output[b][o] = (o == true_label) ? 1.0 - batch_output[b][o] : -batch_output[b][o];
                }
            }

            backpropagationHidden(batch_hidden2, batch_d_hidden2, batch_error_output, output_weights, reluDerivative);
            backpropagationHidden(batch_hidden1, batch_d_hidden1, batch_d_hidden2, hidden_weights2, reluDerivative);

            // Update weights using batch gradients
            updateWeightsWithAdam(hidden_weights1, hidden_bias1, batch_d_hidden1, batch_input, 
                                  m_hidden_weights1, v_hidden_weights1, epoch);
            updateWeightsWithAdam(hidden_weights2, hidden_bias2, batch_d_hidden2, batch_hidden1, 
                                  m_hidden_weights2, v_hidden_weights2, epoch);
            updateWeightsWithAdam(output_weights, output_bias, batch_error_output, batch_hidden2, 
                                  m_output_weights, v_output_weights, epoch);

            if (epoch == EPOCHS) {
                for (int b = 0; b < batch_input.size(); ++b) {
                    int i = indices[idx + b];
                    int predicted_label = max_element(batch_output[b].begin(), batch_output[b].end()) - batch_output[b].begin();
                    train_predictions_map[i] = predicted_label;
                }
            }
        }

        // VALIDATION
        int correct_count = 0;
        for (int i = training_size; i < total_training; i += BATCH_SIZE) {
            // Determine the size of the current batch (it might be smaller than BATCH_SIZE at the end)
            int current_batch_size = min(BATCH_SIZE, total_training - i);

            // Create batch vectors
            vector<vector<double>> batch_input(current_batch_size, vector<double>(INPUT_SIZE, 0.0));
            vector<vector<double>> batch_hidden1(current_batch_size, vector<double>(HIDDEN_SIZE1, 0.0));
            vector<vector<double>> batch_hidden2(current_batch_size, vector<double>(HIDDEN_SIZE2, 0.0));
            vector<vector<double>> batch_output(current_batch_size, vector<double>(OUTPUT_SIZE, 0.0));

            // Fill the batch input
            for (int b = 0; b < current_batch_size; ++b) {
                batch_input[b] = train_vectors[i + b];
            }

            // Forward pass for the batch
            passHidden(batch_input, batch_hidden1, hidden_weights1, hidden_bias1, relu);
            passHidden(batch_hidden1, batch_hidden2, hidden_weights2, hidden_bias2, relu);
            passOutput(batch_hidden2, batch_output, output_weights, output_bias, softmax);

            // Process predictions for the batch
            for (int b = 0; b < current_batch_size; ++b) {
                int predicted_label = max_element(batch_output[b].begin(), batch_output[b].end()) - batch_output[b].begin();
                if (epoch == EPOCHS) train_predictions_map[i + b] = predicted_label;
                correct_count += (predicted_label == train_labels[i + b]) ? 1 : 0;
            }
        }
        cout << "Epoch: " << epoch << ", accuracy on validation data: " << (double)correct_count / validation_size << endl;
    }

    // TESTING
    int correct_count = 0;
    vector<int> test_predictions;
    for (int i = 0; i < test_vectors.size(); i += BATCH_SIZE) {
        // Determine the size of the current batch
        int current_batch_size = min(BATCH_SIZE, static_cast<int>(test_vectors.size()) - i);

        // Create and initialize batch vectors
        vector<vector<double>> batch_input(current_batch_size, vector<double>(INPUT_SIZE, 0.0));
        vector<vector<double>> batch_hidden1(current_batch_size, vector<double>(HIDDEN_SIZE1, 0.0));
        vector<vector<double>> batch_hidden2(current_batch_size, vector<double>(HIDDEN_SIZE2, 0.0));
        vector<vector<double>> batch_output(current_batch_size, vector<double>(OUTPUT_SIZE, 0.0));

        // Fill the batch input
        for (int b = 0; b < current_batch_size; ++b) {
            batch_input[b] = test_vectors[i + b];
        }

        // Forward pass for the batch
        passHidden(batch_input, batch_hidden1, hidden_weights1, hidden_bias1, relu);
        passHidden(batch_hidden1, batch_hidden2, hidden_weights2, hidden_bias2, relu);
        passOutput(batch_hidden2, batch_output, output_weights, output_bias, softmax);

        // Process predictions for the batch
        for (int b = 0; b < current_batch_size; ++b) {
            int predicted_label = max_element(batch_output[b].begin(), batch_output[b].end()) - batch_output[b].begin();
            test_predictions.push_back(predicted_label);
            correct_count += (predicted_label == test_labels[i + b]) ? 1 : 0;
        }
    }
    cout << "Accuracy on test data: " << (double)correct_count / test_vectors.size() << endl;

    // reorder train predictions back
    vector<int> train_predictions;
    for (int i = 0; i < total_training; ++i) {
        train_predictions.push_back(train_predictions_map[i]);
    }

    writePredictions("train_predictions.csv", train_predictions);
    writePredictions("test_predictions.csv", test_predictions);

    return 0;
}
