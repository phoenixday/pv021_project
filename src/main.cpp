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
        
        for (int idx = 0; idx < training_size; ++idx) {
            int i = indices[idx];
            // reset to 0
            fill(hidden1.begin(), hidden1.end(), 0.0);
            fill(hidden2.begin(), hidden2.end(), 0.0);
            fill(output.begin(), output.end(), 0.0);

            // forward pass
            passHidden(train_vectors[i], hidden1, hidden_weights1, hidden_bias1, relu);
            passHidden(hidden1, hidden2, hidden_weights2, hidden_bias2, relu);
            passOutput(hidden2, output, output_weights, output_bias, softmax);

            // add label
            int predicted_label = max_element(output.begin(), output.end()) - output.begin();
            if (epoch == EPOCHS) train_predictions_map[i] = predicted_label;

            // compute error
            vector<double> error_output(OUTPUT_SIZE, 0.0);
            for (int o = 0; o < OUTPUT_SIZE; ++o) {
                error_output[o] = (o == train_labels[i]) ? 1.0 - output[o] : 0.0 - output[o];
            }

            // backpropagation
            vector<double> d_output(OUTPUT_SIZE, 0.0);
            for (int o = 0; o < OUTPUT_SIZE; ++o) {
                d_output[o] = error_output[o]; 
            }

            vector<double> d_hidden2(HIDDEN_SIZE2, 0.0);
            backpropagationHidden(hidden2, d_hidden2, d_output, output_weights, reluDerivative);

            vector<double> d_hidden1(HIDDEN_SIZE1, 0.0);
            backpropagationHidden(hidden1, d_hidden1, d_hidden2, hidden_weights2, reluDerivative);

            // update weights
            updateWeightsWithAdam(hidden_weights1, hidden_bias1, d_hidden1, train_vectors[i],
                                m_hidden_weights1, v_hidden_weights1, epoch);

            updateWeightsWithAdam(hidden_weights2, hidden_bias2, d_hidden2, hidden1,
                                m_hidden_weights2, v_hidden_weights2, epoch);

            updateWeightsWithAdam(output_weights, output_bias, d_output, hidden2,
                                  m_output_weights, v_output_weights, epoch);
        }

        // VALIDATION
        int correct_count = 0;
        for (int i = training_size; i < total_training; ++i) {
            // reset to 0
            fill(hidden1.begin(), hidden1.end(), 0.0);
            fill(hidden2.begin(), hidden2.end(), 0.0);
            fill(output.begin(), output.end(), 0.0);

            // forward pass
            passHidden(train_vectors[i], hidden1, hidden_weights1, hidden_bias1, relu);
            passHidden(hidden1, hidden2, hidden_weights2, hidden_bias2, relu);
            passOutput(hidden2, output, output_weights, output_bias, softmax);
            
            // add label
            int predicted_label = max_element(output.begin(), output.end()) - output.begin();
            if (epoch == EPOCHS) train_predictions_map[i] = predicted_label;
            correct_count += (predicted_label == train_labels[i]) ? 1 : 0;
        }
        cout << "Epoch: " << epoch << ", accuracy on validation data: " << (double)correct_count / validation_size << endl;
    }

    // TESTING
    int correct_count = 0;
    vector<int> test_predictions;
    for (size_t i = 0; i < test_vectors.size(); ++i) {
        // reset to 0
        fill(hidden1.begin(), hidden1.end(), 0.0);
        fill(hidden2.begin(), hidden2.end(), 0.0);
        fill(output.begin(), output.end(), 0.0);

        // forward pass
        passHidden(test_vectors[i], hidden1, hidden_weights1, hidden_bias1, relu);
        passHidden(hidden1, hidden2, hidden_weights2, hidden_bias2, relu);
        passOutput(hidden2, output, output_weights, output_bias, softmax);

        // add label
        int predicted_label = max_element(output.begin(), output.end()) - output.begin();
        test_predictions.push_back(predicted_label);
        correct_count += (predicted_label == test_labels[i]) ? 1 : 0;
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
