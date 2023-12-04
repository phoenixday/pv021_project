#include <iostream> // allowed
#include <fstream> // for reading/writing csv
#include <sstream> // same
#include <vector>
#include <algorithm> // allowed
#include <random> // allowed

using namespace std;

class Matrix {
public:
    vector<vector<double>> data;

    Matrix(int rows, int cols) : data(rows, vector<double>(cols, 0.0)) {}

    int numRows() const { return data.size(); }
    int numCols() const { return data[0].size(); }

    vector<double>& operator[](int index) { return data[index]; }
    const vector<double>& operator[](int index) const { return data[index]; }

    Matrix transpose() const {
        Matrix result(numCols(), numRows());
        for (int i = 0; i < numRows(); ++i)
            for (int j = 0; j < numCols(); ++j)
                result[j][i] = data[i][j];
        return result;
    }

    static Matrix multiply(const Matrix& a, const Matrix& b) {
        if (a.numCols() != b.numRows()) {
            cerr << "Dimension mismatch in matrix multiplication" << endl;
            exit(1); // or handle the error in a more suitable way for your application
        }
        
        Matrix result(a.numRows(), b.numCols());
        for (int i = 0; i < a.numRows(); ++i)
            for (int j = 0; j < b.numCols(); ++j)
                for (int k = 0; k < a.numCols(); ++k)
                    result[i][j] += a[i][k] * b[k][j];
        return result;
    }
};

// Hyperparameters
const int EPOCHS = 8;
const int INPUT_SIZE = 784; // 28x28
const int HIDDEN_SIZE1 = 256;
const int HIDDEN_SIZE2 = 128;
const int HIDDEN_SIZE3 = 128;
const int OUTPUT_SIZE = 10;
const double LEARNING_RATE = 0.001;
const double BETA1 = 0.9;
const double BETA2 = 0.999;
const double EPSILON = 1e-8;
const double LAMBDA = 1e-4;

double relu(double x) {
    return max(0.0, x);
}

double reluDerivative(double x) {
    return (x > 0) ? 1.0 : 0.0;
}

void softmax(Matrix& mat) {
    for (int j = 0; j < mat.numCols(); ++j) {
        double maxVal = mat[0][j];
        for (int i = 1; i < mat.numRows(); ++i) {
            if (mat[i][j] > maxVal) {
                maxVal = mat[i][j];
            }
        }

        double sum = 0.0;
        for (int i = 0; i < mat.numRows(); ++i) {
            mat[i][j] = exp(mat[i][j] - maxVal);
            sum += mat[i][j];
        }

        for (int i = 0; i < mat.numRows(); ++i) {
            mat[i][j] /= sum;
        }
    }
}

void initializeWithDis(Matrix& input, normal_distribution<>& dis, mt19937& gen) {
    for (int i = 0; i < input.numRows(); ++i) {
        for (int j = 0; j < input.numCols(); ++j) {
            input[i][j] = dis(gen);
        }
    }
}

void resetLayer(Matrix& input) {
    for (int i = 0; i < input.numRows(); ++i) {
        for (int j = 0; j < input.numCols(); ++j) {
            input[i][j] = 0.0;
        }
    }
}

void passHidden(const Matrix& prev_layer, Matrix& layer, const Matrix& weights, const vector<double>& bias) {
    Matrix transposed_prev_layer = prev_layer.transpose();
    Matrix product = Matrix::multiply(weights, transposed_prev_layer);

    if (product.numRows() != layer.numRows() || product.numCols() != layer.numCols()) {
        cerr << "Dimension mismatch in passHidden function" << endl;
        exit(1);
    }

    for (int i = 0; i < product.numRows(); ++i) {
        for (int j = 0; j < product.numCols(); ++j) {
            layer[i][j] = relu(product[i][j] + bias[i]);
        }
    }
}

void passOutput(const Matrix& layer, Matrix& output, const Matrix& weights, const vector<double>& bias) {
    Matrix transposed_layer = layer.transpose();
    Matrix product = Matrix::multiply(weights, transposed_layer);

    for (int i = 0; i < product.numRows(); ++i) {
        for (int j = 0; j < product.numCols(); ++j) {
            output[i][j] = product[i][j] + bias[i];
        }
    }
    softmax(output); 
}

void backpropagationHidden(const Matrix& layer, Matrix& d_layer, const Matrix& d_next_layer, const Matrix& next_layer_weights) {
    Matrix transposed_weights = next_layer_weights.transpose();

    for (int i = 0; i < d_layer.numRows(); ++i) {
        for (int j = 0; j < d_layer.numCols(); ++j) {
            double error = 0.0;
            for (int k = 0; k < d_next_layer.numRows(); ++k) {
                error += d_next_layer[k][j] * transposed_weights[i][k];
            }
            d_layer[i][j] = error * reluDerivative(layer[i][j]);
        }
    }
}

void updateWeightsWithAdam(Matrix& weights, vector<double>& bias, const Matrix& gradients, const Matrix& inputs, Matrix& m_weights, Matrix& v_weights, int epoch) {
    for (int i = 0; i < weights.numRows(); ++i) {
        for (int j = 0; j < weights.numCols(); ++j) {
            m_weights[i][j] = BETA1 * m_weights[i][j] + (1.0 - BETA1) * gradients[i][j];
            v_weights[i][j] = BETA2 * v_weights[i][j] + (1.0 - BETA2) * pow(gradients[i][j], 2);

            double m_corr = m_weights[i][j] / (1.0 - pow(BETA1, epoch));
            double v_corr = v_weights[i][j] / (1.0 - pow(BETA2, epoch));

            weights[i][j] += LEARNING_RATE * (m_corr / (sqrt(v_corr) + EPSILON) + LAMBDA * weights[i][j]);
        }
    }

    for (size_t i = 0; i < bias.size(); ++i) {
        bias[i] += LEARNING_RATE * gradients[i][0];
    }
}


vector<Matrix> readVectors(const string &filename) {
    cout << "Reading vectors from " << filename << " ..." << endl;
    ifstream file(filename);
    string line;

    vector<Matrix> data;
    while (getline(file, line)) {
        stringstream lineStream(line);
        string cell;
        vector<double> row;
        while (getline(lineStream, cell, ',')) {
            row.push_back(stod(cell));
        }
        Matrix matrixRow(1, row.size());
        for (size_t j = 0; j < row.size(); ++j) {
            matrixRow[0][j] = row[j];
        }
        data.push_back(matrixRow);
    }

    if (data.empty()) {
        cerr << filename << " is empty or could not be read." << endl;
    } else {
        cout << "Vectors read!" << endl;
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
    } else {
        cout << "Labels read!" << endl;
    }
    return data;
}

void writePredictions(const string& filename, const vector<int>& data) {
    cout << "Writing predictions to " << filename << " ..." << endl;
    ofstream file(filename);
    for (int val : data) {
        file << val << endl;
    }
    cout << "Predictions written!" << endl;
    file.close();
}

int main() {
    vector<Matrix> train_vectors = readVectors("data/fashion_mnist_train_vectors.csv");
    vector<int> train_labels = readLabels("data/fashion_mnist_train_labels.csv");
    vector<Matrix> test_vectors = readVectors("data/fashion_mnist_test_vectors.csv");
    vector<int> test_labels = readLabels("data/fashion_mnist_test_labels.csv");

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
    Matrix hidden_weights1(INPUT_SIZE, HIDDEN_SIZE1);
    Matrix hidden_weights2(HIDDEN_SIZE1, HIDDEN_SIZE2);
    Matrix hidden_weights3(HIDDEN_SIZE2, HIDDEN_SIZE3);
    Matrix output_weights(HIDDEN_SIZE3, OUTPUT_SIZE);
    initializeWithDis(hidden_weights1, dis_hidden1, gen);
    initializeWithDis(hidden_weights2, dis_hidden2, gen);
    initializeWithDis(hidden_weights3, dis_hidden3, gen);
    initializeWithDis(output_weights, dis_output, gen);

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
    Matrix hidden1(1, HIDDEN_SIZE1);
    Matrix hidden2(1, HIDDEN_SIZE2);
    Matrix hidden3(1, HIDDEN_SIZE3);
    Matrix output(1, OUTPUT_SIZE);

    // Initialize Adam weights
    Matrix m_hidden_weights1(INPUT_SIZE, HIDDEN_SIZE1);
    Matrix v_hidden_weights1(INPUT_SIZE, HIDDEN_SIZE1);
    Matrix m_hidden_weights2(HIDDEN_SIZE1, HIDDEN_SIZE2);
    Matrix v_hidden_weights2(HIDDEN_SIZE1, HIDDEN_SIZE2);
    Matrix m_hidden_weights3(HIDDEN_SIZE2, HIDDEN_SIZE3);
    Matrix v_hidden_weights3(HIDDEN_SIZE2, HIDDEN_SIZE3);
    Matrix m_output_weights(HIDDEN_SIZE2, OUTPUT_SIZE);
    Matrix v_output_weights(HIDDEN_SIZE2, OUTPUT_SIZE);

    vector<int> train_predictions;

    // Prepare indices for shuffling
    vector<int> indices(train_vectors.size());
    iota(indices.begin(), indices.end(), 0);

    // Training
    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
        // Shuffle the training data and labels together
        shuffle(indices.begin(), indices.end(), gen);
        
        for (int idx = 0; idx < training_size; ++idx) {
            int i = indices[idx];

            // Reset hidden and output layers
            resetLayer(hidden1);
            resetLayer(hidden2);
            resetLayer(hidden3);
            resetLayer(output);

            // Forward pass
            passHidden(train_vectors[i], hidden1, hidden_weights1, hidden_bias1);
            passHidden(hidden1, hidden2, hidden_weights2, hidden_bias2);
            passHidden(hidden2, hidden3, hidden_weights3, hidden_bias3);
            passOutput(hidden3, output, output_weights, output_bias);

            // Write 
            int predicted_label = distance(output[0].begin(), max_element(output[0].begin(), output[0].end()));
            if (epoch == EPOCHS) train_predictions.push_back(predicted_label);

            // Compute the loss and error
            Matrix error_output(1, OUTPUT_SIZE);
            int actual_label = train_labels[idx];
            for (int i = 0; i < OUTPUT_SIZE; ++i) {
                error_output[0][i] = (i == actual_label) ? 1.0 - output[0][i] : -output[0][i];
            }

            // Backpropagation
            Matrix d_output = error_output;

            Matrix d_hidden3(1, HIDDEN_SIZE3);
            backpropagationHidden(hidden3, d_hidden3, d_output, output_weights);

            Matrix d_hidden2(1, HIDDEN_SIZE2);
            backpropagationHidden(hidden2, d_hidden2, d_hidden3, hidden_weights3);

            Matrix d_hidden1(1, HIDDEN_SIZE1);
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
        for (int i = training_size; i < total_training; ++i) {
            // Reset hidden and output layers
            resetLayer(hidden1);
            resetLayer(hidden2);
            resetLayer(hidden3);
            resetLayer(output);

            // Forward pass
            passHidden(train_vectors[i], hidden1, hidden_weights1, hidden_bias1);
            passHidden(hidden1, hidden2, hidden_weights2, hidden_bias2);
            passHidden(hidden2, hidden3, hidden_weights3, hidden_bias3);
            passOutput(hidden3, output, output_weights, output_bias);
            
            // Write
            int predicted_label = distance(output[0].begin(), max_element(output[0].begin(), output[0].end()));
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
        resetLayer(hidden1);
        resetLayer(hidden2);
        resetLayer(hidden3);
        resetLayer(output);

        // Forward pass with test vectors
        passHidden(test_vectors[i], hidden1, hidden_weights1, hidden_bias1);
        passHidden(hidden1, hidden2, hidden_weights2, hidden_bias2);
        passHidden(hidden2, hidden3, hidden_weights3, hidden_bias3);
        passOutput(hidden3, output, output_weights, output_bias);

        // Write
        int predicted_label = distance(output[0].begin(), max_element(output[0].begin(), output[0].end()));
        test_predictions.push_back(predicted_label);
        correct_count += (predicted_label == test_labels[i]) ? 1 : 0;
    }
    double accuracy = (double)correct_count / test_vectors.size() * 100;
    cout << "Final Test Accuracy: " << accuracy << "%" << endl;

    writePredictions("train_predictions.csv", train_predictions);
    writePredictions("test_predictions.csv", test_predictions);

    return 0;
}
