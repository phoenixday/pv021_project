cmake_minimum_required(VERSION 3.26)
project(pv021_project)

set(CMAKE_CXX_STANDARD 14)

include_directories(src)

add_executable(pv021_project
        data/fashion_mnist_test_labels.csv
        data/fashion_mnist_test_vectors.csv
        data/fashion_mnist_train_labels.csv
        data/fashion_mnist_train_vectors.csv
        evaluator/evaluate.py
        evaluator/README.md
        src/main.cpp
        src/Neuron.cpp
        src/Neuron.h
        src/README.md
        example_test_predictions.csv
        network
        README.md
        run.sh
        src/Layer.h
        src/Layer.cpp
)
