#include <cstdio>
#include <utility>
#include <vector>
#include "Layer.h"
#include "Neuron.h"

void createNetwork(std::vector<int> input) {
    auto *inputLayer = new Layer(std::move(input));

    // adding neurons, their weights and biases manually (for now)
    auto *layer1 = new Layer(inputLayer, std::vector<Neuron>{
            Neuron(std::vector<int>{2, -2}, -1),
            Neuron(std::vector<int>{2, -2}, 3)
    });

    auto *outputLayer = new Layer(layer1, std::vector<Neuron>{
            Neuron(std::vector<int>{1, 1}, -2)
    });

    printf("The output is: %d\n", outputLayer->getOutput());
}

int main() {

    // trying to implement XOR problem

    createNetwork(std::vector<int>{0, 0}); // output 0
    createNetwork(std::vector<int>{0, 1}); // output 1
    createNetwork(std::vector<int>{1, 0}); // output 1
    createNetwork(std::vector<int>{1, 1}); // output 0

    return 0;
}


