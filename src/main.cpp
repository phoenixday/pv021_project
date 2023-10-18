#include <cstdio>
#include "Layer.h"
#include "Neuron.h"

int main() {
    int *input = new int[2]{0,1};
    auto *inputLayer = new Layer(input, 2);

    int *weights1_1 = new int[2]{2, -2};
    int *weights1_2 = new int[2]{2, -2};
    auto *neuron1_1 = new Neuron(weights1_1, -1);
    auto *neuron1_2 = new Neuron(weights1_2, 3);
    auto *neurons1 = new Neuron[2]{*neuron1_1, *neuron1_2};
    auto *layer1 = new Layer(inputLayer, neurons1, 2);

    int *weights2_1 = new int[2]{1, 1};
    auto *neuron2_1 = new Neuron(weights2_1, -2);
    auto *layer2 = new Layer(layer1, neurons1, 2);

    printf("The output is: %d\n", neuron2_1->getOutput());

    return 0;
}