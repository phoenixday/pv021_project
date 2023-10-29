//
// Created by Alina Tsykynovska on 14.10.2023.
//

#include "Neuron.h"

#include <utility>
#include "Layer.h"

Neuron::Neuron(std::vector<int> weights, int bias) : weights(std::move(weights)), bias(bias) {}

Layer *Neuron::getLayer() const {
    return layer;
}

void Neuron::setLayer(Layer *newLayer) {
    Neuron::layer = newLayer;
}

std::vector<int> Neuron::getWeights() const {
    return weights;
}

void Neuron::setWeights(std::vector<int> newWeights) {
    Neuron::weights = std::move(newWeights);
}

int Neuron::getBias() const {
    return bias;
}

void Neuron::setBias(int newBias) {
    Neuron::bias = newBias;
}

int Neuron::getOutput() const {
    return output;
}

int Neuron::calculateSumWeights() {
    int sum = bias;
    Layer* prevLayer = layer->getPrevLayer();
    std::vector<Neuron> neurons = prevLayer->getNeurons();
    // if the previous layer is hidden
    if (!neurons.empty()) {
        for(int i = 0; i < neurons.size(); i++) {
            sum += neurons[i].getOutput() * weights[i];
        }
    }
    // if the previous layer is an input layer
    else {
        std::vector<int> input = prevLayer->getInput();
        for(int i = 0; i < input.size(); i++) {
            sum += input[i] * weights[i];
        }
    }
    return sum;
}

int Neuron::activate() {
    output = (calculateSumWeights() < 0) ? 0 : 1;
    return output;
}

Neuron::~Neuron() {
    layer = nullptr;
}
