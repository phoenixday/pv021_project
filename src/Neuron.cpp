//
// Created by Alina Tsykynovska on 14.10.2023.
//

#include "Neuron.h"
#include "Layer.h"

Neuron::Neuron(int *weights, int bias) : weights(weights), bias(bias) {}

Layer *Neuron::getLayer() const {
    return layer;
}

void Neuron::setLayer(Layer *layer) {
    Neuron::layer = layer;
}

int *Neuron::getWeights() const {
    return weights;
}

void Neuron::setWeights(int *weights) {
    Neuron::weights = weights;
}

int Neuron::getBias() const {
    return bias;
}

void Neuron::setBias(int bias) {
    Neuron::bias = bias;
}

int Neuron::getOutput() const {
    return output;
}

int Neuron::calculateSumWeights() {
    int sum = bias;
    Layer* prevLayer = layer->getPrevLayer();
    if (prevLayer != nullptr) {
        Neuron* neurons = prevLayer->getNeurons();
        for(int i = 0; i < prevLayer->getNeuronsSize(); i++) {
            sum += neurons[i].getOutput() * weights[i];
        }
    } else {
        int* input = layer->getInput();
        for(int i = 0; i < layer->getInputSize(); i++) {
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
    delete[] weights;
    weights = nullptr;
    layer = nullptr;
}
