//
// Created by Alina Tsykynovska on 14.10.2023.
//

#include "Neuron.h"
#include "Layer.h"

Layer::Layer(Layer *prevLayer, Neuron *neurons, int size) : prevLayer(prevLayer), neurons(neurons), neuronsSize(size) {
    input = nullptr;
    inputSize = 0;

    for(int i = 0; i < neuronsSize; i++) {
        neurons[i].setLayer(this);
    }
}

Layer::Layer(int *input, int inputSize) : input(input), inputSize(inputSize) {
    prevLayer = nullptr;
    neurons = nullptr;
    neuronsSize = 0;
}

Neuron *Layer::getNeurons() const {
    return neurons;
}

void Layer::setNeurons(Neuron *neurons) {
    this->neurons = neurons;
}

Layer *Layer::getPrevLayer() const {
    return prevLayer;
}

void Layer::setPrevLayer(Layer *prevLayer) {
    this->prevLayer = prevLayer;
}

int Layer::getNeuronsSize() const {
    return neuronsSize;
}

void Layer::setNeuronsSize(int size) {
    Layer::neuronsSize = size;
}

int *Layer::getInput() const {
    return input;
}

void Layer::setInput(int *input) {
    Layer::input = input;
}

int Layer::getInputSize() const {
    return inputSize;
}

void Layer::setInputSize(int inputSize) {
    Layer::inputSize = inputSize;
}

Layer::~Layer() {
    delete[] neurons;
    neurons = nullptr;
    delete[] input;
    input = nullptr;
    prevLayer = nullptr;
}
