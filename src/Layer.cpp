//
// Created by Alina Tsykynovska on 14.10.2023.
//

#include "Neuron.h"
#include "Layer.h"

#include <utility>

Layer::Layer(Layer *prevLayer, std::vector<Neuron> neurons) : prevLayer(prevLayer), neurons(std::move(neurons)) {
    for(auto & neuron : this->neurons) {
        neuron.setLayer(this);
        neuron.activate();
    }
}

Layer::Layer(std::vector<int> input) : input(std::move(input)) {
    prevLayer = nullptr;
}

std::vector<Neuron> Layer::getNeurons() const {
    return neurons;
}

void Layer::setNeurons(std::vector<Neuron> newNeurons) {
    Layer::neurons = std::move(newNeurons);
}

Layer *Layer::getPrevLayer() const {
    return prevLayer;
}

void Layer::setPrevLayer(Layer *newPrevLayer) {
    this->prevLayer = newPrevLayer;
}

std::vector<int> Layer::getInput() const {
    return input;
}

void Layer::setInput(std::vector<int> newInput) {
    Layer::input = std::move(newInput);
}

int Layer::getOutput() {
    return neurons[0].getOutput();
}

Layer::~Layer() {
    prevLayer = nullptr;
}
