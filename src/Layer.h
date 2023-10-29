//
// Created by Alina Tsykynovska on 14.10.2023.
//

#ifndef PV021_PROJECT_LAYER_H
#define PV021_PROJECT_LAYER_H

#include <vector>

class Neuron;

class Layer {
private:
    Layer* prevLayer;
    std::vector<Neuron> neurons;
    std::vector<int> input;

public:
    /**
     * A constructor for a hidden layer.
     * @param prevLayer
     * @param neurons a pointer to an array of neurons
     */
    Layer(Layer *prevLayer, std::vector<Neuron> neurons);

    /**
     * A constructor for an input layer.
     * @param input a pointer to an input array
     * @param inputSize
     */
    explicit Layer(std::vector<int> input);

    /**
     *
     * @return nullptr for an input layer
     */
    std::vector<Neuron> getNeurons() const;

    void setNeurons(std::vector<Neuron> newNeurons);

    /**
     *
     * @return nullptr for an input layer
     */
    Layer *getPrevLayer() const;

    void setPrevLayer(Layer *newPrevLayer);

    /**
     *
     * @return null for a hidden layer
     */
    std::vector<int> getInput() const;

    void setInput(std::vector<int> newInput);

    /**
     * makes sense only for an output layer
     * @return output of the one and only neuron
     */
    int getOutput();

    virtual ~Layer();
};


#endif //PV021_PROJECT_LAYER_H
