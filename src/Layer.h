//
// Created by Alina Tsykynovska on 14.10.2023.
//

#ifndef PV021_PROJECT_LAYER_H
#define PV021_PROJECT_LAYER_H

class Neuron;

class Layer {
private:
    Layer* prevLayer;
    Neuron* neurons;
    int neuronsSize;
    int* input;
    int inputSize;
public:
    /**
     * A constructor for a hidden layer.
     * @param prevLayer
     * @param neurons a pointer to an array of neurons
     * @param size size of a neurons array
     */
    Layer(Layer *prevLayer, Neuron *neurons, int size);

    /**
     * A constructor for an input layer.
     * @param input a pointer to an input array
     * @param inputSize
     */
    Layer(int *input, int inputSize);

    /**
     *
     * @return nullptr for an input layer
     */
    Neuron *getNeurons() const;

    void setNeurons(Neuron *neurons);

    /**
     *
     * @return nullptr for an input layer
     */
    Layer *getPrevLayer() const;

    void setPrevLayer(Layer *prevLayer);

    int getNeuronsSize() const;

    void setNeuronsSize(int size);

    /**
     *
     * @return nullptr for a hidden layer
     */
    int *getInput() const;

    void setInput(int *input);

    int getInputSize() const;

    void setInputSize(int inputSize);

    virtual ~Layer();
};


#endif //PV021_PROJECT_LAYER_H
