//
// Created by Alina Tsykynovska on 14.10.2023.
//

#ifndef PV021_PROJECT_NEURON_H
#define PV021_PROJECT_NEURON_H

#include <vector>

class Layer;

class Neuron {
private:
    Layer* layer;
    std::vector<int> weights;
    int bias;
    int output;
public:
    Neuron(std::vector<int> weights, int bias);

    Layer *getLayer() const;

    void setLayer(Layer *newLayer);

    std::vector<int> getWeights() const;

    void setWeights(std::vector<int> newWeights);

    int getBias() const;

    void setBias(int newBias);

    /**
     *
     * @return a stored output, calculation is done in calculateSumWeights(), assigning in activate()
     */
    int getOutput() const;

    int calculateSumWeights();

    /**
     *
     * @return calculates and assigns output
     */
    int activate();

    virtual ~Neuron();
};


#endif //PV021_PROJECT_NEURON_H
