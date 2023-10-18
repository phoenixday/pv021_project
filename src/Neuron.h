//
// Created by Alina Tsykynovska on 14.10.2023.
//

#ifndef PV021_PROJECT_NEURON_H
#define PV021_PROJECT_NEURON_H

class Layer;

class Neuron {
private:
    Layer* layer;
    int* weights;
    int bias;
    int output;
public:
    Neuron(int *weights, int bias);

    Layer *getLayer() const;

    void setLayer(Layer *layer);

    int *getWeights() const;

    void setWeights(int *weights);

    int getBias() const;

    void setBias(int bias);

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
