#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <vector>

using namespace std;

double relu(double x);
double reluDerivative(double x);
void softmax(vector<double> &x);

#endif // ACTIVATION_FUNCTIONS_H
