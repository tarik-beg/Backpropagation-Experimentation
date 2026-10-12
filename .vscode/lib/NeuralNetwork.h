#pragma once
#include <vector>
#include "Matrix.h"

namespace NeuralNetwork{

    class SlowNeuralNetwork{
    private:
        /*
            Each row represents the weights connected to a singular neuron in the next layer
            Matrix dimensions are LayerSize(k) x LayerSize(k-1)
            Index 0 is an empty placeholder (layer 0 has no weights) so layer k is at index k
        */
        std::vector<MatrixF> _weights; 
            
        /*
            Matrices are actually column vectors represeinting the bias of neurons in a layer
            Index 0 is an empty placeholder like in _weights
        */
        std::vector<MatrixF> _biases;

        /*
            Neuron values(after applying the activation function) - also column vectors
        */
        std::vector<MatrixF> _layers; //0th layer is input and last is output
        /*
            d(k,i)=s(k,i)*f'(z(k,i))
            Index 0 is an empty placeholder like in _weights
        */
        std::vector<MatrixF> _d;

        /*
            dC/dw(k,i,j) - same dimensions as _weights, including the empty placeholder at index 0
        */
        std::vector<MatrixF> _weightGradients;

        /*
            dC/db(k,i) - same dimensions as _biases, including the empty placeholder at index 0
        */
        std::vector<MatrixF> _biasGradients;

        int _trainingBatchSize;

        /*
            Per output neuron (y(i)-a(L,i))^2 - n(L) x 1, sums to C. Statistics only, not used by backprop
        */
        MatrixF _cost;

        size_t _numOfLayers;
        std::vector<size_t> _layerSizes;

        float _learningRate=0.1f;
        
        void CalculateD(int label);
        void CalculateGradients();
        void UpdateParameters(int batchSamples);

    public:
        SlowNeuralNetwork();
        SlowNeuralNetwork(std::vector<size_t> layerSizes, int trainingBatchSize=1);


        size_t NumOfLayers();
        size_t LayerSize(size_t k);

        static float ReLU(float x);
        static float Sigmoid(float x);

        static float SigmoidDerivative(float x);
        static float ReLUDerivative(float x);

        float GetW(size_t k, size_t i, size_t j);
        void SetW(size_t k, size_t i, size_t j, float value);

        float GetB(size_t k, size_t i);
        void SetB(size_t k, size_t i, float value);

        float GetD(size_t k, size_t i);
        void SetD(size_t k, size_t i, float value);

        void Train(); //MNIST dataset training - no control for now 

        float Test();

        MatrixF& ForwardPass(const MatrixF& Input); //returns a reference to the last layer(output)

    };




}