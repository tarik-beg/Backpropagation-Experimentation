
#include <cstdio>
#include <iostream>

#include <vector>
#include <cmath>
#include "NeuralNetwork.h"
#include "MNISTReader.h"

using std::vector;

/*
MNISTReader train("MNISTData/train-images.idx3-ubyte", "MNISTData/train-labels.idx1-ubyte");
MatrixF image;   // reuse one matrix for every sample
int label;

while (train.Next(image, label)) {
    // image: 784x1 column vector, pixels scaled to [0,1], ready to use as layer 0
    // label: the digit 0-9
}
train.Reset();             // back to the start for the next epoch
train.Get(123, image, label); // jump to one sample, e.g. by walking a shuffled list of indices
*/

namespace NeuralNetwork{

SlowNeuralNetwork::SlowNeuralNetwork(){
    _numOfLayers=0;
    _trainingBatchSize=1;
}

SlowNeuralNetwork::SlowNeuralNetwork(std::vector<size_t> layerSizes, int trainingBatchSize){
    _layerSizes=std::move(layerSizes);
    _numOfLayers=_layerSizes.size();
    _trainingBatchSize=trainingBatchSize;

    //Empty placeholders for layer 0 so layer k is at index k, matching the formulas
    _weights.push_back(MatrixF());
    _biases.push_back(MatrixF());
    _d.push_back(MatrixF());
    _weightGradients.push_back(MatrixF());
    _biasGradients.push_back(MatrixF());

    for(int i= 1;i<_numOfLayers;i++){
        //He initialisation (suited to ReLU): uniform in [-sqrt(6/n(k-1)), sqrt(6/n(k-1))] keeps z from growing with the layer size
        float limit=std::sqrt(6.0f/_layerSizes[i-1]);
        _weights.push_back(MatrixF::random(_layerSizes[i], _layerSizes[i-1], -limit, limit));
        _biases.push_back(MatrixF::random(_layerSizes[i], 1));
        _d.push_back(MatrixF(_layerSizes[i], 1));
        _weightGradients.push_back(MatrixF(_layerSizes[i], _layerSizes[i-1]));
        _biasGradients.push_back(MatrixF(_layerSizes[i], 1));
    }

    for(int i=0;i<_numOfLayers;i++){
        _layers.push_back(MatrixF(_layerSizes[i], 1));
    }

    _cost=MatrixF(_layerSizes[_numOfLayers-1], 1);
}

size_t SlowNeuralNetwork::NumOfLayers(){
    return _numOfLayers;
}

size_t SlowNeuralNetwork::LayerSize(size_t k){
    return _layerSizes[k];
}

/*
    Getters and setters throw std::out_of_range for k=0 (the empty placeholder) or k past the output layer.
*/
float SlowNeuralNetwork::GetW(size_t k, size_t i, size_t j){
    return _weights.at(k)(i, j);
}

void SlowNeuralNetwork::SetW(size_t k, size_t i, size_t j, float value){
    _weights.at(k)(i, j)=value;
}


float SlowNeuralNetwork::GetB(size_t k, size_t i){
    return _biases.at(k)(i, 0);
}

void SlowNeuralNetwork::SetB(size_t k, size_t i, float value){
    _biases.at(k)(i, 0)=value;
}

float SlowNeuralNetwork::GetD(size_t k, size_t i){
    return _d.at(k)(i, 0);
}

void SlowNeuralNetwork::SetD(size_t k, size_t i, float value){
    _d.at(k)(i, 0)=value;
}

float SlowNeuralNetwork::ReLU(float x){
    return x>0.0f ? x : 0.0f;
}
float SlowNeuralNetwork::Sigmoid(float x){
    return 1.0f/(1.0f + std::exp(-x));
}


float SlowNeuralNetwork::SigmoidDerivative(float x){
    return x*(1-x);
}
float SlowNeuralNetwork::ReLUDerivative(float x){
    return x>0 ? 1.0f : 0.0f;
}

MatrixF& SlowNeuralNetwork::ForwardPass(const MatrixF& Input){
    _layers[0]=Input;
    for(int i=1;i<_numOfLayers-1;i++){
        _layers[i]=(_weights[i]*_layers[i-1] + _biases[i]).apply(ReLU);
        
    }
    return _layers[_numOfLayers-1]=
        (_weights[_numOfLayers-1]*_layers[_numOfLayers-2] + _biases[_numOfLayers-1]).apply(Sigmoid);
    
        
}

void SlowNeuralNetwork::CalculateD(int label){
    //d for last layer
    {
        MatrixF temp(_layerSizes[_numOfLayers-1],1,0.0f);
        temp(label,0)=1.0f;
        _d.at(_numOfLayers-1)=
            2*
            (_layers.at(_numOfLayers-1)-temp).hadamard(
            _layers.at(_numOfLayers-1).apply(SigmoidDerivative));
    }

    for(int k=_numOfLayers-2;k>0;k--){ //Over each layer
        
        for(int i=0;i<_layerSizes[k];i++){//D for each neuron in the layer
            float dTemp=0;
            for(int j=0;j<_layerSizes[k+1];j++){//Sum according to formula for d
                dTemp+=GetD(k+1,j)*GetW(k+1,j,i);
            }
            dTemp*=ReLUDerivative(_layers[k](i,0));
            SetD(k,i,dTemp);

        }
    }
}

void SlowNeuralNetwork::CalculateGradients(){
    for(int k=_numOfLayers-1;k>0;k--){
        _biasGradients[k]+=_d[k];
        for(int i=0;i<_layerSizes[k];i++){

            
            for(int j=0;j<_layerSizes[k-1];j++){
                _weightGradients[k](i,j)+=GetD(k,i)*_layers[k-1](j,0);
            }
        }
    }
}

/*
    Gradient descent step using the gradients averaged over the batch, then resets them for the next batch
*/
void SlowNeuralNetwork::UpdateParameters(int batchSamples){
    float step=_learningRate/batchSamples;
    for(int k=1;k<_numOfLayers;k++){
        _weights[k]-=step*_weightGradients[k];
        _biases[k]-=step*_biasGradients[k];
        _weightGradients[k].fill(0.0f);
        _biasGradients[k].fill(0.0f);
    }
}

void SlowNeuralNetwork::Train(){
    MNISTReader train("MNISTData/train-images.idx3-ubyte", "MNISTData/train-labels.idx1-ubyte");
    MatrixF image;   // reuse one matrix for every sample
    int label;
    
    int count=0;
    int batchCount=0; //samples accumulated in the gradients since the last update
    while(train.Next(image,label)){
        count++;
        ForwardPass(image);


        CalculateD(label);
        CalculateGradients();

        batchCount++;
        if(batchCount==_trainingBatchSize){
            UpdateParameters(batchCount);
            batchCount=0;
        }
    }

    if(batchCount>0){ //last, partial batch
        UpdateParameters(batchCount);
    }
}

/*
MatrixF cost(_layerSizes[_numOfLayers-1],1);
cost(label,0)=1.0f;
cost= (cost-output).apply([](float x){return x*x;});
_cost+=cost;
*/

float SlowNeuralNetwork::Test(){
    MNISTReader test(MNISTReader::TestImagesPath,MNISTReader::TestLabelsPath);
    MatrixF image;
    int label;

    MatrixF cost(_layerSizes[_numOfLayers-1],1);
    int i=0;
    int correct=0;
    while(test.Next(image,label)){
        MatrixF& output=ForwardPass(image);
        MatrixF expected(_layerSizes[_numOfLayers-1],1);
        expected(label,0)=1.0f;
        cost+= (expected-output).apply([](float x){return x*x;});

        size_t guess=0; //output neuron with the highest activation
        for(size_t n=1;n<output.rows();n++){
            if(output(n,0)>output(guess,0)) guess=n;
        }
        if(((int)guess)==label) correct++;
        i++;
    }

    cost*=1.0f/i;

    std::cout << "Average squared error per output neuron:\n" << cost;
    return ((float)correct)/i; //accuracy, 0 to 1
}

}

