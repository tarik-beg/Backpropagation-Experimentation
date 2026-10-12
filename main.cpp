#include <iostream>
#include ".vscode/lib/NeuralNetwork.h"

int main(){
    const int epochs=3;
    const int batchSize=10;

    //784 input pixels, two hidden layers of 16, 10 outputs (one per digit)
    NeuralNetwork::SlowNeuralNetwork net({784, 16, 16, 10}, batchSize);

    std::cout << "Before training\n";
    float accuracy=net.Test();
    std::cout << "Accuracy: " << accuracy*100 << "%\n\n";

    for(int epoch=1;epoch<=epochs;epoch++){
        net.Train();

        std::cout << "After epoch " << epoch << "n";
        accuracy=net.Test();
        std::cout << "Accuracy: " << accuracy*100 << "%\n\n";
    }

    return 0;
}
