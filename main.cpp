#include <iostream>
#include <vector>
#include <cmath>
#include <random>

class NeuralNetwork{
    public:
        NeuralNetwork(int inputSize, int hiddenLayerSize, int outputSize, int numHiddenLayers, float biasOutput, float learningRate)
        : inputSize(inputSize), hiddenLayerSize(hiddenLayerSize), outputSize(outputSize), numHiddenLayers(numHiddenLayers) {
            InitializeRandomWeightsAndBiases();
        }

        void AddLayer() {
            _add_layer();
        }

        void Train(std::vector<float> input, float target) {
            _train(input, target);
        }

        float predict(std::vector<float> input) {
            return _predict(input);
        }

    private:
        int inputSize, hiddenLayerSize, outputSize, numHiddenLayers;
        float biasOutput, learningRate;

        std::vector<std::vector<std::vector<float>>> weights;
        std::vector<std::vector<float>> biases;

        void _add_layer() {
            std::random_device rd;
            std::mt19937 gen(2024);
            std::uniform_real_distribution<> dis(-1.0, 1.0);

            // Adjust the number of hidden layers
            numHiddenLayers++;

            // Add new weights and biases for the new hidden layer
            weights.insert(weights.begin() + numHiddenLayers - 1, 
                std::vector<std::vector<float>>(hiddenLayerSize, std::vector<float>(hiddenLayerSize, dis(gen))));
            biases.insert(biases.begin() + numHiddenLayers - 1, 
                std::vector<float>(hiddenLayerSize, dis(gen)));

            // Adjust output layer connections
            weights.push_back(std::vector<std::vector<float>>(outputSize, std::vector<float>(hiddenLayerSize, dis(gen))));
            biases.push_back(std::vector<float>(outputSize, dis(gen)));
        }

        void _train(std::vector<float> input, float target) {
            std::vector<float> current = input;
            std::vector<std::vector<float>> layer_outputs;
            layer_outputs.push_back(current);

            // Forward pass through hidden layers
            for (int i = 0; i < numHiddenLayers; i++) {
                std::vector<float> nextLayer(hiddenLayerSize, 0.0f);
                for (int j = 0; j < hiddenLayerSize; j++) {
                    float sum = biases[i][j];
                    for (int k = 0; k < current.size(); k++) {
                        sum += current[k] * weights[i][j][k];
                    }
                    nextLayer[j] = ReLU(sum);
                }
                current = nextLayer;
                layer_outputs.push_back(current);
            }

            // Forward pass to the output layer
            float output = biasOutput;
            for (int i = 0; i < hiddenLayerSize; i++) {
                output += current[i] * weights[numHiddenLayers][0][i];
            }
            //float yPred = Sigmoid(output);
            
            // Calculate error
            //float error = MSE(target, yPred);
            //float d_error = MSEDeriv(target, yPred);
            float error = MSE(target, output);
            float d_error = MSEDeriv(target, output);

            // Backpropagation for the output neuron
            float output_delta = d_error * SigmoidDeriv(output);

            // Update weights and biases for the output neuron
            for (int i = 0; i < hiddenLayerSize; i++) {
                weights[numHiddenLayers][0][i] -= learningRate * output_delta * layer_outputs[numHiddenLayers][i];
            }
            biasOutput -= learningRate * output_delta;

            // Backpropagate the error through hidden layers
            std::vector<float> delta(hiddenLayerSize, 0.0f);
            for (int layer = numHiddenLayers - 1; layer >= 0; --layer) {
                std::vector<float> nextDelta(hiddenLayerSize, 0.0f);
                for (int j = 0; j < hiddenLayerSize; j++) {
                    float error = 0.0f;
                    for (int k = 0; k < weights[layer + 1].size(); k++) {
                        error += delta[k] * weights[layer + 1][k][j];
                    }
                    float delta_j = error * ReLUDeriv(layer_outputs[layer + 1][j]);
                    nextDelta[j] = delta_j;

                    // Update weights and biases for this neuron
                    for (int k = 0; k < layer_outputs[layer].size(); k++) {
                        weights[layer][j][k] -= learningRate * delta_j * layer_outputs[layer][k];
                    }
                    biases[layer][j] -= learningRate * delta_j;
                }
                delta = nextDelta;
            }
        }


        float _predict(std::vector<float> input) {
            std::vector<float> current = input;
            std::vector<std::vector<float>> layer_outputs;
            layer_outputs.push_back(current);

            // forward pass through hiddens
            for (int i = 0; i < numHiddenLayers; i++) {
                std::vector<float> nextLayer(hiddenLayerSize, 0.0f);

                for (int j = 0; j < hiddenLayerSize; j++) {
                    float sum = biases[i][j];

                    for (int k=0; k < current.size(); k++) {
                        sum += current[k] * weights[i][j][k];
                    }
                    nextLayer[j] = ReLU(sum);
                }

                current = nextLayer;
                layer_outputs.push_back(current);
            }

            // forward pass to output
            float output = biasOutput;
            for (int i=0; i < hiddenLayerSize; i++) {
                output += current[i] * weights[numHiddenLayers][0][i];
            }
            //float ypred = Sigmoid(output);
            //return ypred

            return output;
        }


        float Sigmoid(float x) {
            return 1 / (1 + exp(-x));
        }

        float SigmoidDeriv(float x) {
            return Sigmoid(x) * (1 - Sigmoid(x));
        }

        float ReLU(float x) {
            if (x > 0) {
                return x;
            }
            return 0;
        }

        float ReLUDeriv(float x) {
            if (x > 0) {
                return 1;
            }
            return 0;
        }

        float MSE(float y, float yPred) {
            return 0.5 * (y - yPred) * (y - yPred);
        }

        float MSEDeriv(float y, float yPred) {
            return yPred - y;
        }

        void InitializeRandomWeightsAndBiases() {
            std::random_device rd;
            std::mt19937 gen(2024);
            std::uniform_real_distribution<> dis(-1.0, 1.0);

            // Input layer to first hidden layer
            weights.push_back(std::vector<std::vector<float>>());
            biases.push_back(std::vector<float>());
            for (int i = 0; i < hiddenLayerSize; ++i) {
                weights[0].push_back(std::vector<float>());
                biases[0].push_back(dis(gen));
                for (int j = 0; j < inputSize; ++j) {
                    weights[0][i].push_back(dis(gen));
                }
            }

            // Hidden layers
            for (int layer = 1; layer < numHiddenLayers; ++layer) {
                weights.push_back(std::vector<std::vector<float>>());
                biases.push_back(std::vector<float>());
                for (int i = 0; i < hiddenLayerSize; ++i) {
                    weights[layer].push_back(std::vector<float>());
                    biases[layer].push_back(dis(gen));
                    for (int j = 0; j < hiddenLayerSize; ++j) {
                        weights[layer][i].push_back(dis(gen));
                    }
                }
            }

            // Output layer
            weights.push_back(std::vector<std::vector<float>>());
            biases.push_back(std::vector<float>());
            for (int i = 0; i < outputSize; ++i) {
                weights[numHiddenLayers].push_back(std::vector<float>());
                biases[numHiddenLayers].push_back(dis(gen));
                for (int j = 0; j < hiddenLayerSize; ++j) {
                    weights[numHiddenLayers][i].push_back(dis(gen));
                }
            }

            biasOutput = dis(gen);
        }

};

int main() {

    int inputSize = 5; // n amount of last values to predict with
    int hiddenLayerSize = 3;
    int outputSize = 1; // Predicting a single value
    int numHiddenLayers = 1;
    float biasOutput = 0.0f;
    float learningRate = 0.2f;

    NeuralNetwork nn(inputSize, hiddenLayerSize, outputSize, numHiddenLayers, biasOutput, learningRate);

    // Generate synthetic time series data (e.g., sine wave with noise)
    std::vector<float> timeSeries;
    int dataLength = 100;
    std::random_device rd;
    std::mt19937 gen(2024);
    std::normal_distribution<> noise(0.0, 0.1); // Add small noise

    for (int i = 0; i < dataLength; ++i) {
        float value = std::sin(0.1f * i) + noise(gen); // Sine wave + noise
        timeSeries.push_back(value);
    }

    // Prepare training data using a sliding window
    std::vector<std::vector<float>> inputs;
    std::vector<float> targets;
    for (int i = 2; i < dataLength; ++i) {
        inputs.push_back({timeSeries[i - 2], timeSeries[i - 1]});
        targets.push_back(timeSeries[i]);
    }

    // Training loop
    int epochs = 500;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            nn.Train(inputs[i], targets[i]);
        }
    }

    // Testing the network
    std::vector<float> testInput = {timeSeries[dataLength - 2], timeSeries[dataLength - 1]};
    float prediction = nn.predict(testInput);
    std::cout << "Prediction for next value: " << prediction << std::endl;

    // Print actual next value for comparison
    float actual = std::sin(0.1f * dataLength) + noise(gen);
    std::cout << "Actual next value: " << actual << std::endl;

    nn.AddLayer();

    // Training loop for new layers
    epochs = 200;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            nn.Train(inputs[i], targets[i]);
        }
    }

    prediction = nn.predict(testInput);
    std::cout << "Prediction for next value: " << prediction << std::endl;

    // Print actual next value for comparison
    actual = std::sin(0.1f * dataLength) + noise(gen);
    std::cout << "Actual next value: " << actual << std::endl;

    return 0;
}
