#ifndef NeuralNetwork_H
#define NeuralNetwork_H

#include <vector>
#include <immintrin.h>
#include <string>


class NeuralNetwork
{
  // Each layer holds an array of neuron outputs. (eg. _layers[0] = 10 sets the neuron output in the first layer to 10)
  std::vector<int> _layers;
  // _neurons[layer].size() == the number of neruons in that layer
  std::vector<std::vector<__m256*>> _neurons;
  // _weights[layer].size() == the number of connections between the neurons in layer and layer + 1
  std::vector<std::vector<__m256*>> _weights;

  float pieceCountWeight;
  float _pieceCount;
  float kingValue;

//evaluation functions
  void resetNeurons();
  void randomizeWeights();
  __m256 sigmoidFunction(__m256 x);
  float sigmoidFunction(float x);
  __m256 getRandomWeight();
  float getLayerEvaluation();
  float simdSumOfFloats(__m256 floats);

//save/load functions
  std::vector<float> parseFile(std::string fname);

public:
  // for each integer, creates a layer with format[index] neurons
  NeuralNetwork(const std::vector<int> & layers);
  NeuralNetwork(std::string fname, bool augFlag);
  bool saveNetwork(std::string fname);
  int getNeuronCount();
  int getWeightCount();
  float GetBoardEvaluation(bool isRedPlayer, const std::vector<char> & board);
};

#endif
