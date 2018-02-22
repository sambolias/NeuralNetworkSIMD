#include <vector>
using std::vector;
#include <memory>
using std::shared_ptr;
using std::make_shared;
#include <iostream>
using std::cout;
using std::endl;
#include <cstdlib>
#include <ctime>
using std::srand;
using std::time;
#include <cmath>
using std::pow;
#include "Board.hpp"
#include "NeuralNetwork.hpp"
#include "avx_mathfun.h"

//format like (N0, N1, N2,..., Ni) where N is the number of neurons in the given layer i
NeuralNetwork::NeuralNetwork(const std::vector<int> & layers)
{

  if (layers.empty() || (layers[0] != 32))
  {
    cout << "Error neural network is either empty, or the input layer does not have 32 neurons.\n";
    return;
  }
  //seed rand
  srand(time(0));
  //random kingValue [1,3]
  kingValue = (rand() % 101) * 2.0 / 100.0 + 1.0;

  _layers = layers;
  resetNeurons();
  randomizeWeights();

}

int NeuralNetwork::getNeuronCount()
{
  int total = 0;
  for (const auto & neuronLayer : _neurons)
  {
    total += neuronLayer.size()*8;
  }
  //add one for output neuron
  return ++total;
}

int NeuralNetwork::getWeightCount()
{
  int total = 0;
  for (const auto & weightLayer : _weights)
  {
    total += weightLayer.size()*8;
  }
  //add one for pieceCountWeight
  return ++total;
}

void NeuralNetwork::randomizeWeights()
{
  _weights = vector<vector<__m256*>>(_layers.size());

  for (int layer = 0; layer < _layers.size() - 1; layer++)
  {
    _weights[layer] = vector<__m256*>((_layers[layer] * _layers[layer + 1])/8);
    for (int weightIndex = 0; weightIndex < _weights[layer].size(); ++weightIndex)
    {
      __m256 randoms = getRandomWeight();
      _weights[layer][weightIndex] = &randoms;//getRandomWeight();
    }
  }
  //set pieceCountWeight
  pieceCountWeight = rand() % 101 / 100. * 0.4 - 0.2;
}

// [-0.2, 0.2]
__m256 NeuralNetwork::getRandomWeight()
{
	float randomWeight[8];

  for(int i = 0; i < 8; i++)
  {
    randomWeight[i] = rand() % 101; // [0, 100]
  	randomWeight[i] /= 100.0; // [0, 1];
  	randomWeight[i] *= 0.4; // [0, 0.4]
  	randomWeight[i] -= 0.2; // [-0.2, 0.2]
  }
  auto rw = _mm256_load_ps(&randomWeight[0]);
	return rw;
}

void NeuralNetwork::resetNeurons()
{
  _neurons = vector<vector<__m256*>>(_layers.size());

  for (int layer = 0; layer < _layers.size()-1; ++layer)
  {
    float z = 0.;
    __m256 zeros = _mm256_broadcast_ss(&z);
    _neurons[layer] = vector<__m256*>(_layers[layer]/8, &zeros);
  }
}

// returns a float between -1 and 1
float NeuralNetwork::sigmoidFunction(float x)
{
	const float s = 4;
	const float e = 2.718281828;
	// Can graph this function to make sure on [-1,1] : 2 / (1 + e^-sx) - 1
	return 2.0 / (1 + pow(e, (-s * x))) - 1;
}

// returns a float between -1 and 1
__m256 NeuralNetwork::sigmoidFunction(__m256 x)
{
	const float s = -4.;
  __m256 _s = _mm256_broadcast_ss(&s);
	// const float e = 2.718281828;
  // __m256 _e = _mm256_broadcast_ss(&e);
  const float one = 1.;
  __m256 _one = _mm256_broadcast_ss(&one);
  const float two = 2.;
  __m256 _two = _mm256_broadcast_ss(&two);
  //return 2.0 / (1 + pow(e, (-s * x))) - 1;
  __m256 _sig = (_two / (_one + exp256_ps(_s*x))) - _one;
  return _sig;

}

float NeuralNetwork::GetBoardEvaluation(bool isRedPlayer, const vector<char> & board)
{
  //resetNeurons();
  float input;
  char boardSquare;
  vector<float> firstLayer(_layers[0]);
  _pieceCount = 0;

//probably make assert
  if(firstLayer.size() != board.size())
    cout<<"bad board in GetBoardEvaluation()\n";

  for (int i = 0; i < board.size(); ++i)
  {
    boardSquare = board[i];
    input = 0;
    if (boardSquare == Board::RED)
    {
      input = 1.0;
    }
    else if (boardSquare == Board::K_RED)
    {
      input = kingValue;
    }
    else if (boardSquare == Board::BLACK)
    {
      input = -1.0;
    }
    else if (boardSquare == Board::K_BLACK)
    {
      input = -kingValue;
    }
    if (!isRedPlayer)
    {
      input *= -1;
    }
    _pieceCount += input;
    firstLayer[i] = input;
  }

  //converts vector to array
  float * fl = &firstLayer[0];
  //load firstLayer into neurons
  for(int i = 0; i < firstLayer.size(); i+=8)
  {
    __m256 temp = _mm256_load_ps(&fl[i]);
    _neurons[0][i/8] = &temp;
  }

  return getLayerEvaluation();
}

// This evaluation requires weights to be stored in a specific way
// N = number of neurons in current layer
// The first N weights must connect from the N neurons in the current layer to the first neuron in the next layer
// Then the next N weights connect from the N neurons in the current layer to the second neuron in the next layer
// Then so on...
/*
Ex: layer 1 has ABC, layer 2 has DEF, and layer 3 has G
    The order of the weights would be as shown below
      A --1-- D
      B --2-- D
      C --3-- D

      A --4-- E
      B --5-- E
      C --6-- E

      A --7-- F
      B --8-- F
      C --9-- F

      D --10-- G
      E --11-- G
      F --12-- G
*/
float NeuralNetwork::getLayerEvaluation()
{
  float z = 0.;
  __m256 zeros = _mm256_broadcast_ss(&z);
  for(int layer = 1; layer < _layers.size()-1; layer++)
  {
    int previousLayer = layer - 1;
    int previousLayerSize = _layers[previousLayer]/8;
    int layerSize = _layers[layer]/8;
    int weightsIndex;
    __m256 currentNeurons;
    for (int neuronsIndex = 0; neuronsIndex < layerSize; ++neuronsIndex)
    {
        currentNeurons = zeros;
        //openmp slows it way down, too small of problem to justify threads I think
  //    #pragma omp parallel for reduction(+:currentNeurons)
      for (int previousNeuronsIndex = 0; previousNeuronsIndex < previousLayerSize; ++previousNeuronsIndex)
      {
        weightsIndex = previousLayerSize * neuronsIndex + previousNeuronsIndex;
        currentNeurons = currentNeurons + ((*_weights[previousLayer][weightsIndex]) * (*_neurons[previousLayer][previousNeuronsIndex]));
      }
      currentNeurons = sigmoidFunction(currentNeurons);
      _neurons[layer][neuronsIndex] = &currentNeurons;
    }
  }

  //need to handle last neuron solo
  float outputNeuronInput = 0;
  int lastLayer = _layers.size()-2;
  int lastLayerSize = _layers[lastLayer]/8;
//   #pragma omp parallel for reduction(+:outputNeuronInput) num_threads(4)
  for (int previousNeuronsIndex = 0; previousNeuronsIndex < lastLayerSize; ++previousNeuronsIndex)
  {
    int weightsIndex = previousNeuronsIndex;
    outputNeuronInput += simdSumOfFloats((*_weights[lastLayer][weightsIndex]) * (*_neurons[lastLayer][previousNeuronsIndex]));
  }
  //add in _pieceCount
  outputNeuronInput += pieceCountWeight*_pieceCount;

  //return final output
  return sigmoidFunction(outputNeuronInput);
}

float NeuralNetwork::simdSumOfFloats(__m256 floats)
{
  float f[8];
  _mm256_store_ps(&f[0], floats);
  float sum = 0.;
  for(int i = 0; i < 8; i++)
    sum += f[i];

  return sum;
}
