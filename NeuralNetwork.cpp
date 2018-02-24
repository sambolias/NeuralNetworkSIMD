#include <vector>
using std::vector;
#include <memory>
using std::shared_ptr;
using std::make_shared;
#include <iostream>
using std::cout;
using std::endl;
#include <string>
using std::string;
#include <cstdlib>
#include <ctime>
using std::srand;
using std::time;
#include <cmath>
using std::pow;
#include "Board.hpp"
#include "NeuralNetwork.hpp"

//this doesnt work on windows
//source http://software-lisc.fbk.eu/avx_mathfun/
//TODO find license info
//#include "avx_mathfun.h"

//this one doesnt work on linux, need macro to include correct lib for os
//#include "avx_exp.h"


//format like (N0, N1, N2,..., Ni) where N is the number of neurons in the given layer i
NeuralNetwork::NeuralNetwork(const std::vector<int> & layers)
{
  //TODO make sure all but last layers are multiples of 8

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

//TODO make augFlag augment weights randomly - need some augmentation factor too that can be saved i think
NeuralNetwork::NeuralNetwork(std::string fname, bool augFlag)
{
  vector<float> raw = parseFile(fname);
  if(!raw.empty())
  {
    //set other variables
    vector<int> layers(raw[0]);
    int idx = 1;
    for(idx; idx < 1+raw[0]; idx++)
    {
      layers[idx-1] = raw[idx];
    }
    _layers = layers;
    kingValue = raw[idx];
    cout<<"kingValue="<<kingValue<<"\n";
    pieceCountWeight = raw[++idx];
    cout<<"pieceCountWeight="<<pieceCountWeight<<"\n";

    float * f = &raw[0];
    //set weights

    for (int layer = 0; layer < _layers.size() - 1; layer++)
    {
      _weights[layer] = vector<__m256*>((_layers[layer] * _layers[layer + 1])/8);
      for (int weightIndex = 0; weightIndex < _weights[layer].size(); ++weightIndex)
      {
        __m256 weight = _mm256_load_ps(&f[idx]);
        _weights[layer][weightIndex] = &weight;
        idx+=8; //grabbing 8 raw weights at a time
      }
    }
  }
  else
  {
    //TODO make some bad flag for network
  }

}
bool NeuralNetwork::saveNetwork(std::string fname)
{
  //TODO write to file #of layers / Layers / kingValue/ pieceCountWeight / weights
  return false;
}
vector<float> NeuralNetwork::parseFile(std::string fname)
{
  //TODO parse directly into vector - same format as saveNetwork
  vector<float> values;
  return values;
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
      _weights[layer][weightIndex] = &randoms;
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
	// [-1,1] : 2 / (1 + e^-sx) - 1
	return 2.0 / (1 + pow(e, (-s * x))) - 1;
}

// returns a float between -1 and 1
__m256 NeuralNetwork::sigmoidFunction(__m256 x)
{
	const float s = -4.;
  __m256 _s = _mm256_broadcast_ss(&s);
  const float one = 1.;
  __m256 _one = _mm256_broadcast_ss(&one);
  const float two = 2.;
  __m256 _two = _mm256_broadcast_ss(&two);

  //return 2.0 / (1 + pow(e, (-s * x))) - 1;
  __m256 _sig = _mm256_sub_ps(_mm256_div_ps(_two, (_mm256_add_ps(_one , exp256_ps(_mm256_mul_ps(_s,x))))) , _one);
  return _sig;
}

/*  alternate sigmoids not dependant on exponent - cross platform
float NeuralNetwork::sigmoidFunction(float x)
{
	return x / (1 + abs(x));
}
__m256 NeuralNetwork::sigmoidFunction(__m256 x)
{
	const float one = 1.;
	__m256 _one = _mm256_broadcast_ss(&one);
	//ugly square root for abs() = sqrt(x*x)...
	return _mm256_div_ps(x, (_mm256_add_ps(_mm256_sqrt_ps(_mm256_mul_ps(x, x)), _one)));
}
*/
float NeuralNetwork::GetBoardEvaluation(bool isRedPlayer, const vector<char> & board)
{
  float input;
  char boardSquare;
  vector<float> firstLayer(_layers[0]);
  _pieceCount = 0;

//TODO probably make assert
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
        currentNeurons = _mm256_add_ps(currentNeurons , (_mm256_mul_ps((*_weights[previousLayer][weightsIndex]) , (*_neurons[previousLayer][previousNeuronsIndex]))));
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
    outputNeuronInput += simdSumOfFloats(_mm256_mul_ps((*_weights[lastLayer][weightsIndex]) , (*_neurons[lastLayer][previousNeuronsIndex])));
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

//doesnt work in linux, needs define
//ported for windows from avx_mathfun.h
//source http://software-lisc.fbk.eu/avx_mathfun/
__m256 NeuralNetwork::exp256_ps(__m256 x) {
  __m256 tmp = _mm256_setzero_ps(), fx;
  __m256i imm0;
  float t = 1.;
  __m256 one = _mm256_broadcast_ss(&t);

  t = 88.3762626647949f;
  __m256 _ps256_exp_hi = _mm256_broadcast_ss(&t);
  x = _mm256_min_ps(x, _ps256_exp_hi);
  t = -88.3762626647949f;
  __m256 _ps256_exp_lo = _mm256_broadcast_ss(&t);
  x = _mm256_max_ps(x, _ps256_exp_lo);

  /* express exp(x) as exp(g + n*log(2)) */
  t = 1.44269504088896341;
  __m256 _ps256_cephes_LOG2EF = _mm256_broadcast_ss(&t);
  fx = _mm256_mul_ps(x, _ps256_cephes_LOG2EF);
  t = 0.5f;
  __m256 _ps256_0p5 = _mm256_broadcast_ss(&t);
  fx = _mm256_add_ps(fx, _ps256_0p5);

  /* how to perform a floorf with SSE: just below */
  //imm0 = _mm256_cvttps_epi32(fx);
  //tmp  = _mm256_cvtepi32_ps(imm0);

  tmp = _mm256_floor_ps(fx);

  /* if greater, substract 1 */
  //__m256 mask = _mm256_cmpgt_ps(tmp, fx);
  //int _CMP_GT_OS = 13;  //check this
  __m256 mask = _mm256_cmp_ps(tmp, fx, 13);
  mask = _mm256_and_ps(mask, one);
  fx = _mm256_sub_ps(tmp, mask);

  t = 0.693359375;
  __m256 _ps256_cephes_exp_C1 = _mm256_broadcast_ss(&t);
  tmp = _mm256_mul_ps(fx, _ps256_cephes_exp_C1);
  t = -2.12194440e-4;
  __m256 _ps256_cephes_exp_C2 = _mm256_broadcast_ss(&t);
  __m256 z = _mm256_mul_ps(fx, _ps256_cephes_exp_C2);
  x = _mm256_sub_ps(x, tmp);
  x = _mm256_sub_ps(x, z);

  z = _mm256_mul_ps(x,x);

  t = 1.9875691500E-4;
  __m256 _ps256_cephes_exp_p0 = _mm256_broadcast_ss(&t);
  t = 1.3981999507E-3;
  __m256 _ps256_cephes_exp_p1 = _mm256_broadcast_ss(&t);
  t = 8.3334519073E-3;
  __m256 _ps256_cephes_exp_p2 = _mm256_broadcast_ss(&t);
  t = 4.1665795894E-2;
  __m256 _ps256_cephes_exp_p3 = _mm256_broadcast_ss(&t);
  t = 1.6666665459E-1;
  __m256 _ps256_cephes_exp_p4 = _mm256_broadcast_ss(&t);
  t = 5.0000001201E-1;
  __m256 _ps256_cephes_exp_p5 = _mm256_broadcast_ss(&t);
  __m256 y = _ps256_cephes_exp_p0;
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, _ps256_cephes_exp_p1);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, _ps256_cephes_exp_p2);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, _ps256_cephes_exp_p3);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, _ps256_cephes_exp_p4);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, _ps256_cephes_exp_p5);
  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, x);
  y = _mm256_add_ps(y, one);

  /* build 2^n */
  imm0 = _mm256_cvttps_epi32(fx);
  // another two AVX2 instructions
  int tt = 0x7f;
  __m256i _pi32_256_0x7f = _mm256_set1_epi32(tt);
  imm0 = _mm256_add_epi32(imm0, _pi32_256_0x7f);
  imm0 = _mm256_slli_epi32(imm0, 23);
  __m256 pow2n = _mm256_castsi256_ps(imm0);
  y = _mm256_mul_ps(y, pow2n);
  return y;
}
