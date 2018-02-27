#include "NeuralNetwork.hpp"
#include <vector>
using std::vector;
#include <iostream>
using std::cout;
#include <chrono>


void timeNN(NeuralNetwork & net, vector<char> & board)
{
  int avgNum = 1000000;
  std::chrono::high_resolution_clock clock;
  std::chrono::nanoseconds ellapsed (0);
  auto start = clock.now();
  #pragma omp parallel for
  for(int i = 0; i < avgNum; i++)
  {
      net.GetBoardEvaluation(false, board);
  }
  auto diff = clock.now() - start;
  auto ns = std::chrono::duration_cast<std::chrono::nanoseconds> (diff);
  ellapsed += ns;
  cout<<"average time over "<<avgNum<<" evaluations\n";
  cout<<"with "<<net.getNeuronCount()<<" neurons over "<<net.getWeightCount()<<" weights: \n";
  cout<<ellapsed.count()/avgNum<<" ns per evaluation\n";
  cout<<(double)(1.0e9/(ellapsed.count()/avgNum))<<" board evaluations per second\n";
}

int main()
{

  vector<char> board =  {
                        'r','r','r','r',
                        'r','r','r','r',
                        'r','r','r','r',
                        ' ',' ',' ',' ',
                        ' ',' ',' ',' ',
                        ' ',' ',' ',' ',
                        ' ',' ',' ',' ',
                        'b','b','b','b',
                    };


  NeuralNetwork test({32, 40, 16, 1});
  NeuralNetwork test2({32, 112, 80, 64, 16, 1});
  test2.saveNetwork("test.save.txt");
  NeuralNetwork test3("test.save.txt", false);

  cout<<"Test board has 3x more red pieces than black, random weights [-.2,.2]\n";
  cout<<"4 Layer Network output\n";
  cout<< "Black evaluation: "<<test.GetBoardEvaluation(false, board)<<"\n";
  cout<< "Red evaluation: "<<test.GetBoardEvaluation(true, board)<<"\n";
  cout<<"6 Layer Network output\n";
  cout<< "Black evaluation: "<<test2.GetBoardEvaluation(false, board)<<"\n";
  cout<< "Red evaluation: "<<test2.GetBoardEvaluation(true, board)<<"\n";
  cout<<"Output from saved/loaded copy of above network\n";
  cout<< "Black evaluation: "<<test3.GetBoardEvaluation(false, board)<<"\n";
  cout<< "Red evaluation: "<<test3.GetBoardEvaluation(true, board)<<"\n";

  //timing
  timeNN(test, board);
  timeNN(test2, board);

  return 0;
}
