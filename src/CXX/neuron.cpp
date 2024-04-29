#include "neuron.hpp"

template<typename T>
void backPropagation(Brain<T>& brain, int N, int test) {
  for (int i = 0; i < N; i++) {
    if ((i % test) == 0) {
      std::cout << " - " << i << " - " << std::endl;
      brain.HelloWorld();
      brain.test();
    }
    brain.backPropagate();
  }
}

template <typename T>
void encoder(Brain<T>& brain, int N, int test) {
  for (int i = 0; i < N; i++) {
    if ((i % test) == 0) {
      std::cout << " - " << i << " - " << std::endl;
      brain.HelloWorld();
      brain.test();
    }
    brain.encode();
    brain.decode();
  }
}

/*
 * Brain
 *  - Layer
 *    - neuron
 *    - neuron
 *      ...
 *    - neuron
 *  - Layer
 *    - neuron
 *    - neuron
 *      ...
 *    - neuron
 *    ...
 *  - Layer
 *    - neuron
 *    - neuron
 *      ...
 *    - neuron
 *
 * Time  /
 *      /
 *     /
 *    /
 *   /
 *  /
 * /
 * \
 *  \
 *   \
 *    \
 *     \
 *      \
 *
 */
int main(int argc, char *argv[]) {
  Brain<double> myBrain(1e-1, "data/Brain.init", INIT_STATE_SIGMA);
  backPropagation(myBrain, std::atoi(argv[1]), 10);
  std::cout << " - " << std::atoi(argv[1]) << " - " << std::endl;
  myBrain.HelloWorld();
  return 0;
}