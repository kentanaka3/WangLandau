#include "neuron.hpp"

/*
 * Brain
 *  - Layer0        - MPI0
 *    - neuron
 *    - neuron
 *      ...
 *    - neuron
 *  - Layer1        - MPI for MPI_RANK in range(1, MPI_SIZE):
 *    - neuron
 *    - neuron
 *      ...
 *    - neuron
 *    ...
 *  - Layer(N + 1)  - MPI0
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
  std::cout << " - " << std::atoi(argv[1]) << " - " << std::endl;
  myBrain.HelloWorld();
  return 0;
}