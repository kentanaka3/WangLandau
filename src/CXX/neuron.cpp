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
 * /____________________
 * \
 *  \
 *   \
 *    \
 *     \
 *      \
 *
 */
int main(int argc, char *argv[]) {
  #ifdef _MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &MPI_GRANK);
  MPI_Comm_size(MPI_COMM_WORLD, &MPI_GSIZE);
  #endif
  Brain myBrain(argv[1]);
  myBrain.HelloWorld();
  // Example training data (XOR problem)
  std::vector<std::vector<double>> train_inputs = {{0, 0}, {0, 1},
                                                   {1, 0}, {1, 1}};
  std::vector<std::vector<double>> train_outputs = {{0}, {1},
                                                    {1}, {0}};

  // Train the network
  myBrain.train(train_inputs, train_outputs);

  // Test the trained network
  std::vector<double> output = myBrain.forward(train_inputs[0]);
  std::cout << "Input: " << train_inputs[0][0] << ", " << train_inputs[0][1] \
            << " Output: " << output[0] << std::endl;
  #ifdef _MPI
  MPI_Finalize();
  #endif
  return 0;
}