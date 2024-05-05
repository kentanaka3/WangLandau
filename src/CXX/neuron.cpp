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
  Brain myBrain(1e-1, argv[1]);
  myBrain.HelloWorld();
  std::vector<double> input(20, 0.5);
  for (auto& o : myBrain.forward(input)) {std::cout << o << std::endl;}
  // Example training data (XOR problem)
  std::vector<std::vector<double>> train_inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  std::vector<std::vector<double>> train_outputs = {{0}, {1}, {1}, {0}};

  // Train the network
  int epochs = 10000;
  myBrain.train(train_inputs, train_outputs, epochs);

  // Test the trained network
  for (const auto& input : train_inputs) {
    std::vector<double> output = myBrain.forward(input);
    std::cout << "Input: " << input[0] << ", " << input[1]
              << " Output: " << output[0] << std::endl;
  }
  return 0;
}