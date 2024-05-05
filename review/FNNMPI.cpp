#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <mpi.h>

// Define your activation function (e.g., ReLU) here
template<typename T>
T relu(T x) {return x > 0 ? x : 0;}

template<typename T>
class FNNLayer {
private:
  size_t input_size;
  size_t output_size;
  std::vector<std::vector<T>> weights; // Weight matrix
  std::vector<T> biases; // Bias vector

public:
  FNNLayer(size_t input_size, size_t output_size);
  void forward(const std::vector<T>& input, std::vector<T>& output);
};

template<typename T>
FNNLayer<T>::FNNLayer(size_t input_size, size_t output_size) :
  input_size(input_size), output_size(output_size) {
  // Initialize weights and biases
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<T> dist(0.0, 1.0); // Normal distribution with mean 0 and standard deviation 1

  weights.resize(output_size, std::vector<T>(input_size));
  biases.resize(output_size);

  // Initialize weights and biases using MPI
  for (size_t i = 0; i < output_size; ++i) {
    for (size_t j = 0; j < input_size; ++j) {
      weights[i][j] = dist(gen); // Sample from the normal distribution
    }
    biases[i] = dist(gen);
  }

  // Broadcast weights and biases to all processes
  MPI_Bcast(weights.data(), weights.size() * input_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(biases.data(), biases.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

template<typename T>
void FNNLayer<T>::forward(const std::vector<T>& input, std::vector<T>& output) {
  output.resize(output_size);
  for (size_t i = 0; i < output_size; ++i) {
    T sum = biases[i];
    for (size_t j = 0; j < input_size; ++j) {
      sum += input[j] * weights[i][j];
    }
    // Apply activation function (e.g., ReLU)
    output[i] = relu(sum);
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  // Example usage
  size_t input_size = 4;
  size_t hidden_size = 5;
  size_t output_size = 3;

  std::vector<double> input = {1, 2, 3, 4};

  FNNLayer<double> hidden_layer(input_size, hidden_size);
  FNNLayer<double> output_layer(hidden_size, output_size);

  std::vector<double> hidden_output;
  hidden_layer.forward(input, hidden_output);

  std::vector<double> final_output;
  output_layer.forward(hidden_output, final_output);

  // Print final output on root process
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::cout << "Final output: ";
    for (size_t i = 0; i < final_output.size(); ++i) {
      std::cout << final_output[i] << " ";
    }
    std::cout << std::endl;
  }

  MPI_Finalize();
  return 0;
}
