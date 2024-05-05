#include <iostream>
#include <vector>
#include <cmath>

template<typename T>
class RNNLayer {
private:
  size_t input_size;
  size_t hidden_size;
  size_t output_size;
  std::vector<std::vector<T>> weights_xh; // Weight matrix for input to hidden
  std::vector<std::vector<T>> weights_hh; // Weight matrix for hidden to hidden
  std::vector<std::vector<T>> weights_hy; // Weight matrix for hidden to output
  std::vector<T> biases_h; // Bias vector for hidden layer
  std::vector<T> biases_y; // Bias vector for output layer

public:
  RNNLayer(size_t input_size, size_t hidden_size, size_t output_size);
  void forward(const std::vector<std::vector<T>>& input, std::vector<std::vector<T>>& hidden_state, std::vector<std::vector<T>>& output);
};

template<typename T>
RNNLayer<T>::RNNLayer(size_t input_size, size_t hidden_size,
                      size_t output_size) : input_size(input_size),
  hidden_size(hidden_size), output_size(output_size) {
  // Initialize weights and biases
  weights_xh.resize(hidden_size, std::vector<T>(input_size));
  weights_hh.resize(hidden_size, std::vector<T>(hidden_size));
  weights_hy.resize(output_size, std::vector<T>(hidden_size));
  biases_h.resize(hidden_size);
  biases_y.resize(output_size);
  // Initialize weights and biases with random values
  // (You may want to initialize them differently based on your application)
  // Initialization of weights and biases omitted for brevity
}

template<typename T>
void RNNLayer<T>::forward(const std::vector<std::vector<T>>& input, std::vector<std::vector<T>>& hidden_state, std::vector<std::vector<T>>& output) {
  // Initialize hidden state with zeros
  hidden_state.clear();
  hidden_state.resize(input.size() + 1, std::vector<T>(hidden_size, 0));

  // Forward pass through time steps
  for (size_t t = 0; t < input.size(); ++t) {
    // Update hidden state using input and previous hidden state
    for (size_t i = 0; i < hidden_size; ++i) {
      hidden_state[t + 1][i] = biases_h[i];
      for (size_t j = 0; j < input_size; ++j) {
        hidden_state[t + 1][i] += weights_xh[i][j] * input[t][j];
      }
      for (size_t j = 0; j < hidden_size; ++j) {
        hidden_state[t + 1][i] += weights_hh[i][j] * hidden_state[t][j];
      }
      hidden_state[t + 1][i] = std::tanh(hidden_state[t + 1][i]);
    }

    // Compute output based on hidden state
    output[t].resize(output_size);
    for (size_t i = 0; i < output_size; ++i) {
      output[t][i] = biases_y[i];
      for (size_t j = 0; j < hidden_size; ++j) {
        output[t][i] += weights_hy[i][j] * hidden_state[t + 1][j];
      }
      // Apply activation function (if any) to the output
      // For simplicity, let's assume it's a linear activation
    }
  }
}

int main() {
  // Example usage
  size_t input_size = 3;
  size_t hidden_size = 2;
  size_t output_size = 2;

  std::vector<std::vector<double>> input = {
    {0.1, 0.2, 0.3},
    {0.4, 0.5, 0.6},
    {0.7, 0.8, 0.9}
  };

  RNNLayer<double> rnn_layer(input_size, hidden_size, output_size);

  std::vector<std::vector<double>> hidden_state;
  std::vector<std::vector<double>> output(input.size());
  rnn_layer.forward(input, hidden_state, output);

    // Print output
  for (size_t t = 0; t < output.size(); ++t) {
    std::cout << "Time step " << t << " output: ";
    for (size_t i = 0; i < output[t].size(); ++i) {
      std::cout << output[t][i] << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}
