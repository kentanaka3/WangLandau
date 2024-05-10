#include <iostream>
#include <vector>
#include <cmath>

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
FNNLayer<T>::FNNLayer(size_t input_size, size_t output_size)
    : input_size(input_size), output_size(output_size) {
    // Initialize weights and biases
    // Initialization of weights and biases omitted for brevity
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

int main() {
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

    // Print final output
    std::cout << "Final output: ";
    for (size_t i = 0; i < final_output.size(); ++i) {
        std::cout << final_output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
