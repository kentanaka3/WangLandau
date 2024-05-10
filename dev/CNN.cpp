#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <filesystem>
#include <map>
#include <complex>
#include <random>
#include <omp.h>

// Define constants and functions here

template<typename T>
class CNNLayer {
private:
  // Number of filters
  size_t num_filters;
  // Size of each filter (assumed square)
  size_t filter_size;
  // Stride for convolution
  size_t stride;
  // Activation function
  T (*activation)(T);
  // Weights for filters
  std::vector<std::vector<std::vector<double>>> filters;
  // Bias for each filter
  std::vector<double> biases;
  // Feature maps produced by convolution
  std::vector<std::vector<std::vector<T>>> feature_maps;
  // Size of feature maps
  size_t map_size;

public:
  CNNLayer(size_t num_filters, size_t filter_size, size_t stride,
           T (*activation)(T));
  void forward(const std::vector<std::vector<std::vector<T>>>& input);
  void printFeatureMaps();
};

template<typename T>
CNNLayer<T>::CNNLayer(size_t num_filters, size_t filter_size, size_t stride,
                      T (*activation)(T)) : num_filters(num_filters),
  filter_size(filter_size), stride(stride), activation(activation) {
  // Initialize filters and biases
  filters.resize(num_filters, std::vector<std::vector<double>>(filter_size,
                                std::vector<double>(filter_size)));
  biases.resize(num_filters);

  // Initialize feature maps
  map_size = (input_size - filter_size) / stride + 1;
  feature_maps.resize(num_filters, std::vector<std::vector<T>>(map_size, std::vector<T>(map_size)));
}

template<typename T>
void CNNLayer<T>::forward(const std::vector<std::vector<std::vector<T>>>& input) {
  // Convolution operation
  for (size_t f = 0; f < num_filters; ++f) {
    for (size_t i = 0; i < map_size; ++i) {
      for (size_t j = 0; j < map_size; ++j) {
        T sum = 0;
        for (size_t x = 0; x < filter_size; ++x) {
          for (size_t y = 0; y < filter_size; ++y) {
            sum += input[i * stride + x][j * stride + y] * filters[f][x][y];
          }
        }
        sum += biases[f];
        feature_maps[f][i][j] = activation(sum);
      }
    }
  }
}

template<typename T>
void CNNLayer<T>::printFeatureMaps() {
  // Print feature maps for debugging
  for (size_t f = 0; f < num_filters; ++f) {
    std::cout << "Feature Map " << f << ":" << std::endl;
    for (size_t i = 0; i < map_size; ++i) {
      for (size_t j = 0; j < map_size; ++j) {
        std::cout << feature_maps[f][i][j] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}

int main() {
  // Example usage
  std::vector<std::vector<std::vector<double>>> input = {
    {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}},
    {{17, 18, 19, 20}, {21, 22, 23, 24}, {25, 26, 27, 28}, {29, 30, 31, 32}},
    {{33, 34, 35, 36}, {37, 38, 39, 40}, {41, 42, 43, 44}, {45, 46, 47, 48}},
    {{49, 50, 51, 52}, {53, 54, 55, 56}, {57, 58, 59, 60}, {61, 62, 63, 64}}
  };
  CNNLayer<double> layer(2, 3, 1, [](double x) { return x > 0 ? x : 0; });
  layer.forward(input);
  layer.printFeatureMaps();
  return 0;
}
