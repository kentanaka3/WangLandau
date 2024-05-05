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

#include "utils.hpp"

std::complex<double> INIT_STATE_CMPLX = 1.;
bool INIT_STATE_ISING = true;
size_t INIT_STATE_POTTS = 0;
double INIT_STATE_SIGMA = 0.999999;
int MAX_LAYERS = 100, MAX_NEURONS = 1000;
std::ostringstream DATA_PATH{"data/"};
unsigned int MARKOVIANITY = 0;

template<typename T>
T sgmd(const T& x) {return 1./(1. + std::exp(-x));}
template<typename T>
T D_sgmd(const T& x) {
  double activated = sgmd(x);
  return activated * (1 - activated);
}
template<typename T>
T relu(const T& x) {return (x < T(0) ? T(0) : T(1));}

std::random_device rndm;
std::mt19937_64 gen(rndm());
std::uniform_int_distribution<> unf_dist(0., 1.);

template<typename T>
class neuron {
private:
  // Bias
  T Bias;
  // Value
  T Value;
  // Past Values
  std::vector<T> Past;
  // Error
  double Error;
public:
  // Weights
  std::vector<double> Weights;
  bool train;
  neuron(const std::string& file);
  ~neuron() {};
  void HelloWorld();

};

// Definitions for neuron class methods
template<typename T>
neuron<T>::neuron(const std::string& file) {

}

template<typename T>
void neuron<T>::HelloWorld() {
  std::cout << this->Value << ", ";
}



template<typename T>
class Layer {
private:
  // Neurons
  std::vector<neuron<T>> Neurons;
  // Learning Rate
  double LrnRate;
public:
  size_t N_nodes;
  size_t N_connections;
  std::string filename;
  Layer(const size_t& n_nodes, const size_t& n_connections, const T& state,
        const std::string& file);
  ~Layer() {};
  void HelloWorld();
  std::vector<std::vector<double>> Weights();
  std::vector<double> Weights(size_t i);
  double Weights(size_t i, size_t j);
};

// Definitions for Layer class methods
template<typename T>
Layer<T>::Layer(const size_t& n_nodes, const size_t& n_connections,
                const T& state, const std::string& file) : N_nodes(n_nodes),
  N_connections(n_connections), filename(file), LrnRate(0.9) {
  std::filesystem::create_directories(file);
  // Create Layer file
  std::ostringstream filepath("");
  filepath << file << "/L.dat";
  std::ofstream fp;
  fp.open(filepath.str());
  if (fp.is_open()) {
    fp << "# Type           double" << std::endl \
       << "# N_connections  " << N_connections << std::endl;
    fp.close();
  } else std::cerr << "Failed to open file : " << filepath.str() << std::endl;
  Neurons.resize(N_nodes, neuron<T>(filepath.str()));
  // Create Weights file
  filepath.str("");
  filepath << file << "/W.dat";
  // Print Matrix
  printMtx(filepath.str(), N_nodes, N_connections, this->Weights());
}

template<typename T>
void Layer<T>::HelloWorld() {for (neuron<T>& n : Neurons) n.HelloWorld();}

template<typename T>
std::vector<std::vector<double>> Layer<T>::Weights() {
  std::vector<std::vector<double>> W(N_nodes);
  for (size_t n = 0; n < N_nodes; n++) {
    int i = 0;
    for (const auto& w: Neurons[n].Weights) {
      W[n][i] = w;
      i++;
    }
  }
  return W;
}

template<typename T>
class Brain {
public:
  size_t l;
  double flopRate;
  std::vector<Layer<T>> Layers;
  std::string filename;
  Brain(const double& fR, const std::string& file, const T& state);
  ~Brain() {};
  void HelloWorld();
  void forward(std::vector<T> input); // Changed return type to void
};

// Definitions for Brain class methods
template<typename T>
Brain<T>::Brain(const double& fR, const std::string& file, const T& state) : \
  flopRate(fR), \
  filename(std::filesystem::path(file).replace_filename("Brain/")) {
  std::filesystem::create_directories(filename);
  std::ifstream fr(file);
  if (!fr.is_open()) {
    std::cerr << "Error: Unable to init file " << file << "\n";
  }
  std::map<std::string, std::string> params;
  params["L"] = "4";
  params["Layers"] = "6";
  std::string line;
  // Read Parameters by parsing with RegEx
  while (std::getline(fr, line)) params = paramMap(line, params);
  fr.close();
  l = std::atoi(params["L"].c_str()) - 1;
  std::vector<int> L(l, 0);
  param2vec(params["Layers"], L, l);
  if (l + 1 <= MAX_LAYERS) {
    // Input Layer
    size_t layer = 0;
    std::ostringstream filepath;
    filepath.str("");
    filepath << filename << std::setfill('0') \
             << std::setw(static_cast<int>(log10(MAX_LAYERS))) << layer;
    Layers.push_back(Layer<T>(L[layer], 1, state, filepath.str()));
    // Intermediate layers
    for (++layer; layer <= l - 1; layer++) {
      filepath.str("");
      filepath << filename << std::setfill('0') \
               << std::setw(static_cast<int>(log10(MAX_LAYERS))) << layer;
      Layers.push_back(Layer<T>(L[layer], L[layer - 1], state, filepath.str()));
    }
    // Output layer
    filepath.str("");
    filepath << filename << std::setfill('0') \
             << std::setw(static_cast<int>(log10(MAX_LAYERS))) << layer;
    Layers.push_back(Layer<T>(1, L[layer - 1], state, filepath.str()));
  }
}

template<typename T>
void Brain<T>::HelloWorld() {
  size_t i = 0;
  for (Layer<T>& L : Layers) {
    std::cout << "L" << std::setfill('0') \
              << std::setw(static_cast<int>(log10(MAX_LAYERS))) << i << ": ";
    L.HelloWorld();
    std::cout << std::endl;
    i++;
  }
}

template<typename T>
void Brain<T>::forward(std::vector<T> input) {
  /*
   * I ----- O
   * |       |
   * |   *   |
   * | * *   |
   * | * * * |
   * * * * * |
   * * * * * *
   * 0 1 2 3 4 -> L
   * X | | | |
   * s(X)| | |
   * s(s(X)) |
   * s(s(s(X)))
   * s(s(s(s(X))))
   */
  std::vector<T> current_input = input;
  // Propagate input through each layer
  for (size_t i = 0; i < Layers.size(); ++i) {
    std::vector<T> layer_output(Layers[i].N_nodes);
    // Propagate input through each neuron in the layer
    for (size_t j = 0; j < Layers[i].N_nodes; ++j) {
      T neuron_output = 0;
      // Calculate weighted sum of inputs
      for (size_t k = 0; k < current_input.size(); ++k) {
        neuron_output += current_input[k] * Layers[i].Weights[j][k];
      }
      // Apply activation function
      neuron_output = sgmd(neuron_output + Layers[i].Neurons[j].Bias);
      Layers[i].Neurons[j].Value = neuron_output;
      layer_output[j] = neuron_output;
    }
    // Set current input to the output of this layer for the next layer
    current_input = layer_output;
  }
}
