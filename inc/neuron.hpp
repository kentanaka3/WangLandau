#include <iostream>
#include <map>
#include <omp.h>
#include <math.h>
#include <random>
#include <iomanip>
#include <istream>
#include <complex>
#include <filesystem>

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
  // Weights
  std::vector<double> Weights;
  // Log file of the neuron
  std::string filename;
public:
  bool train;
  neuron(const T& value, const std::vector<T> weights, T bias, double error,
         const std::string& file) : Value(value), train(true), filename(file),
    Bias(bias), Past(std::vector<T>(MARKOVIANITY, Value)), Weights(weights),
    Error(error) {};
  ~neuron() {};
  T readLastLine(const std::string& file);
  void HelloWorld();
  void capture();

  T frontPropagate(const std::vector<T>& a);
  void backPropagate();
  void test();
  std::vector<double> updateWeights(double LrnRt, const std::vector<T>& a);
};

template<typename T>
T neuron<T>::readLastLine(const std::string& file) {
  std::map<std::string, std::string> params;
  if constexpr (std::is_same_v<T, int>) {
    params["Value"] = "4";
  } else if constexpr (std::is_same_v<T, double>) {
    params["Value"] = "2";
  } else if constexpr (std::is_same_v<T, bool>) {
    params["Value"] = "5";
  } else if constexpr (std::is_same_v<T, std::complex>) {
    params["Value"] = "6";
  }
  params = paramMap(lastLine(file), params);
  if constexpr (std::is_same_v<T, int>) {
    return std::atoi(params["Value"].c_str());
  } else if constexpr (std::is_same_v<T, double>) {
    return std::atof(params["Value"].c_str());
  } else if constexpr (std::is_same_v<T, bool>) {
    return params["Value"] == "1" ? true : false;
  } else if constexpr (std::is_same_v<T, std::complex>) {
    return 1.;
  }
}

template<typename T>
void neuron<T>::backPropagate() {
  // Calculate delta error
  double delta = Value * (1.0 - Value) * (Value - Past[MARKOVIANITY - 1]);
  // Update weights
  for (size_t i = 0; i < Weights.size(); ++i)
    Weights[i] += unf_dist(gen) * delta;
  // Update bias
  Bias += delta;
  // Update the Past vector for Markovianity
  Past.erase(Past.begin());
  Past.push_back(Value);
}

template<typename T>
void neuron<T>::capture() {
  std::ofstream fp(filename, std::ios_base::app);
  if (fp) {
    fp << "# Value\t" << Value << std::endl;
    fp.close();
  } else std::cerr << "Failed to open file : " << filename << std::endl;
}

template<typename T>
void neuron<T>::HelloWorld() {
  std::cout << this->Value << ", ";
  capture();
}

template<typename T>
void neuron<T>::test() {

}

template<typename T>
T neuron<T>::frontPropagate(const std::vector<T>& a) {
  T sum = 0;
  #pragma omp parallel for reduce(+:sum)
  for (size_t i = 0; i < Weights.size(); ++i) sum += Weights[i] * a[i];
  Value = sgmd(sum + Bias);
  // Update the Past vector for Markovianity
  if (Past.size()) {
    Past.erase(Past.begin());
    Past.push_back(Value);
  }
  return Value;
}

template<typename T>
std::vector<double> neuron<T>::updateWeights(double LrnRt,
                                             const std::vector<T>& a) {
  #pragma omp parallel for
  for (size_t i = 0; i < Weights.size(); ++i)
    Weights[i] += LrnRt * Error * a[i];
  Bias += LrnRt * Error;
  return Weights;
}

template<typename T>
class Layer {
private:
  // Neurons
  std::vector<neuron<T>> Neurons;
  // Weights
  std::vector<std::vector<double>> Weights;
public:
  size_t N_nodes;
  std::string filename;
  Layer(const size_t& n_nodes, const size_t& n_u, const T& state,
        const std::string& file);
  ~Layer() {};
  void HelloWorld();
  void frontPropagate();
  void backPropagate();
  void encode();
  void decode();
  void test();
  void updateWeights();
};

template<typename T>
Layer<T>::Layer(const size_t& n_nodes, const size_t& n_u, const T& state,
                const std::string& file) : N_nodes(n_nodes), filename(file) {
  std::filesystem::create_directories(file);
  std::ostringstream filepath("");
  filepath << file << "/W.dat";
  std::ofstream fp;
  fp.open(filepath.str());
  if (fp.is_open()) fp.close();
  else std::cerr << "Failed to open file : " << filepath.str() << std::endl;
  filepath.str("");
  filepath << file << "/" << std::setfill('0') \
           << std::setw(static_cast<int>(log10(MAX_NEURONS))) << 0 << ".dat";
  Neurons.resize(N_nodes, neuron<T>(n_u, state, filepath.str()));
  for (size_t n = 0; n < n_nodes; n++) {
    filepath.str("");
    filepath << file << "/" << std::setfill('0') \
             << std::setw(static_cast<int>(log10(MAX_NEURONS))) << n << ".dat";
    Neurons[n].filename = filepath.str();
    std::ofstream fp;
    fp.open(filepath.str());
    if (fp.is_open()) fp.close();
    else std::cerr << "Failed to open file : " << filepath.str() << std::endl;
  }
}

template<typename T>
void Layer<T>::HelloWorld() {for (neuron<T>& n : Neurons) n.HelloWorld();}

// Forward propagate through neurons in this layer
template<typename T>
void Layer<T>::frontPropagate() {
  for (neuron<T>& n : Neurons) n.frontPropagate(filename);
}

// Backward propagate through neurons in this layer
template<typename T>
void Layer<T>::backPropagate() {
  for (neuron<T>& n : Neurons) n.backPropagate();
}

// Encode the states of neurons in this layer
template<typename T>
void Layer<T>::encode() {for (neuron<T>& n : Neurons) n.encode();}

// Decode the states of neurons in this layer
template<typename T>
void Layer<T>::decode() {for (neuron<T>& n : Neurons) n.decode();}

// Decode the states of neurons in this layer
template<typename T>
void Layer<T>::test() {for (neuron<T>& n : Neurons) n.test();}

// Decode the states of neurons in this layer
template<typename T>
void Layer<T>::test() {for (neuron<T>& n : Neurons) n.test();}

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
  void backPropagate();
  void encode();
  void decode();
  void test();
  void backwardPass(const std::vector<double>& expectedOutputs);
  
};

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
void Brain<T>::backPropagate() {
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
  size_t layer = 1;
  for (layer; layer < l; layer++) Layers[layer].frontPropagate();
  // FILL IN HERE
  for (layer; layer >= 1; layer--) Layers[layer].backPropagate();
}

// Decode the states of neurons in all layers
template<typename T>
void Brain<T>::decode() {for (Layer<T>& L : Layers) L.decode();}

// Encode the states of neurons in all layers
template<typename T>
void Brain<T>::encode() {for (Layer<T>& L : Layers) L.encode();}

// Test the states of neurons in all layers
template<typename T>
void Brain<T>::test() {for (Layer<T>& L : Layers) L.test();}

// Backward pass through the network (Backpropagation)
template<typename T>
void Brain<T>::backwardPass(const std::vector<double>& expectedOutputs) {
  // Calculate errors for output layer
  std::vector<T> outputErrors;
  std::vector<T> outputs = Layers.back().getOutputs();
  for (int i = 0; i < outputs.size(); ++i) {
    outputErrors.push_back(outputs[i] - expectedOutputs[i]);
  }
  Layers.back().setErrors(outputErrors);

  // Update weights starting from the output layer
  for (int i = Layers.size() - 1; i >= 0; --i) {
    std::vector<double> inputs;
    inputs = (i == 0) ? std::vector<double>() : Layers[i - 1].getOutputs();
    // Inputs are outputs of previous layer
    // Input layer has no inputs
    Layers[i].updateWeights(inputs, learningRate);
    // Calculate errors for hidden layers (if any)
    if (i > 0) {
      std::vector<double> errors;
      for (int j = 0; j < Layers[i].getOutputs().size(); ++j) {
        double error = 0.;
        for (const auto& neuron : Layers[i].neurons) {
          error += neuron.getError() * neuron.activatePrime(neuron.getOutput()) * neuron.getOutput();
        }
        errors.push_back(error);
      }
      Layers[i - 1].setErrors(errors);
    }
  }
}