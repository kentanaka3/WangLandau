#include "utils.hpp"

#include <complex>
#include <filesystem>
#include <random>

double binaryCrossEntropy(const std::vector<double>& y_true,
													const std::vector<double>& y_pred) {
	if (y_true.size() != y_pred.size()) {
		std::cerr << "Error: Size mismatch between true and predicted values.\n";
		return -1; // Return error code
	}
	double sumCrossEntropy = 0.0;
	for (size_t i = 0; i < y_true.size(); ++i) {
		sumCrossEntropy += y_true[i] * std::log(y_pred[i]) \
											+ (1 - y_true[i]) * std::log(1 - y_pred[i]);
	}
	return -sumCrossEntropy / static_cast<double>(y_true.size());
}

const std::complex<double> INIT_STATE_CMPLX = 1.;
const bool INIT_STATE_ISING = true;
const size_t INIT_STATE_POTTS = 0;
const double INIT_STATE_SIGMA = 1.;
const std::ostringstream DATA_PATH{"data/"};
const unsigned int MARKOVIANITY = 0;

std::random_device rndm;
std::mt19937_64 gen(rndm());
std::uniform_int_distribution<> unf_dist(-1., 1.);

class neuron {
private:
public:
  // Bias
  double Bias;
  // Value
  double Value = INIT_STATE_SIGMA;
  // State
  bool State = true;
  // Past Values
  std::vector<double> Past;
  // Error
  double Error;
  // Weights
  std::vector<double> Weights;
  int N_connections;
  bool train;
  neuron(const std::string& file);
  ~neuron() {};
  void HelloWorld();
  double forward(const std::vector<double>& input);
};

// Definitions for neuron class methods
neuron::neuron(const std::string& file) : \
  Past(std::vector<double>(MARKOVIANITY, INIT_STATE_SIGMA)) {
  std::cout << "Creating Neuron @ " << file << std::endl;
  std::map<std::string, std::string> params;
  params["Type"] = "6";
  params["LrnRate"] = "2";
  params["Bias"] = "3";
  params["N_connections"] = "4";
  std::ifstream fr(file);
  if (!fr.is_open()) {
    std::cerr << "Error: Unable to init file " << file << "\n";
  }
  std::string line;
  // Read Parameters by parsing with RegEx
  while (std::getline(fr, line)) params = paramMap(line, params);
  fr.close();

  N_connections = std::atoi(params["N_connections"].c_str());
  for (size_t i = 0; i < N_connections; i++) Weights.push_back(unf_dist(gen));
  Bias = unf_dist(gen);
}

void neuron::HelloWorld() {std::cout << this->Value << ", ";}

double neuron::forward(const std::vector<double>& input) {
  double sum = 0.;
  for (size_t i = 0; i < N_connections; i++) {sum += input[i] * Weights[i];}
  return relu(sum + Bias);
}














int MAX_NEURONS = 1000;

class Layer {
private:
public:
// Learning Rate
  double LrnRate;
  // Neurons
  std::vector<neuron> Neurons;
  size_t N_nodes;
  size_t N_connections;
  std::string filename;
  std::vector<std::vector<double>> Weights();
  std::vector<double> Weights(size_t i);
  double Weights(size_t i, size_t j);
  void HelloWorld();
  std::vector<double> forward(const std::vector<double>& input);
  Layer(const size_t& n_nodes, const size_t& n_connections,
        const std::string& file);
  ~Layer() {};
};

// Definitions for Layer class methods
std::vector<std::vector<double>> Layer::Weights() {
  std::vector<std::vector<double>> W(N_nodes,
                                     std::vector<double>(N_connections, 0.));
  for (size_t n = 0; n < N_nodes; n++) {
    int i = 0;
    for (const auto& w: Neurons[n].Weights) W[n][i++] = w;
  }
  return W;
}

std::vector<double> Layer::Weights(size_t i) {return Neurons[i].Weights;}

double Layer::Weights(size_t i, size_t j) {return Neurons[i].Weights[j];}

void Layer::HelloWorld() {for (auto& n : Neurons) n.HelloWorld();}

Layer::Layer(const size_t& n_nodes, const size_t& n_connections,
             const std::string& file) : N_nodes(n_nodes),
  N_connections(n_connections), filename(file), LrnRate(0.9) {
  std::cout << "Creating Layer @ " << filename << ", w/ " << N_nodes \
            << " nodes and " << N_connections << " connections." << std::endl;
  std::filesystem::create_directories(file);
  // Create Layer file
  std::ostringstream filepath("");
  filepath << file << "/L.dat";
  std::ofstream fp;
  fp.open(filepath.str());
  if (fp.is_open()) {
    fp << "# Type           double" << std::endl \
       << "# LrnRate        0.1" << std::endl \
       << "# Bias           10." << std::endl \
       << "# N_connections  " << N_connections << std::endl;
    fp.close();
  } else std::cerr << "Failed to open file : " << filepath.str() << std::endl;
  Neurons.resize(N_nodes, neuron(filepath.str()));
  // Create Weights file
  filepath.str("");
  filepath << file << "/W.dat";
  // Print Matrix
  printMtx(filepath.str(), N_nodes, N_connections, this->Weights());
}

std::vector<double> Layer::forward(const std::vector<double>& input) {
  std::vector<double> v(N_nodes);
  for (size_t i = 0; i < N_nodes; i++) v[i] = Neurons[i].forward(input);
  return v;
}





















int MAX_LAYERS = 100;

class Brain {
public:
  size_t N_Layers;
  double flopRate;
  std::vector<Layer> Layers;
  std::string filename;
  Brain(double fR, std::string file);
  ~Brain() {};
  void HelloWorld();
  std::vector<double> forward(std::vector<double> X);
  void backPropagate(const std::vector<double>& X,
                     const std::vector<double>& Y);
  void train(const std::vector<std::vector<double>>& train_inputs,
             const std::vector<std::vector<double>>& train_outputs,
             int epochs);
};
// Definitions for Brain class methods
Brain::Brain(double fR, std::string file) : \
  flopRate(fR), \
  filename(std::filesystem::path(file).replace_filename("Brain/")) {
  std::cout << "Creating Brain from " << filename << std::endl;
  std::filesystem::create_directories(filename);
  std::ifstream fr(file);
  if (!fr.is_open()) {
    std::cerr << "Error: Unable to init file " << file << "\n";
  }
  std::map<std::string, std::string> params;
  params["N_Layers"] = "4";
  params["Layers"] = "6";
  std::string line;
  // Read Parameters by parsing with RegEx
  while (std::getline(fr, line)) params = paramMap(line, params);
  fr.close();
  N_Layers = std::atoi(params["N_Layers"].c_str());
  std::vector<int> L(N_Layers, 0);
  param2vec(params["Layers"], L, N_Layers);
  if (N_Layers + 1 <= MAX_LAYERS) {
    for (size_t layer = 0; layer < N_Layers; layer++) {
      std::ostringstream filepath("");
      filepath << filename << std::setfill('0') \
               << std::setw(static_cast<int>(log10(MAX_LAYERS))) << layer;
      Layers.push_back(Layer(L[layer], !(layer) ? 0 : L[layer - 1],
                       filepath.str()));
    }
  }
}

void Brain::HelloWorld() {
  size_t i = 0;
  for (auto& layer : Layers) {
    std::cout << "L" << std::setfill('0') \
              << std::setw(static_cast<int>(log10(MAX_LAYERS))) << i << ": ";
    layer.HelloWorld();
    std::cout << std::endl;
    i++;
  }
}

std::vector<double> Brain::forward(std::vector<double> X) {
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
  for (auto& L : Layers) {X = L.forward(X);}
  return X;
}

void Brain::backPropagate(const std::vector<double>& X,
                          const std::vector<double>& Y) {
  // Forward pass to get predicted output
  std::vector<double> prediction = forward(X);

  // Compute error between predicted output and actual output
  std::vector<double> error(prediction.size(), 0.);
  #pragma omp parallel for shared(error)
  for (size_t i = 0; i < prediction.size(); ++i)
    error[i] = Y[i] - prediction[i];

  // Backpropagation to update weights
  for (size_t layer = N_Layers - 1; layer >= 0; --layer) {
    Layer& currentLayer = Layers[layer];
    Layer* prevLayer = (layer > 0) ? &Layers[layer - 1] : nullptr;

    // Compute gradients for the current layer
    std::vector<double> gradients(currentLayer.N_nodes);
    for (size_t n = 0; n < currentLayer.N_nodes; ++n) {
      double neuronOutput = currentLayer.Neurons[n].Value;
      gradients[n] = error[n] * neuronOutput * (1 - neuronOutput);
    }

    // Update weights for the current layer
    for (size_t n = 0; n < currentLayer.N_nodes; ++n) {
      neuron& currentNeuron = currentLayer.Neurons[n];
      for (size_t weightIndex = 0; weightIndex < currentNeuron.Weights.size(); ++weightIndex) {
        double deltaWeight = (prevLayer != nullptr) ?
                              prevLayer->Neurons[weightIndex].Value * gradients[n] :
                              X[weightIndex] * gradients[n];
        currentNeuron.Weights[weightIndex] += currentLayer.LrnRate * deltaWeight;
      }
      // Update bias
      currentNeuron.Bias += currentLayer.LrnRate * gradients[n];
    }

    // Update error for the next layer
    if (prevLayer != nullptr) {
      std::vector<double> newError(prevLayer->N_nodes, 0.0);
      for (size_t prevNeuronIndex = 0; prevNeuronIndex < prevLayer->N_nodes; ++prevNeuronIndex) {
        for (size_t n = 0; n < currentLayer.N_nodes; ++n) {
          newError[prevNeuronIndex] += gradients[n] *
                                      currentLayer.Neurons[n].Weights[prevNeuronIndex];
        }
      }
      error = newError;
    }
  }
}

// Train the network using stochastic gradient descent
void Brain::train(const std::vector<std::vector<double>>& train_inputs,
                  const std::vector<std::vector<double>>& train_outputs,
                  int epochs) {
  for (int epoch = 0; epoch < epochs; ++epoch)
    for (size_t i = 0; i < train_inputs.size(); ++i)
      backPropagate(train_inputs[i], train_outputs[i]);
}

class master {
private:
public:
  master(/* args */);
  ~master();
};

master::master(/* args */) {
}

master::~master() {
}
