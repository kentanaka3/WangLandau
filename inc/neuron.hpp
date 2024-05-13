#include "utils.hpp"

#include <complex>
#include <filesystem>
#include <random>
#ifdef _MPI
#include <mpi.h>
int MPI_LSIZE, MPI_LRANK, offset, chunksize;
#endif

const std::complex<double> INIT_STATE_CMPLX = 1.;
const bool INIT_STATE_ISING = true;
const size_t INIT_STATE_POTTS = 0;
const double INIT_STATE_SIGMA = 1.;
const std::ostringstream DATA_PATH{"data/"};
const unsigned int MARKOVIANITY = 1;
int MPI_GSIZE = 1;
int MPI_GRANK = 0;

std::random_device rndm;
std::mt19937_64 gen(rndm());
std::uniform_real_distribution<> unf_dist(-1., 1.);

/*****************************************************************************
 *
 *                                NEURON CLASS
 *
 *****************************************************************************/
class neuron {
private:
public:
  // Bias
  double Bias;
  // Value
  double Value;
  // State
  bool State;
  // Past Values
  std::vector<double> Past;
  // Weights
  std::vector<double> Weights;
  unsigned int N_connections;
  bool train;
  neuron(const std::string& file);
  ~neuron() {};
  void HelloWorld();
  double forward(const std::vector<double>& input);
};

// Definitions for neuron class methods
neuron::neuron(const std::string& file) : State(true), train(true), \
  Value(INIT_STATE_SIGMA), \
  Past(std::vector<double>(MARKOVIANITY, INIT_STATE_SIGMA)) {
  #if DEBUG >= 3
  if (MPI_GRANK == 0)
    std::cout << "DEBUG(3): Creating Neuron @ " << file << std::endl;
  #endif
  std::map<std::string, std::string> params;
  params["Type"] = "6";
  params["LrnRate"] = "2";
  params["N_connections"] = "4";
  std::ifstream fr(file);
  if (!fr.is_open()) {
    if (MPI_GRANK == 0)
      std::cerr << "CRITICAL: Unable to init file " << file << "\n";
    exit(EXIT_FAILURE);
  }
  std::string line;
  // Read Parameters by parsing with RegEx
  while (std::getline(fr, line)) params = paramMap(line, params);
  fr.close();

  N_connections = std::atoi(params["N_connections"].c_str());
  Weights.resize(N_connections, 0.);
  #pragma omp parallel for shared(Weights)
  for (size_t i = 0; i < N_connections; i++) Weights[i] = unf_dist(gen);
  Bias = unf_dist(gen);
}

void neuron::HelloWorld() {std::cout << this->Value << ", ";}

double neuron::forward(const std::vector<double>& input) {
  double sum = 0.;
  for (size_t i = 0; i < N_connections; i++) {sum += input[i] * Weights[i];}
  Value = relu(sum + Bias);
  return Value;
}

const unsigned short MAX_NEURONS = 1000;
/******************************************************************************
 *
 *                                LAYER CLASS
 *
 *****************************************************************************/
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
  std::vector<double> Weights(const size_t& i);
  double Weights(const size_t& i, const size_t& j);
  std::vector<double> Bias();
  double Bias(const size_t& i);
  std::vector<double> Value();
  double Value(const size_t& i);
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
  #pragma omp parallel for shared(W) collapse(2)
  for (size_t n = 0; n < N_nodes; n++) {
    for (size_t i = 0; i < N_connections; i++) W[n][i] = Weights(n, i);
  }
  return W;
}
std::vector<double> Layer::Weights(const size_t& i) {
  return Neurons[i].Weights;
}
double Layer::Weights(const size_t& i, const size_t& j) {
  return Neurons[i].Weights[j];
}

double Layer::Bias(const size_t& i) {return Neurons[i].Bias;}
std::vector<double> Layer::Bias() {
  std::vector<double> B(N_nodes, 0.);
  #pragma omp parallel for shared(B)
  for (size_t i = 0; i < N_nodes; i++) B[i] = Bias(i);
  return B;
};

std::vector<double> Layer::Value() {
  std::vector<double> V(N_nodes, 0.);
  #pragma omp parallel for shared(V)
  for (size_t i = 0; i < N_nodes; i++) V[i] = Value(i);
  return V;
}
double Layer::Value(const size_t& i) {return Neurons[i].Value;}

void Layer::HelloWorld() {for (auto& n : Neurons) n.HelloWorld();}

Layer::Layer(const size_t& n_nodes, const size_t& n_connections,
             const std::string& file) : N_nodes(n_nodes),
  N_connections(n_connections), filename(file), LrnRate(0.9) {
  #if DEBUG >= 3
  std::cout << "DEBUG(3): Creating Layer @ " << filename << ", w/ " << N_nodes\
            << " nodes and " << N_connections << " connections in Proc " \
            << MPI_GRANK << std::endl;
  #endif
  std::filesystem::create_directories(file);
  // Create Layer file
  std::ostringstream filepath("");
  filepath << file << "/L.dat";
  std::ofstream fp;
  fp.open(filepath.str());
  if (fp.is_open()) {
    fp << "# Type           double" << std::endl \
       << "# LrnRate        0.9" << std::endl \
       << "# N_connections  " << N_connections << std::endl;
    fp.close();
  } else {
    std::cerr << "CRITICAL: Failed to open file : " << filepath.str() \
              << std::endl;
    exit(EXIT_FAILURE);
  }
  Neurons.resize(N_nodes, neuron(filepath.str()));
  // Create Weights file
  filepath.str("");
  filepath << file << "/W.dat";
  printMtx(filepath.str(), N_nodes, N_connections, Weights());
  // Create Bias file
  filepath.str("");
  filepath << file << "/B.dat";
  printVec(filepath.str(), Bias(), N_nodes);
}

std::vector<double> Layer::forward(const std::vector<double>& input) {
  std::vector<double> v(N_nodes, 0.);
  for (size_t i = 0; i < N_nodes; i++) v[i] = Neurons[i].forward(input);
  return v;
}

const unsigned short MAX_LAYERS = 100;
/******************************************************************************
 *
 *                                BRAIN CLASS
 * TODO: This can be improved to be learned by N MPI processors working in
 *       parallel, see the following snippet:
 *
 *       L04 - In_0 ----------------- MPI_0 }
 *       L03 - In_1 ----------- MPI_1       } }
 *       L02 - In_2 ----------- MPI_1       } } } MPI_GSIZE = 3 = N
 *       L01 - In_3 ------MPI_2             } }
 *       L00 - In_4 ----------------- MPI_0 }
 *
 *       Each MPI processor will be responsible for a chunk of the layers and
 *       will communicate with the next processor to send the output of the
 *       last layer and receive the input of the first layer.
 *
 *****************************************************************************/
class Brain {
public:
  size_t N_layers;
  double flopRate;
  std::vector<Layer> Layers;
  std::string filename;
  std::string Type;
  unsigned short epochs;
  unsigned short testit;
  Brain(std::string file);
  ~Brain() {};
  void HelloWorld();
  std::vector<double> forward(std::vector<double> X);
  unsigned short backPropagate(const std::vector<double>& X,
                               const std::vector<double>& Y,
                              unsigned short& epochs);
  unsigned short train(std::vector<double>& train_input,
                       std::vector<double>& train_output,
                       unsigned short epochs);
  void train(std::vector<double>& train_input,
             std::vector<double>& train_output);
  void train(std::vector<std::vector<double>> train_inputs,
             std::vector<std::vector<double>> train_outputs);
  void train(std::vector<std::vector<double>> train_inputs,
             std::vector<std::vector<double>> train_outputs,
             std::vector<std::vector<double>> test_inputs,
             std::vector<std::vector<double>> test_outputs);
  void test(const std::vector<double>& test_input,
            const std::vector<double>& test_output);
};
// Definitions for Brain class methods
Brain::Brain(std::string file) : \
  N_layers(0), flopRate(0.), Type(""), epochs(0), testit(0), \
  filename(std::filesystem::path(file).replace_filename("Brain/")) {
  #if DEBUG >= 3
  if (MPI_GRANK == 0)
    std::cout << "DEBUG(3): Creating Brain from " << file << std::endl;
  #endif
  std::filesystem::create_directories(filename);
  std::ifstream fr(file);
  if (!fr.is_open()) {
    if (MPI_GRANK == 0)
      std::cerr << "CRITICAL: Unable to init file " << file << "\n";
    exit(EXIT_FAILURE);
  }
  std::map<std::string, std::string> params;
  params["N_layers"] = "4";
  params["Type"] = "6";
  params["Layers"] = "6";
  params["flopRate"] = "2";
  params["epochs"] = "4";
  params["testit"] = "4";
  std::string line;
  // Read Parameters by parsing with RegEx
  while (std::getline(fr, line)) params = paramMap(line, params);
  fr.close();
  testit = std::atoi(params["testit"].c_str());
  Type = params["Type"];
  epochs = std::atoi(params["epochs"].c_str());
  // Global number of Layers
  N_layers = std::atoi(params["N_layers"].c_str());
  if (2 > N_layers || N_layers > MAX_LAYERS) {
    if (MPI_GRANK == 0)
      std::cerr << "CRITICAL: Number of Layers exceeds MAX_LAYERS" \
                << std::endl;
    exit(EXIT_FAILURE);
  }
  #ifdef _MPI
  if (MPI_GSIZE == 1) {
    std::cerr << "CRITICAL: Number of processors must be at least 2" \
              << std::endl;
    exit(EXIT_FAILURE);
  }
  chunksize = (N_layers - 2) / (MPI_GSIZE - 1);
  offset = (N_layers - 2) % (MPI_GSIZE - 1);
  N_layers = MPI_GRANK ? chunksize + (MPI_GRANK <= offset) : 2;
  #endif
  std::vector<int> L(N_layers, 0);
  #ifdef _MPI
  if (MPI_GRANK) param2vec(params["Layers"], L, N_layers, 1 + MPI_GRANK);
  else {
    param2vec(params["Layers"], L, 1, MPI_GSIZE - 1);
    param2vec(params["Layers"], L, 1, 0, 1);
  }
  #else
  param2vec(params["Layers"], L, N_layers);
  #endif
  flopRate = std::atof(params["flopRate"].c_str());
  if (N_layers + 1 <= MAX_LAYERS) {
    for (size_t layer = 0; layer < N_layers; layer++) {
      std::ostringstream filepath("");
      filepath << filename << std::setfill('0') \
               << std::setw(static_cast<int>(log10(MAX_LAYERS))) \
               << layer + MPI_GRANK;
      Layers.push_back(Layer(L[layer], !(layer) ? 0 : L[layer - 1],
                       filepath.str()));
    }
  }
}

void Brain::HelloWorld() {
  if (MPI_GRANK) return;
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
  for (auto& L : Layers) {X = L.forward(X);}
  #ifdef _MPI
  MPI_Send(X.data(), X.size(), MPI_DOUBLE, MPI_GRANK,
           (MPI_GRANK + 1) % MPI_GSIZE, MPI_COMM_WORLD);
  #endif
  return X;
}

// Backpropagation algorithm
// X: Input vector
// Y: Target vector
//
// Only called by MPI_GRANK = 0
unsigned short Brain::backPropagate(const std::vector<double>& X,
                                    const std::vector<double>& Y,
                                    unsigned short& epochs) {
  // Forward pass to get predicted output
  std::vector<double> prediction = forward(X);
  #ifdef _MPI
  MPI_Send(prediction.data(), prediction.size(), MPI_DOUBLE, MPI_GRANK + 1, 0,
           MPI_COMM_WORLD);
  if (epochs > N_layers) {
    MPI_Recv(prediction.data(), prediction.size(), MPI_DOUBLE, MPI_GSIZE - 1,
             0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  } else {
    return ++epochs;
  }
  #endif

  // Compute error between predicted output and actual output
  std::vector<double> error(prediction.size(), 0.);
  #pragma omp parallel for shared(error)
  for (size_t i = 0; i < prediction.size(); ++i)
    error[i] = Y[i] - prediction[i];

  // Backpropagate the error
  for (size_t i = N_layers - 1; i > 0; i--) {
    // Compute the gradient of the error with respect to the output
    std::vector<double> gradient(prediction.size(), 0.);
    #pragma omp parallel for shared(gradient)
    for (size_t j = 0; j < prediction.size(); j++)
      gradient[j] = error[j] * D_relu(prediction[j]);
    // Compute the gradient of the error with respect to the weights
    std::vector<std::vector<double>> delta_weights(Layers[i].N_nodes,
                                                   std::vector<double>(Layers[i].N_connections, 0.));
    #pragma omp parallel for shared(delta_weights)
    for (size_t j = 0; j < Layers[i].N_nodes; j++) {
      for (size_t k = 0; k < Layers[i].N_connections; k++)
        delta_weights[j][k] = gradient[j] * Layers[i - 1].Value(k);
    }
    // Update the weights
    #pragma omp parallel for shared(Layers)
    for (size_t j = 0; j < Layers[i].N_nodes; j++) {
      for (size_t k = 0; k < Layers[i].N_connections; k++)
        Layers[i].Neurons[j].Weights[k] += Layers[i].LrnRate * delta_weights[j][k];
    }
    // Compute the gradient of the error with respect to the output of the previous layer
    std::vector<double> next_layer_error(Layers[i - 1].N_nodes, 0.);
    #pragma omp parallel for shared(next_layer_error)
    for (size_t j = 0; j < Layers[i - 1].N_nodes; j++) {
      for (size_t k = 0; k < Layers[i].N_nodes; k++)
        next_layer_error[j] += Layers[i].Neurons[k].Weights[j] * gradient[k];
    }
    // Update the error
    error = next_layer_error;
  }
  return ++epochs;
}

// Train the network using stochastic gradient descent
unsigned short Brain::train(std::vector<double>& train_input,
                            std::vector<double>& train_output,
                            unsigned short epochs) {
  if (MPI_GRANK == 0) backPropagate(train_input, train_output, epochs);
  else {
    #ifdef _MPI
    MPI_Recv(train_input.data(), train_input.size(), MPI_DOUBLE, MPI_GRANK - 1,
             0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    forward(train_input);
    MPI_Send(train_input.data(), train_input.size(), MPI_DOUBLE,
             MPI_GRANK < MPI_GSIZE - 1 ? MPI_GRANK + 1 : 0, 0, MPI_COMM_WORLD);
    #endif
  }
  return ++epochs;
}

void Brain::train(std::vector<double>& train_input,
                  std::vector<double>& train_output) {
  train(train_input, train_output, 0);
}

void Brain::train(std::vector<std::vector<double>> train_inputs,
                  std::vector<std::vector<double>> train_outputs) {
  int test_every = testit ? testit : 1;
  int test_count = 0;
  // Train the network
  size_t j = 0;
  for (size_t i = 0; i < test_every; i++) {
    for (j; j < i * train_inputs.size() / test_every; j++)
      train(train_inputs[j], train_outputs[j], 0);
    // Test the trained network
    if (testit) test(train_inputs[i], train_outputs[i]);
  }
}

void Brain::train(std::vector<std::vector<double>> train_inputs,
                  std::vector<std::vector<double>> train_outputs,
                  std::vector<std::vector<double>> test_inputs,
                  std::vector<std::vector<double>> test_outputs) {
  int test_every = testit ? testit : 1;
  int test_count = 0;
  // Train the network
  size_t j = 0;
  for (size_t i = 0; i < test_every; i++) {
    for (j; j < i * train_inputs.size() / test_every; j++)
      train(train_inputs[j], train_outputs[j], 0);
    // Test the trained network
    if (testit) test(test_inputs[i], test_outputs[i]);
  }
}

void Brain::test(const std::vector<double>& test_input,
                 const std::vector<double>& test_output) {
  double error = meanSquaredError(test_output, forward(test_input));
  std::cout << "Error: " << error << std::endl;
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
