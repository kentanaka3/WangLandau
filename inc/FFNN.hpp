#include "WL.hpp"
#include <filesystem>

std::string MLP_STR = "MLP";

class MLP : public WL {
private:
  /* data */
public:
  // Number of Layers
  int N_layers;
  // Update Probability per Layer (layerProb[0] is the first hidden layer)
  std::vector<double> layerProb;

  // Number of Input patterns
  int N_inputs;

  // Number of Units per Layer (N_units[0] is the Input Layer)
  std::vector<int> N_units;

  // Input patterns
  std::vector<std::vector<double>> input;
  // Output targets
  std::vector<double> output;

  // Network Weights, Layer x n_to x n_from
  std::vector<std::vector<std::vector<double>>> W;

  // Biases, Layer x n_to
  std::vector<std::vector<double>> biases;

  // Input file path
  std::string input_file;
  // Output file path
  std::string output_file;

  // Active nodes
  std::vector<std::vector<bool>> nodeState;
  std::vector<std::vector<int>> nodeChanged;

  // Activation Function
  double (*act) (double);
  // ID for Activation Function
  //  0 = Tanh
  //  1 = ReLU
  int actID;
  MLP(const std::string& filename);
  ~MLP();

  double compEnergy() override;
  void moveAccepted() override;
  double moveProposed() override;
  void moveSingleProposed() override;
  void moveRejected() override;

  void confRead(const std::string& file);
  void confPrint();
};