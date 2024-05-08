#include "utils.hpp"

#include "FNN.hpp"

WL * set_problem(const std::string& network, const std::string& filename) {
  if (network == MLP_STR) {
    return new MLP(filename);
  } else {
    std::cerr << "CRITICAL: " << network << " not implemented yet." \
              << std::endl;
    exit(EXIT_FAILURE);
  }
}