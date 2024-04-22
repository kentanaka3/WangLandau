#include "utils.hpp"

#include "FFNN.hpp"

WL * set_problem(const std::string& network, const std::string& filename) {
  if (network == MLP_STR) {
    return new MLP(filename);
  } else {
    std::cerr << "CRITICAL: WL " << network.c_str() \
              << " not implemented yet." << std::endl;
    exit(EXIT_FAILURE);
  }
}