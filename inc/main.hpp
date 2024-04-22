#include "utils.hpp"

#include "FFNN.hpp"

const std::string FFNN_STR = "FFNN";

WL * set_problem(const std::string& network, const std::string& filename) {
  if (network == FFNN_STR) {
    return new FFNN(filename);
  } else {
    std::cerr << "CRITICAL: WL " << network.c_str() \
              << " not implemented yet." << std::endl;
    exit(EXIT_FAILURE);
  }
}