#include "FFNN.hpp"

MLP::MLP(const std::string& file) {
  filepath = std::filesystem::path(file).remove_filename().string();
  
}

MLP::~MLP() {
}