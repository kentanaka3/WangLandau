#include "FFNN.hpp"

MLP::MLP(const std::string& file) {
  filepath = std::filesystem::path(file).remove_filename().string();
  paramRead(file);
  #if DEBUG >= 3
  paramPrint();
  #endif
  configRead(file);

}

MLP::~MLP() {
}

void MLP::configRead(const std::string& file) {
  
}