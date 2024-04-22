#include "WL.hpp"
#include <filesystem>

std::string MLP_STR = "MLP";

class MLP : public WL {
private:
  /* data */
public:
  MLP(const std::string& filename);
  ~MLP();
  void configRead(const std::string& file);
};