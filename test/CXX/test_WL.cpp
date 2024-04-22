#include "gtest/gtest.h"

#include "main.hpp"

TEST(TestFFNN, Init) {
  std::string filename, network;
  filename = "test/data/FFNN/model.inp";
  WL * myWL = set_problem(MLP_STR, filename);
  ASSERT_EQ(myWL->filepath, "test/data/FFNN/");
}