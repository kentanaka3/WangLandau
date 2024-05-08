#include "gtest/gtest.h"

#include "main.hpp"

TEST(TestFNN, Init) {
  std::string filename, network;
  filename = "test/data/FFNN/model.inp";
  WL * myWL = set_problem(MLP_STR, filename);
  // Parameters
  ASSERT_EQ(myWL->min_MCS, 200);
  ASSERT_EQ(myWL->log_F_end, 1e-6);
  ASSERT_EQ(myWL->delX, 1.);
  ASSERT_EQ(myWL->flatness, 0.95);
  ASSERT_EQ(myWL->check_histo, 100);
  ASSERT_EQ(myWL->N_rand, 200);
  ASSERT_EQ(myWL->prob_single, 1.);
  ASSERT_EQ(myWL->explorable[0], 0);
  ASSERT_EQ(myWL->explorable[1], 22);
  ASSERT_EQ(myWL->filepath, "test/data/FNN/");
  /*
  // Configuration
  ASSERT_EQ(myWL->N_inputs, 22);
  ASSERT_EQ(myWL->actID, 0);
  ASSERT_EQ(myWL->N_layers, 2);
  ASSERT_EQ(myWL->output_file, "output.dat");
  ASSERT_EQ(myWL->input_file, "input.dat");
  ASSERT_EQ(myWL->N_units[0], 11);
  ASSERT_EQ(myWL->N_units[1], 9);
  ASSERT_EQ(myWL->N_units[2], 1);
  ASSERT_EQ(myWL->maxNorm, 22);
  ASSERT_EQ(myWL->layerProb[0], 1);
  ASSERT_EQ(myWL->layerProb[1], 1);
  */
  delete myWL;
}