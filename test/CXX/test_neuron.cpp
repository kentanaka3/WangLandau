#include "gtest/gtest.h"
#include "neuron.hpp"

TEST(TestBrain, HelloWorld) {
  Brain myBrain("test/data/Neuron/FNN.init");
  ASSERT_EQ(myBrain.N_layers, 3);
  ASSERT_EQ(myBrain.filename, "test/data/Neuron/Brain/");
  ASSERT_EQ(myBrain.flopRate, 0.001);
  ASSERT_EQ(myBrain.Type, "FNN");
  ASSERT_EQ(myBrain.Layers.size(), 3);
  ASSERT_EQ(myBrain.Layers[0].N_nodes, 2);
  ASSERT_EQ(myBrain.Layers[2].N_nodes, 1);
  ASSERT_EQ(myBrain.Layers[0].Neurons.size(), 2);
  ASSERT_EQ(myBrain.Layers[2].Neurons.size(), 1);
}