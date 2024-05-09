#include "gtest/gtest.h"
#include "neuron.hpp"

TEST(TestNeuron, Init) {
  neuron myNeuron("test/data/Neuron/Brain/01/L.dat");
  ASSERT_EQ(myNeuron.N_connections, 2);
}

TEST(TestBrain, init) {
  Brain myBrain("test/data/Neuron/FNN.init");
  ASSERT_EQ(myBrain.N_layers, 5);
  ASSERT_EQ(myBrain.filename, "test/data/Neuron/Brain/");
  ASSERT_EQ(myBrain.flopRate, 0.001);
  ASSERT_EQ(myBrain.Type, "FNN");
  ASSERT_EQ(myBrain.Layers[0].N_nodes, 2);
  ASSERT_EQ(myBrain.Layers[2].N_nodes, 5);
  myBrain.HelloWorld();
}