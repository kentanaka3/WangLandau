#include "gtest/gtest.h"
#include "utils.hpp"

TEST(TestVector, doubles) {
  std::string filename;
  filename = "test/data/FFNN/Network/B000.dat";
  std::vector<double> vec = readVec(filename, 9);
  ASSERT_DOUBLE_EQ(vec[0], 3.111656382679939270e-02);
  ASSERT_DOUBLE_EQ(vec[4], -1.502808742225170135e-02);
  ASSERT_DOUBLE_EQ(vec[8], -1.059713889844715595e-03);
}

TEST(TestMatrix, doubles) {
  std::string filename;
  filename = "test/data/FFNN/Network/W000.dat";
  std::vector<std::vector<double>> mtx = readMtx(filename, 9, 11);
  ASSERT_DOUBLE_EQ(mtx[0][0], 7.201808691024780273e-02);
  ASSERT_DOUBLE_EQ(mtx[4][4], -4.726820159703493118e-03);
  ASSERT_DOUBLE_EQ(mtx[8][10], 8.432347094640135765e-04);
}

TEST(TestActFun, Tanh) {
  double (*act_fun) (double); // activation function
  set_act(0, &act_fun);
  ASSERT_DOUBLE_EQ(act_fun(-1), -0.76159415595576485);
  ASSERT_DOUBLE_EQ(act_fun(0), 0);
  ASSERT_DOUBLE_EQ(act_fun(1), 0.76159415595576485);
}

TEST(TestActFun, ReLU) {
  double (*act_fun) (double); // activation function
  set_act(1, &act_fun);
  ASSERT_DOUBLE_EQ(act_fun(-1), 0);
  ASSERT_DOUBLE_EQ(act_fun(0), 0);
  ASSERT_DOUBLE_EQ(act_fun(1), 1);
}

TEST(TestParams, ReadParam) {
  std::map<std::string, std::string> params;
  params["double"]  = "2";
  params["float"]   = "3";
  params["int"]     = "4";
  params["bool"]    = "5";
  params["string"]  = "6";
  params["ncores"]  = "4";
  std::string line = "";
  line = "# double    0.95";
  params = readParam(line, params);
  line = "# string     MLP";
  params = readParam(line, params);
  line = "# int        200";
  params = readParam(line, params);
  line = "# bool      TRUE";
  params = readParam(line, params);
  line = "# float   3.1415";
  params = readParam(line, params);
  line = "# ncores 4";
  params = readParam(line, params);
  ASSERT_EQ(params["double"], "0.95");
  ASSERT_EQ(params["bool"], "TRUE");
  ASSERT_EQ(params["int"], "200");
  ASSERT_EQ(params["string"], "MLP");
  ASSERT_EQ(params["float"], "3.1415");
  params["double"]  = "2";
  line = "# double    1e-6";
  params = readParam(line, params);
  params["bool"]    = "5";
  line = "# bool     False";
  params = readParam(line, params);
  ASSERT_EQ(params["bool"], "False");
  ASSERT_EQ(params["double"], "1e-6");
}

TEST(TestParams, ParamVec) {
  std::map<std::string, std::string> params;
  const int N = 3;
  std::vector<double> myDoubleParams(N, 0.);
  std::vector<int> myIntegerParams(N, 0);
  params["myDoubleParams"] = "6";
  params["myIntegerParams"] = "6";
  std::string line = "";
  line = "# myDoubleParams    0.95 1 3.0";
  params = readParam(line, params);
  line = "# myIntegerParams    1 2 3";
  params = readParam(line, params);
  ASSERT_EQ(params["myDoubleParams"], "0.95 1 3.0");
  ASSERT_EQ(params["myIntegerParams"], "1 2 3");
  param2vec(params["myIntegerParams"], myIntegerParams, N);
  ASSERT_EQ(myIntegerParams[0], 1);
  ASSERT_EQ(myIntegerParams[1], 2);
  ASSERT_EQ(myIntegerParams[2], 3);
  param2vec(params["myDoubleParams"], myDoubleParams, N);
  ASSERT_EQ(myDoubleParams[0], 0.95);
  ASSERT_EQ(myDoubleParams[1], 1.);
  ASSERT_EQ(myDoubleParams[2], 3.);
}