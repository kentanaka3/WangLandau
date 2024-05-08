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
  double (*act_fun) (const double); // activation function
  double (*D_act) (const double); // derivative
  set_act(0, &act_fun, &D_act);
  ASSERT_DOUBLE_EQ(act_fun(-1), -0.76159415595576485);
  ASSERT_DOUBLE_EQ(act_fun(0), 0);
  ASSERT_DOUBLE_EQ(act_fun(1), 0.76159415595576485);
}

TEST(TestActFun, TanhVec) {
  std::vector<double> (*act_fun) (const std::vector<double>); // activation function
  std::vector<double> (*D_act) (const std::vector<double>); // derivative
  set_act(1, &act_fun, &D_act);
  std::vector<double> x = {1, 2, 3};
  std::vector<double> y = act_fun(x);
  ASSERT_DOUBLE_EQ(y[0], 1);
  ASSERT_DOUBLE_EQ(y[1], 2);
  ASSERT_DOUBLE_EQ(y[2], 3);
}

TEST(TestActFun, ReLU) {
  double (*act_fun) (const double); // activation function
  double (*D_act) (const double); // derivative
  set_act(1, &act_fun, &D_act);
  ASSERT_DOUBLE_EQ(act_fun(-1), 0);
  ASSERT_DOUBLE_EQ(act_fun(0), 0);
  ASSERT_DOUBLE_EQ(act_fun(1), 1);
}

TEST(TestActFun, ReLUVec) {
  std::vector<double> (*act_fun) (const std::vector<double>); // activation function
  std::vector<double> (*D_act) (const std::vector<double>); // derivative
  set_act(1, &act_fun, &D_act);
  std::vector<double> x = {1, 2, 3};
  std::vector<double> y = act_fun(x);
  ASSERT_DOUBLE_EQ(y[0], 1);
  ASSERT_DOUBLE_EQ(y[1], 2);
  ASSERT_DOUBLE_EQ(y[2], 3);
}

TEST(TestActFun, Sigmoid) {
  double (*act_fun) (const double); // activation function
  double (*D_act) (const double); // derivative
  set_act(2, &act_fun, &D_act);
  ASSERT_DOUBLE_EQ(act_fun(-1), 0.2689414213699951);
  ASSERT_DOUBLE_EQ(act_fun(0), 0.5);
  ASSERT_DOUBLE_EQ(act_fun(1), 0.7310585786300049);
}

TEST(TestActFun, SigmoidVec) {
  std::vector<double> (*act_fun) (const std::vector<double>); // activation function
  std::vector<double> (*D_act) (const std::vector<double>); // derivative
  set_act(2, &act_fun, &D_act);
  std::vector<double> x = {1, 2, 3};
  std::vector<double> y = act_fun(x);
  ASSERT_DOUBLE_EQ(y[0], 0.7310585786300049);
  ASSERT_DOUBLE_EQ(y[1], 0.8807970779778823);
  ASSERT_DOUBLE_EQ(y[2], 0.9525741268224334);
}

TEST(TestActFun, LeakyReLU) {
  double (*act_fun) (const double, const double); // activation function
  double (*D_act) (const double, const double); // derivative
  set_act(3, &act_fun, &D_act, 0.01);
  ASSERT_DOUBLE_EQ(act_fun(-1, 0.01), -0.01);
  ASSERT_DOUBLE_EQ(act_fun(0, 0.01), 0);
  ASSERT_DOUBLE_EQ(act_fun(1, 0.01), 1);
}

TEST(TestActFun, LeakyReLUVec) {
  std::vector<double> (*act_fun) (const std::vector<double>, const double); // activation function
  std::vector<double> (*D_act) (const std::vector<double>, const double); // derivative
  set_act(3, &act_fun, &D_act, 0.01);
  std::vector<double> x = {1, 2, 3};
  std::vector<double> y = act_fun(x, 0.01);
  ASSERT_DOUBLE_EQ(y[0], 1);
  ASSERT_DOUBLE_EQ(y[1], 2);
  ASSERT_DOUBLE_EQ(y[2], 3);
}

TEST(TestActFun, Softmax) {
  std::vector<double> (*act_fun) (const std::vector<double>,
                                  const double); // activation function
  std::vector<double> (*D_act) (const std::vector<double>,
                                const double); // derivative
  set_act(4, &act_fun, &D_act, 1);
  std::vector<double> x = {1, 2, 3};
  std::vector<double> y = act_fun(x, 1);
  ASSERT_DOUBLE_EQ(y[0], 0.09003057317038046);
  ASSERT_DOUBLE_EQ(y[1], 0.24472847105479767);
  ASSERT_DOUBLE_EQ(y[2], 0.6652409557748219);
}

TEST(TestLossFun, MSE) {
  std::vector<double> y_true = {1, 2, 3};
  std::vector<double> y_pred = {1, 2, 3};
  ASSERT_DOUBLE_EQ(meanSquaredError(y_true, y_pred), 0);
  y_pred = {1, 2, 4};
  ASSERT_DOUBLE_EQ(meanSquaredError(y_true, y_pred), 1./3);
  y_pred = {1, 3, 3};
  ASSERT_DOUBLE_EQ(meanSquaredError(y_true, y_pred), 1./3);
  y_pred = {2, 2, 3};
  ASSERT_DOUBLE_EQ(meanSquaredError(y_true, y_pred), 1./3);
}

TEST(TestLossFun, BCE) {
  std::vector<double> y_true = {1, 0, 1};
  std::vector<double> y_pred = {1, 0, 1};
  ASSERT_DOUBLE_EQ(binaryCrossEntropy(y_true, y_pred), 0);
  y_pred = {1, 0, 0};
  ASSERT_DOUBLE_EQ(binaryCrossEntropy(y_true, y_pred), 0.6931471805599453);
  y_pred = {0, 0, 1};
  ASSERT_DOUBLE_EQ(binaryCrossEntropy(y_true, y_pred), 0.6931471805599453);
  y_pred = {0, 1, 1};
  ASSERT_DOUBLE_EQ(binaryCrossEntropy(y_true, y_pred), 0.6931471805599453);
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
  params = paramMap(line, params);
  line = "# string     MLP";
  params = paramMap(line, params);
  line = "# int        200";
  params = paramMap(line, params);
  line = "# bool      TRUE";
  params = paramMap(line, params);
  line = "# float   3.1415";
  params = paramMap(line, params);
  line = "# ncores 4";
  params = paramMap(line, params);
  ASSERT_EQ(params["double"], "0.95");
  ASSERT_EQ(params["bool"], "TRUE");
  ASSERT_EQ(params["int"], "200");
  ASSERT_EQ(params["string"], "MLP");
  ASSERT_EQ(params["float"], "3.1415");
  params["double"]  = "2";
  line = "# double    1e-6";
  params = paramMap(line, params);
  params["bool"]    = "5";
  line = "# bool     False";
  params = paramMap(line, params);
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
  params = paramMap(line, params);
  line = "# myIntegerParams    1 2 3";
  params = paramMap(line, params);
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