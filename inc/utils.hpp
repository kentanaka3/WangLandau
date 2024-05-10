#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <regex>
#include <map>

#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif


#ifndef DEBUG
/* = DEBUG =
 * -1: No comments (Performance)
 * 0: (Default) Minimal comments
 * 1: Helper comments will be printed
 * 2: File comments will be printed
 * 3: DEBUG, all comments will be printed
 */
#define DEBUG 3
#endif

std::vector<double> tanh(const std::vector<double> x);
double D_tanh(const double x);
std::vector<double> D_tanh(const std::vector<double> x);
double sigmoid(const double x);
std::vector<double> sigmoid(const std::vector<double> x);
double D_sigmoid(const double x);
std::vector<double> D_sigmoid(const std::vector<double> x);
double relu(const double x);
std::vector<double> relu(const std::vector<double> x);
double D_relu(const double x);
std::vector<double> D_relu(const std::vector<double> x);
double leaky_relu(const double x, const double y);
std::vector<double> leaky_relu(const std::vector<double> x, const double y);
double D_leaky_relu(const double x, const double y);
std::vector<double> D_leaky_relu(const std::vector<double> x, const double y);

std::vector<double> softmax(const std::vector<double> x, const double temp);
std::vector<double> D_softmax(const std::vector<double> x, const double temp);

template <typename T>
int sgn(const T& val);

double meanSquaredError(const std::vector<double>& y_true,
												const std::vector<double>& y_pred);

double binaryCrossEntropy(const std::vector<double>& y_true,
													const std::vector<double>& y_pred);

double crossEntropy(const std::vector<double>& y_true,
										const std::vector<double>& y_pred);

std::vector<double> readVec(const std::string& filename, const int& N);

std::vector<std::vector<double>> readMtx(const std::string& filename,
																				 const int& rows, const int& cols);

void printVec(const std::string& filename, std::vector<double> vec,
              const int& N, const int& ax);
void printVec(const std::string& filename, std::vector<double> vec,
              const int& N);

void printMtx(const std::string& filename, const size_t& rows,
							const size_t& cols, std::vector<std::vector<double>> mtx);

std::string lastLine(const std::string& filepath);

void set_act(const int& actnum, double (**act) (const double),
						 double (**D_act) (const double));
void set_act(const int& actnum, double (**act) (const double, const double),
						 double (**D_act) (const double, const double), const double& y);
void set_act(const int& actnum,
						 std::vector<double> (**act) (const std::vector<double>),
						 std::vector<double> (**D_act) (const std::vector<double>));
void set_act(const int& actnum,
						 std::vector<double> (**act) (const std::vector<double>, const double),
						 std::vector<double> (**D_act) (const std::vector<double>, const double),
						 const double& y);

std::map<std::string, std::string> paramMap(
	const std::string line, std::map<std::string, std::string> params);

void param2vec(const std::string& param, std::vector<int>& vec,
							 const size_t& N);
void param2vec(const std::string& param, std::vector<int>& vec,
							 const size_t& N, const size_t& start);
void param2vec(const std::string& param, std::vector<int>& vec,
							 const size_t& N, const size_t& start, const size_t& offset);
void param2vec(const std::string& param, std::vector<double>& vec,
							 const size_t& N);
void param2vec(const std::string& param, std::vector<double>& vec,
							 const size_t& N, const size_t& start);
void param2vec(const std::string& param, std::vector<double>& vec,
							 const size_t& N, const size_t& start, const size_t& offset);

size_t argMax(const std::vector<double> vec);
size_t argMax(const std::vector<double> vec, const size_t& N);