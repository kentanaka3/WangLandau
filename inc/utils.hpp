#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <regex>
#include <map>

#include <math.h>

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

std::vector<double> readVec(const std::string& filename, const int& N);

std::vector<std::vector<double>> readMtx(const std::string& filename,
																				 const int& rows, const int& cols);

void printVec(const std::string& filename, std::vector<double> vec,
              const int& N, const int& ax);

void printMtx(const std::string& filename, const int& rows, const int& cols,
							std::vector<std::vector<double>> mtx);

std::string lastLine(const std::string& filepath);

double relu(const double x);

template <typename T>
int sgn(const T& val);

void set_act(const int& actnum, double (**act) (double));

std::map<std::string, std::string> paramMap(
	const std::string line, std::map<std::string, std::string> params);

void param2vec(const std::string& param, std::vector<int>& vec, const int& N);
void param2vec(const std::string& param, std::vector<double>& vec,
							 const int& N);

int argMax(std::vector<double> vec, int length);