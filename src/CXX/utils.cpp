#include "utils.hpp"

std::vector<double> tanh(const std::vector<double> x) {
	std::vector<double> output(x.size(), 0.);
	#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) output[i] = std::tanh(x[i]);
	return output;
}
double D_tanh(const double x) {
	double activated = tanh(x);
	return 1. - activated * activated;	// 1 - tanh^2(x)
}
std::vector<double> D_tanh(const std::vector<double> x) {
	std::vector<double> output(x.size(), 0.);
	#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) output[i] = D_tanh(x[i]);
	return output;
}
double sigmoid(const double x) {return 1./(1. + std::exp(-x));}
std::vector<double> sigmoid(const std::vector<double> x) {
	std::vector<double> output(x.size(), 0.);
	#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) output[i] = sigmoid(x[i]);
	return output;
}
double D_sigmoid(const double x) {
  double activated = sigmoid(x);
  return activated * (1 - activated); // sigmoid(x) * (1 - sigmoid(x))
}
std::vector<double> D_sigmoid(const std::vector<double> x) {
	std::vector<double> output(x.size(), 0.);
	#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) output[i] = D_sigmoid(x[i]);
	return output;
}
double relu(const double x) {return (x > 0.) ? x : 0.;}
std::vector<double> relu(const std::vector<double> x) {
	std::vector<double> output(x.size(), 0.);
	#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) output[i] = relu(x[i]);
	return output;
}
double D_relu(const double x) {return (x > 0.) ? 1. : 0.;}
std::vector<double> D_relu(const std::vector<double> x) {
	std::vector<double> output(x.size(), 0.);
	#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) output[i] = D_relu(x[i]);
	return output;
}
std::vector<double> softmax(const std::vector<double> x, const double temp) {
	std::vector<double> output(x.size(), 0.);
	double max_val = *std::max_element(x.begin(), x.end());
	double sum = 0;
	for (int i = 0; i < x.size(); i++) {
		output[i] = std::exp((x[i] - max_val) / temp);
		sum += output[i];
	}
	// Normalize
	#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) output[i] /= sum;
	return output;
}
std::vector<double> D_softmax(const std::vector<double> x,
															const double temp) {
	std::vector<double> output(x.size(), 0.);
	std::vector<double> softmax_x = softmax(x, temp);
	#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) output[i] = softmax_x[i] * (1. - softmax_x[i]);
	return output;
}

double leaky_relu(const double x, const double y) {
	return x > 0 ? x : y * x;
}
std::vector<double> leaky_relu(const std::vector<double> x, const double y) {
	std::vector<double> output(x.size(), 0.);
	#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) output[i] = leaky_relu(x[i], y);
	return output;
}
double D_leaky_relu(const double x, const double y) {
	return x > 0 ? 1 : y;
}
std::vector<double> D_leaky_relu(const std::vector<double> x, const double y) {
	std::vector<double> output(x.size(), 0.);
	#pragma omp parallel for
	for (int i = 0; i < x.size(); i++) output[i] = D_leaky_relu(x[i], y);
	return output;
}

template <typename T>
int sgn(const T& val) {
	return (T(0) < val) - (val <= T(0));
}

double meanSquaredError(const std::vector<double>& y_true,
												const std::vector<double>& y_pred) {
	if (y_true.size() != y_pred.size()) {
		std::cerr << "Error: Size mismatch between true and predicted values.\n";
		exit(EXIT_FAILURE);
	}
	double total = 0.;
	#pragma omp parallel for reduction(+:total)
	for (size_t i = 0; i < y_true.size(); ++i) {
		double error = y_true[i] - y_pred[i];
		total += error * error;
	}
	return total / static_cast<double>(y_true.size());
}

double binaryCrossEntropy(const std::vector<double>& y_true,
													const std::vector<double>& y_pred) {
	double total = 0.;
	#pragma omp parallel for reduction(+:total)
	for (int i = 0; i < y_true.size(); i++)
		total += y_true[i] * log(y_pred[i]) + (1 - y_true[i]) * log(1 - y_pred[i]);
	return -total / y_true.size();
}

double crossEntropy(const std::vector<double>& y_true,
										const std::vector<double>& y_pred) {
	double total = 0.;
	#pragma omp parallel for reduction(+:total)
	for (int i = 0; i < y_true.size(); i++)
		total += y_true[i] * log(y_pred[i]);
	return -total / y_true.size();
}

std::vector<double> readVec(const std::string& filename, const int& N) {
  std::vector<double> vec(N, 0);
	std::ifstream myFile(filename.c_str());
	if (myFile.is_open()) {
		std::string line;
		for (int i = 0; i < N; i++) {
			std::getline(myFile, line);
			vec[i] = std::stod(line);
		}
		myFile.close();
	} else {
		std::cerr << "CRITICAL: " << filename << " does not exist" << std::endl;
		exit(EXIT_FAILURE);
	}
	return vec;
}

std::vector<std::vector<double>> readMtx(const std::string& filename,
																				 const int& rows, const int& cols) {
	std::vector<std::vector<double>> mtx(rows, std::vector<double>(cols, 0));
	std::ifstream myFile(filename.c_str());
	if (myFile.is_open()) {
		std::string line;
		for (int i = 0; i < rows; i++) {
			std::getline(myFile, line);
			std::istringstream iss(line);
			for (int j = 0; j < cols; j++) iss >> mtx[i][j];
		}
		myFile.close();
	} else {
		std::cerr << "CRITICAL: " << filename << " does not exist" << std::endl;
		exit(EXIT_FAILURE);
	}
	return mtx;
}

void printVec(const std::string& filename, std::vector<double> vec,
							const int& N, const int& ax) {
	std::ofstream file_out;
	file_out.open(filename.c_str());
	if (ax == 0) { // Print as row
		for (int i = 0; i < N; i++) file_out << vec[i] << "\t";
		file_out << std::endl;
	} else if (ax == 1) { // Print as column
		for (int i = 0; i < N; i++) file_out << vec[i] << std::endl;
	}
	file_out.close();
}
void printVec(const std::string& filename, std::vector<double> vec,
							const int& N) {printVec(filename, vec, N, 0);}

void printMtx(const std::string& filename, const size_t& rows,
							const size_t& cols, const std::vector<std::vector<double>> mtx) {
	std::ofstream file_out;
	file_out.open(filename.c_str());
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) file_out << mtx[i][j] << "\t";
		file_out << std::endl;
	}
	file_out.close();
}

void set_act(const int& actnum, double (**act) (const double),
						 double (**D_act) (const double)) {
	std::cout << "Activation Function: ";
	switch (actnum) {
		case 0:
			*act = &tanh;
			*D_act = &D_tanh;
			std::cout << "tanh";
			break;
		case 1:
			*act = &relu;
			*D_act = &D_relu;
			std::cout << "relu";
			break;
		case 2:
			*act = &sigmoid;
			*D_act = &D_sigmoid;
			std::cout << "sigmoid";
			break;
		case 3:
			std::cerr << "CRITICAL: Missing parameter for leaky_relu" << std::endl;
			exit(EXIT_FAILURE);
		default:
			std::cerr << "CRITICAL: INVALID ACTIVATION FUNCTION" << std::endl;
			exit(EXIT_FAILURE);
	}
	std::cout << std::endl;
}
void set_act(const int& actnum, double (**act) (const double, const double),
						 double (**D_act) (const double, const double), const double& y) {
	std::cout << "Activation Function: ";
	switch (actnum) {
		case 3:
			*act = &leaky_relu;
			*D_act = &D_leaky_relu;
			std::cout << "leaky_relu";
			break;
		default:
			std::cerr << "CRITICAL: INVALID ACTIVATION FUNCTION" << std::endl;
			exit(EXIT_FAILURE);
	}
	std::cout << std::endl;
}
void set_act(const int& actnum,
						 std::vector<double> (**act) (const std::vector<double>),
						 std::vector<double> (**D_act) (const std::vector<double>)) {
	std::cout << "Activation Function: ";
	switch (actnum) {
		case 0:
			*act = &tanh;
			*D_act = &D_tanh;
			std::cout << "tanh";
			break;
		case 1:
			*act = &relu;
			*D_act = &D_relu;
			std::cout << "relu";
			break;
		case 2:
			*act = &sigmoid;
			*D_act = &D_sigmoid;
			std::cout << "sigmoid";
			break;
		case 3:
			std::cerr << "CRITICAL: Missing parameter for leaky_relu" << std::endl;
			exit(EXIT_FAILURE);
		case 4:
			std::cerr << "CRITICAL: Missing parameter for softmax" << std::endl;
			exit(EXIT_FAILURE);
		default:
			std::cerr << "CRITICAL: INVALID ACTIVATION FUNCTION" << std::endl;
			exit(EXIT_FAILURE);
	}
	std::cout << std::endl;
}
void set_act(const int& actnum,
						 std::vector<double> (**act) (const std::vector<double>,
						 															const double),
						 std::vector<double> (**D_act) (const std::vector<double>,
						 																const double), const double& y) {
	std::cout << "Activation Function: ";
	switch (actnum) {
		case 3:
			*act = &leaky_relu;
			*D_act = &D_leaky_relu;
			std::cout << "leaky_relu";
			break;
		case 4:
			*act = &softmax;
			*D_act = &D_softmax;
			std::cout << "softmax";
			break;
		default:
			std::cerr << "CRITICAL: INVALID ACTIVATION FUNCTION" << std::endl;
			exit(EXIT_FAILURE);
	}
	std::cout << std::endl;
}
std::string lastLine(const std::string& filepath) {
	std::ifstream fin;
  fin.open(filepath);
	std::string line;
	if (fin.is_open()) {
    fin.seekg(-1, std::ios_base::end);			// go to one spot before the EOF
		bool keepLooping = true;
    while(keepLooping) {
			char ch;
			fin.get(ch);                // Get current byte's data

			if ((int)fin.tellg() <= 1) {// If the data was at or before the 0th byte
				fin.seekg(0);                       // The first line is the last line
				keepLooping = false;                // So stop there
			} else if(ch == '\n') {                   // If the data was a newline
				keepLooping = false;                // Stop at the current position.
			} else {                                  // If the data was neither a newline nor at the 0 byte
				fin.seekg(-2, std::ios_base::cur);	// Move to the front of that data,
																						// then to the front of the data
																						// before it
			}
		}
		getline(fin, line);
		fin.close();
  } else {
		std::cerr << "Error: Unable to read file " << filepath << "\n";
    return "";
	}
	return line;
}

// TODO: Consider reference instead of value
std::map<std::string, std::string> paramMap(const std::string line,
               std::map<std::string, std::string> params) {
  // RegEx group         0,   1,      2,     3,   4,    5,      6 =
  //              original, key, double, float, int, bool, string
  std::regex expr("^# (\\w+)[\\t ]*" \
                  "(?:(-*0*\\.\\d+|\\de-*\\d+)|" \
                  "(-*\\d+\\.\\d+|\\d+\\.)|" \
                  "(-*\\d+)|" \
                  "(TRUE|FALSE)|" \
                  "((\\w|\\.| )*))$", std::regex_constants::icase);
  std::smatch matches;
  if (std::regex_search(line, matches, expr)) {
		std::map<std::string, std::string>::iterator it = params.find(matches[1]);
		if (it != params.end()) {
			params[matches[1]] = matches[std::stoi(params[matches[1]])];
		} else {
			#if DEBUG>= 1
			std::cout << "WARNING: Parameter in: \"" << line << "\" NOT set." \
								<< std::endl;
			#endif
		}
  } else {
    for (size_t i = 0; i < matches.size(); ++i) {
      std::cout << i << ": " << matches[i] << std::endl;
    }
  }
  return params;
}

void param2vec(const std::string& param, std::vector<int>& vec,
							 const size_t& N, const size_t& start, const size_t& offset) {
	std::istringstream iss(param);
	std::string line;
	size_t i = 0, j = offset;
	while (std::getline(iss, line, ' ') && i < N) {
		if (i >= start) {
			vec[j] = std::stod(line);
			j++;
		}
		i++;
	}
}
void param2vec(const std::string& param, std::vector<int>& vec,
							 const size_t& N, const size_t& start) {
	param2vec(param, vec, N, start, 0);
}
void param2vec(const std::string& param, std::vector<int>& vec,
							 const size_t& N) {param2vec(param, vec, N, 0, 0);}

void param2vec(const std::string& param, std::vector<double>& vec,
							 const size_t& N, const size_t& start, const size_t& offset) {
	std::istringstream iss(param);
	std::string line;
	size_t i = 0, j = offset;
	while (std::getline(iss, line, ' ') && i < N) {
		if (i >= start) {
			vec[j] = std::stod(line);
			j++;
		}
		i++;
	}
}
void param2vec(const std::string& param, std::vector<double>& vec,
							 const size_t& N, const size_t& start) {
	param2vec(param, vec, N, start, 0);
}
void param2vec(const std::string& param, std::vector<double>& vec,
							 const size_t& N) {param2vec(param, vec, N, 0, 0);}

size_t argMax(const std::vector<double> vec) {
	int i_max = 0;
	#pragma omp parallel for reduction(max:i_max)
	for (size_t i = 1; i < vec.size(); i++)
		i_max = (vec[i] > vec[i_max]) ? i : i_max;
	return i_max;
}
size_t argMax(const std::vector<double> vec, const size_t& N) {
	int i_max = 0;
	#pragma omp parallel for reduction(max:i_max)
	for (size_t i = 1; i < N; i++)
		i_max = (vec[i] > vec[i_max]) ? i : i_max;
	return i_max;
}