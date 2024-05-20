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

double accuracy(const std::vector<double>& y_true,
								const std::vector<double>& y_pred) {
	if (y_true.size() != y_pred.size()) {
		std::cerr << "Error: Size mismatch between true and predicted values.\n";
		exit(EXIT_FAILURE);
	}
	double total = 0.;
	#pragma omp parallel for reduction(+:total)
	for (size_t i = 0; i < y_true.size(); ++i)
		total += (y_true[i] == y_pred[i]);
	return total / static_cast<double>(y_true.size());
}

std::vector<double> stochasticGradientDescent(
	const std::vector<double>& gradient, const double& learning_rate) {
	std::vector<double> update(gradient.size(), 0.);
	#pragma omp parallel for
	for (int i = 0; i < gradient.size(); i++)
		update[i] = -learning_rate * gradient[i];
	return update;
}
std::vector<double> stochasticGradientDescent(
	const std::vector<double>& gradient, const double& learning_rate,
	const double& momentum, std::vector<double>& weights) {
	std::vector<double> update = stochasticGradientDescent(gradient,
																												 learning_rate);
	#pragma omp parallel for
	for (int i = 0; i < gradient.size(); i++) update[i] -= momentum * weights[i];
	return update;
}

std::vector<double> momentum(const std::vector<double>& gradient,
	const double& learning_rate, const double& b, const std::vector<double>& v) {
	std::vector<double> update(gradient.size(), 0.);
	#pragma omp parallel for
	for (int i = 0; i < gradient.size(); i++) {
		update[i] = b * v[i] - learning_rate * gradient[i];
	}
	return update;
}
std::vector<double> momentum(const std::vector<double>& gradient,
	const double& learning_rate, const double& b, const std::vector<double>& v,
	const double& decay, std::vector<double>& weights) {
	std::vector<double> update = momentum(gradient, learning_rate, b, v);
	#pragma omp parallel for
	for (int i = 0; i < gradient.size(); i++) update[i] -= decay * weights[i];
	return update;
}

std::vector<double> momentumNesterov(const std::vector<double>& gradient,
	const double& learning_rate, const double& beta, std::vector<double>& v) {
	std::vector<double> update(gradient.size(), 0.);
	#pragma omp parallel for
	for (int i = 0; i < gradient.size(); i++) {
		v[i] = beta * v[i] - learning_rate * gradient[i];
		update[i] = -beta * v[i] + (1 + beta) * v[i];
	}
	return update;
}
std::vector<double> momentumNesterov(
	const std::vector<double>& gradient, const double& learning_rate,
	const double& beta, std::vector<double>& v, const double& momentum,
	std::vector<double>& weights) {
	std::vector<double> update(gradient.size(), 0.);
	#pragma omp parallel for
	for (int i = 0; i < gradient.size(); i++) {
		v[i] = beta * v[i] - learning_rate * gradient[i] - momentum * weights[i];
		update[i] = -beta * v[i] + (1 + beta) * v[i];
	}
	return update;
}

std::vector<double> RMSprop(const std::vector<double>& gradient,
	const double& learning_rate, const double& beta, std::vector<double>& cache){
	std::vector<double> update(gradient.size(), 0.);
	#pragma omp parallel for
	for (int i = 0; i < gradient.size(); i++) {
		cache[i] = beta * cache[i] + (1 - beta) * gradient[i] * gradient[i];
		update[i] = -learning_rate * gradient[i] / (std::sqrt(cache[i]) + EPS);
	}
	return update;
}
std::vector<double> RMSprop(const std::vector<double>& gradient,
	const double& learning_rate,const double& beta, std::vector<double>& cache,
	const double& momentum, std::vector<double>& weights) {
	std::vector<double> update = RMSprop(gradient, learning_rate, beta, cache);
	#pragma omp parallel for
	for (int i = 0; i < gradient.size(); i++) update[i] -= momentum * weights[i];
	return update;
}

std::vector<double> RMSpropGraves(const std::vector<double>& gradient,
	const double& learning_rate, const double& beta, std::vector<double>& cache){
	std::vector<double> update(gradient.size(), 0.);
	#pragma omp parallel for
	for (int i = 0; i < gradient.size(); i++) {
		cache[i] = beta * cache[i] + (1 - beta) * gradient[i] * gradient[i];
		update[i] = -learning_rate * gradient[i] / (std::sqrt(cache[i]) + EPS);
	}
	return update;
}
std::vector<double> RMSpropGraves(const std::vector<double>& gradient,
	const double& learning_rate, const double& beta, std::vector<double>& cache,
	const double& momentum, std::vector<double>& weights) {
	std::vector<double> update = RMSpropGraves(gradient, learning_rate, beta, \
																						 cache);
	#pragma omp parallel for
	for (int i = 0; i < gradient.size(); i++) update[i] -= momentum * weights[i];
	return update;
}

std::vector<double> Adam(const std::vector<double>& gradient,
	const double& learning_rate, const double& beta1, const double& beta2,
	std::vector<double>& m, std::vector<double>& v, const int& t) {
	std::vector<double> update(gradient.size(), 0.);
	#pragma omp parallel for
	for (int i = 0; i < gradient.size(); i++) {
		m[i] = beta1 * m[i] + (1 - beta1) * gradient[i];
		v[i] = beta2 * v[i] + (1 - beta2) * gradient[i] * gradient[i];
		update[i] = (-learning_rate * m[i] / (1 - std::pow(beta1, t))) / (std::sqrt(v[i] / (1 - std::pow(beta2, t))) + EPS);
	}
	return update;
}
std::vector<double> Adam(const std::vector<double>& gradient,
	const double& learning_rate, const double& beta1, const double& beta2,
	std::vector<double>& m, std::vector<double>& v, const int& t,
	const double& momentum, std::vector<double>& weights) {
	std::vector<double> update = Adam(gradient, learning_rate, beta1, beta2, \
																		m, v, t);
	#pragma omp parallel for
	for (int i = 0; i < gradient.size(); i++) update[i] -= momentum * weights[i];
	return update;
}

std::vector<double> AdaMax(const std::vector<double>& gradient,
	const double& learning_rate, const double& beta1, const double& beta2,
	std::vector<double>& m, std::vector<double>& u, const int& t) {
	std::vector<double> update(gradient.size(), 0.);
	#pragma omp parallel for
	for (int i = 0; i < gradient.size(); i++) {
		m[i] = beta1 * m[i] + (1 - beta1) * gradient[i];
		u[i] = std::max(beta2 * u[i], std::abs(gradient[i]));
		update[i] = -learning_rate / (1 - std::pow(beta1, t)) * m[i] / (u[i] + EPS);
	}
	return update;
}
std::vector<double> AdaMax(const std::vector<double>& gradient,
	const double& learning_rate, const double& beta1, const double& beta2,
	std::vector<double>& m, std::vector<double>& u, const int& t,
	const double& momentum, std::vector<double>& weights) {
	std::vector<double> update = AdaMax(gradient, learning_rate, beta1, beta2, \
																			m, u, t);
	#pragma omp parallel for
	for (int i = 0; i < gradient.size(); i++) update[i] -= momentum * weights[i];
	return update;
}

std::vector<double> nAdam(const std::vector<double>& gradient,
	const double& learning_rate, const double& beta1, const double& beta2,
	std::vector<double>& m, std::vector<double>& v, const int& t) {
	std::vector<double> update(gradient.size(), 0.);
	#pragma omp parallel for
	for (int i = 0; i < gradient.size(); i++) {
		m[i] = beta1 * m[i] + (1 - beta1) * gradient[i];
		v[i] = beta2 * v[i] + (1 - beta2) * gradient[i] * gradient[i];
		double m_hat = m[i] / (1 - std::pow(beta1, t));
		double v_hat = v[i] / (1 - std::pow(beta2, t));
		update[i] = -learning_rate * (beta1 * m_hat + (1 - beta1) * gradient[i]) /
								(std::sqrt(v_hat) + EPS);
	}
	return update;
}
std::vector<double> nAdam(const std::vector<double>& gradient,
	const double& learning_rate, const double& beta1, const double& beta2,
	std::vector<double>& m, std::vector<double>& v, const int& t,
	const double& momentum, std::vector<double>& weights) {
	std::vector<double> update = nAdam(gradient, learning_rate, beta1, beta2, \
																			m, v, t);
	#pragma omp parallel for
	for (int i = 0; i < gradient.size(); i++) update[i] -= momentum * weights[i];
	return update;
}

std::vector<double> AdaGrad(const std::vector<double>& gradient,
	const double& learning_rate, std::vector<double>& cache) {
	std::vector<double> update(gradient.size(), 0.);
	#pragma omp parallel for
	for (int i = 0; i < gradient.size(); i++) {
		cache[i] += gradient[i] * gradient[i];
		update[i] = -learning_rate * gradient[i] / (std::sqrt(cache[i]) + EPS);
	}
	return update;
}
std::vector<double> AdaGrad(const std::vector<double>& gradient,
	const double& learning_rate, std::vector<double>& cache,
	const double& momentum, std::vector<double>& weights) {
	std::vector<double> update = AdaGrad(gradient, learning_rate, cache);
	#pragma omp parallel for
	for (int i = 0; i < gradient.size(); i++) update[i] -= momentum * weights[i];
	return update;
}

std::vector<double> AdaDelta(const std::vector<double>& gradient,
	const double& learning_rate, const double& beta, std::vector<double>& cache,
	std::vector<double>& update) {
	std::vector<double> new_cache(gradient.size(), 0.);
	#pragma omp parallel for
	for (int i = 0; i < gradient.size(); i++) {
		cache[i] = beta * cache[i] + (1 - beta) * gradient[i] * gradient[i];
		update[i] = -std::sqrt(new_cache[i] + EPS) / std::sqrt(cache[i] + EPS) * gradient[i];
		new_cache[i] = beta * new_cache[i] + (1 - beta) * update[i] * update[i];
	}
	return update;
}
std::vector<double> AdaDelta(const std::vector<double>& gradient,
	const double& learning_rate, const double& beta, std::vector<double>& cache,
	std::vector<double>& update,
	const double& momentum, std::vector<double>& weights) {
	std::vector<double> new_cache(gradient.size(), 0.);
	#pragma omp parallel for
	for (int i = 0; i < gradient.size(); i++) {
		cache[i] = beta * cache[i] + (1 - beta) * gradient[i] * gradient[i];
		update[i] = -std::sqrt(new_cache[i] + EPS) / std::sqrt(cache[i] + EPS) * gradient[i] - momentum * weights[i];
		new_cache[i] = beta * new_cache[i] + (1 - beta) * update[i] * update[i];
	}
	return update;
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

void printVec(const std::string& filename, const std::vector<double>& vec,
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
void printVec(const std::string& filename, const std::vector<double>& vec,
							const int& N) {printVec(filename, vec, N, 0);}
void printVec(const std::string& filename, const std::vector<double>& vec) {
	printVec(filename, vec, vec.size(), 0);
}
void printVec(const std::vector<double>& vec, const int& N, const int& ax) {
	if (ax == 0) { // Print as row
		for (int i = 0; i < N; i++) std::cout << vec[i] << "\t";
		std::cout << std::endl;
	} else if (ax == 1) { // Print as column
		for (int i = 0; i < N; i++) std::cout << vec[i] << std::endl;
	}
}
void printVec(const std::vector<double>& vec, const int& N) {
	printVec(vec, N, 0);
}
void printVec(const std::vector<double>& vec) {printVec(vec, vec.size(), 0);}

void printMtx(const std::string& filename, const size_t& rows,
							const size_t& cols, const std::vector<std::vector<double>>& mtx){
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

// std::map<std::string, TokenType> KEYWORDS_ML = {
// 	{"∧", TokenType::AND}, 			// Logical and
// 	{",", TokenType::COMMA},
// 	{"\\", TokenType::CONTROL},
// 	{"÷", TokenType::DIVISION}, // Division
// 	{"}", TokenType::END},			// End C++
// 	{"=", TokenType::EQUAL}, 		// Equal
// 	{".", TokenType::DOT},			// Dot
// 	{"¿", TokenType::ELSE},			// Else
// 	{"\n", TokenType::END},			// End of line
// 	{"\0", TokenType::EOF_},		// End of file
// 	{">", TokenType::GREATER},	// Greater than
// 	{"?", TokenType::IF},				// If
// 	{"(", TokenType::LEFT_PAREN},
// 	{"<", TokenType::LESS},			// Less than
// 	{"-", TokenType::MINUS},		// Minus
// 	{"¬", TokenType::NOT},			// Logical not
// 	{"∂", TokenType::OPERATOR}, // Partial Derivative
// 	{"\'", TokenType::OPERATOR}, // First Exact Derivative
// 	{"\"", TokenType::OPERATOR}, // Second Exact Derivative
// 	{"∇", TokenType::OPERATOR}, // Gradient
// 	{"∑", TokenType::OPERATOR}, // Discrete Sum
// 	{"∏", TokenType::OPERATOR}, // Product
// 	{"×", TokenType::OPERATOR}, // Multiplication
// 	{"!", TokenType::OPERATOR}, // Factorial
// 	{"⋅", TokenType::OPERATOR}, // Dot product
// 	{"○", TokenType::OPERATOR}, // O operator
// 	{"∙", TokenType::OPERATOR}, // Dot operator
// 	{"√", TokenType::OPERATOR}, // Square root
// 	{"∛", TokenType::OPERATOR}, // Cube root
// 	{"∜", TokenType::OPERATOR}, // Fourth root
// 	{"∫", TokenType::OPERATOR}, // Integral
// 	{"∬", TokenType::OPERATOR}, // Double integral
// 	{"∭", TokenType::OPERATOR}, // Triple integral
// 	{"∮", TokenType::OPERATOR}, // Contour integral
// 	{"∯", TokenType::OPERATOR}, // Surface integral
// 	{"∰", TokenType::OPERATOR}, // Volume integral
// 	{"∴", TokenType::OPERATOR}, // Therefore
// 	{"∵", TokenType::OPERATOR}, // Because
// 	{"≠", TokenType::OPERATOR}, // Not equal
// 	{"≡", TokenType::OPERATOR}, // Identical
// 	{"≢", TokenType::OPERATOR}, // Not identical
// 	{"≣", TokenType::OPERATOR}, // Not identical
// 	{"≤", TokenType::OPERATOR}, // Less than or equal
// 	{"≥", TokenType::OPERATOR}, // Greater than or equal
// 	{"≦", TokenType::OPERATOR}, // Less than or equal
// 	{"≧", TokenType::OPERATOR}, // Greater than or equal
// 	{"≨", TokenType::OPERATOR}, // Less than or equivalent
// 	{"≩", TokenType::OPERATOR}, // Greater than or equivalent
// 	{"≪", TokenType::OPERATOR}, // Much less than
// 	{"≫", TokenType::OPERATOR}, // Much greater than
// 	{"≬", TokenType::OPERATOR}, // Between
// 	{"≭", TokenType::OPERATOR}, // Not between
// 	{"≮", TokenType::OPERATOR}, // Not less than
// 	{"≯", TokenType::OPERATOR}, // Not greater than
// 	{"≰", TokenType::OPERATOR}, // Neither less than nor equal
// 	{"≱", TokenType::OPERATOR}, // Neither greater than nor equal
// 	{"≲", TokenType::OPERATOR}, // Less than or equivalent
// 	{"≳", TokenType::OPERATOR}, // Greater than or equivalent
// 	{"≴", TokenType::OPERATOR}, // Less than or greater than
// 	{"≵", TokenType::OPERATOR}, // Greater than or less than
// 	{"≶", TokenType::OPERATOR}, // Less than or equivalent
// 	{"≷", TokenType::OPERATOR}, // Greater than or equivalent
// 	{"≸", TokenType::OPERATOR}, // Less than or greater than
// 	{"≹", TokenType::OPERATOR}, // Greater than or less than
// 	{"≺", TokenType::OPERATOR}, // Precedes
// 	{"≻", TokenType::OPERATOR}, // Succeeds
// 	{"≼", TokenType::OPERATOR}, // Precedes or equal
// 	{"≽", TokenType::OPERATOR}, // Succeeds or equal
// 	{"≾", TokenType::OPERATOR}, // Precedes or equivalent
// 	{"≿", TokenType::OPERATOR}, // Succeeds or equivalent
// 	{"⊀", TokenType::OPERATOR}, // Does not precede
// 	{"⊁", TokenType::OPERATOR}, // Does not succeed
// 	{"⊂", TokenType::OPERATOR}, // Subset
// 	{"⊃", TokenType::OPERATOR}, // Superset
// 	{"⊄", TokenType::OPERATOR}, // Not a subset
// 	{"⊅", TokenType::OPERATOR}, // Not a superset
// 	{"⊆", TokenType::OPERATOR}, // Subset or equal
// 	{"⊇", TokenType::OPERATOR}, // Superset or equal
// 	{"⊈", TokenType::OPERATOR}, // Neither subset nor equal
// 	{"⊉", TokenType::OPERATOR}, // Neither superset nor equal
// 	{"⊊", TokenType::OPERATOR}, // Subset but not equal
// 	{"⊋", TokenType::OPERATOR}, // Superset but not equal
// 	{"⊌", TokenType::OPERATOR}, // Neither subset nor superset
// 	{"⊍", TokenType::OPERATOR}, // Subset of or equal to
// 	{"⊎", TokenType::OPERATOR}, // Superset of or equal to
// 	{"⊏", TokenType::OPERATOR}, // Neither subset nor equal
// 	{"⊐", TokenType::OPERATOR}, // Neither superset nor equal
// 	{"⊑", TokenType::OPERATOR}, // Subset but not equal
// 	{"⊒", TokenType::OPERATOR}, // Superset but not equal
// 	{"⊓", TokenType::OPERATOR}, // Neither subset nor superset
// 	{"⊔", TokenType::OPERATOR}, // Subset of or equal to
// 	{"⊖", TokenType::OPERATOR}, //
// 	{"*", TokenType::OPERATOR}, // Katri-Rao product
// 	{"⊗", TokenType::OPERATOR}, // Tensor product
// 	{"⊘", TokenType::OPERATOR}, // Division sign
// 	{"⊙", TokenType::OPERATOR}, // Hadamard product
// 	{"⊚", TokenType::OPERATOR}, //
// 	{"⊛", TokenType::OPERATOR}, //
// 	{"⊜", TokenType::OPERATOR}, //
// 	{"⊝", TokenType::OPERATOR}, //
// 	{"⊞", TokenType::OPERATOR}, //
// 	{"⊟", TokenType::OPERATOR}, //
// 	{"⊠", TokenType::OPERATOR}, //
// 	{"⊡", TokenType::OPERATOR}, //
// 	{"⊬", TokenType::OPERATOR}, // Does not prove
// 	{"⊰", TokenType::OPERATOR}, // Precedes under relation
// 	{"⊱", TokenType::OPERATOR}, // Succeeds under relation
// 	{"⊲", TokenType::OPERATOR}, // Normal subgroup of
// 	{"⊳", TokenType::OPERATOR}, // Contains as normal subgroup
// 	{"⊴", TokenType::OPERATOR}, // Normal subgroup of or equal to
// 	{"⊵", TokenType::OPERATOR}, // Contains as normal subgroup or equal
// 	{"⊹", TokenType::OPERATOR}, // Hermitian conjugate matrix
// 	{"⊻", TokenType::OPERATOR}, // Xor
// 	{"⊼", TokenType::OPERATOR}, // Nand
// 	{"⊽", TokenType::OPERATOR}, // Nor
// 	{"⋀", TokenType::OPERATOR}, // N-ary logical and
// 	{"⋁", TokenType::OPERATOR}, // N-ary logical or
// 	{"⋂", TokenType::OPERATOR}, // N-ary intersection
// 	{"⋃", TokenType::OPERATOR}, // N-ary union
// 	{"⋄", TokenType::OPERATOR}, // Diamond operator
// 	{"⋅", TokenType::OPERATOR}, // Dot operator
// 	{"⋋", TokenType::OPERATOR}, // Left semidirect product
// 	{"⋌", TokenType::OPERATOR}, // Right semidirect product
// 	{"⋒", TokenType::OPERATOR}, // Double intersection
// 	{"⋓", TokenType::OPERATOR}, // Double union
// 	{"⋕", TokenType::OPERATOR}, // Equal and parallel to
// 	{"⋚", TokenType::OPERATOR}, // Less-than equal to or greater-than
// 	{"⋛", TokenType::OPERATOR}, // Greater-than equal to or less-than
// 	{"⋜", TokenType::OPERATOR}, // Equal to or less-than
// 	{"⋝", TokenType::OPERATOR}, // Equal to or greater-than
// 	{"⋞", TokenType::OPERATOR}, // Equal to or precedes
// 	{"⋟", TokenType::OPERATOR}, // Equal to or succeeds
// 	{"⋠", TokenType::OPERATOR}, // Not equal to
// 	{"⋡", TokenType::OPERATOR}, // Not less-than
// 	{"⋢", TokenType::OPERATOR}, // Not greater-than
// 	{"⋣", TokenType::OPERATOR}, // Neither less-than nor equal to
// 	{"⋤", TokenType::OPERATOR}, // Neither greater-than nor equal to
// 	{"⋥", TokenType::OPERATOR}, // Neither less-than nor greater-than
// 	{"⋦", TokenType::OPERATOR}, // Precedes or equal to
// 	{"⋧", TokenType::OPERATOR}, // Succeeds or equal to
// 	{"⋨", TokenType::OPERATOR}, // Precedes or equivalent to
// 	{"⋩", TokenType::OPERATOR}, // Succeeds or equivalent to
// 	{"⌅", TokenType::OPERATOR},	// Projective
// 	{"⌆", TokenType::OPERATOR},	// Perspective
// 	{"⊕", TokenType::OPERATOR}, // Direct sum
// 	{"∨", TokenType::OR}, 			// Logical or
// 	{"+", TokenType::PLUS}, 		// Plus
// 	{")", TokenType::RIGHT_PAREN},
// 	{"√", TokenType::SQRT},			// Square root
// 	{"{", TokenType::START},		// Start C++
// 	{"[", TokenType::TENSOR},		// Tensor
// 	{"]", TokenType::TENSOR},		// Tensor
// 	{"∴", TokenType::THEN},			// Therefore
// 	{"~", TokenType::WHILE},		// While
// 	{"ⲁ", TokenType::VARIABLE},	// Alpha
// 	{"ⲃ", TokenType::VARIABLE},	// Beta
// 	{"ⲅ", TokenType::VARIABLE},	// Gamma
// 	{"ⲇ", TokenType::VARIABLE},	// Delta
// 	{"∆", TokenType::VARIABLE}, // Delta
// 	{"ⲉ", TokenType::VARIABLE},	// Epsilon
// 	{"ⲋ", TokenType::VARIABLE},	// Zeta
// 	{"ⲍ", TokenType::VARIABLE},	// Eta
// 	{"ⲏ", TokenType::VARIABLE},
// 	{"ⲑ", TokenType::VARIABLE},	// Theta
// 	{"ⲓ", TokenType::VARIABLE},	// Iota
// 	{"ⲕ", TokenType::VARIABLE},	// Kappa
// 	{"ⲗ", TokenType::VARIABLE},	// Lambda
// 	{"ⲙ", TokenType::VARIABLE},	// Mu
// 	{"ⲛ", TokenType::VARIABLE},	// Nu
// 	{"ⲝ", TokenType::VARIABLE},	// Xi (Noise)
// 	{"ⲟ", TokenType::VARIABLE},	// Omicron
// 	{"ⲡ", TokenType::VARIABLE},	// Pi
// 	{"ⲣ", TokenType::VARIABLE},	// Rho
// 	{"ⲥ", TokenType::VARIABLE},	// Sigma
// 	{"ⲧ", TokenType::VARIABLE},	// Tau
// 	{"ⲩ", TokenType::VARIABLE},	// Upsilon
// 	{"ⲫ", TokenType::VARIABLE},	// Phi
// 	{"ⲭ", TokenType::VARIABLE},	// Chi
// 	{"ⲯ", TokenType::VARIABLE},	// Psi
// 	{"ⲱ", TokenType::VARIABLE},	// Omega
// 	// {"#", TokenType::FALSE},
// 	// {"$", TokenType::TRUE},
// 	// {, enType::BOOLEAN},
// 	// {, enType::INTEGER},
// 	// {, enType::COMPLEX},
// 	// {, enType::FLOAT},
// 	// {, enType::DOUBLE},
// 	// {, enType::LONG},
// 	// {, enType::SHORT},
// 	// {, enType::CHAR},
// 	// {, enType::STRING},
// 	// {, enType::ARRAY},
// 	// {, enType::VECTOR},
// 	// {, enType::MATRIX},
// 	// {, enType::TENSOR},
// 	// {, enType::LIST},
// 	// {, enType::SET},
// 	// {, enType::MAP},
// 	// {, enType::GRAPH},
// 	// {, enType::TREE},
// 	// {, enType::STACK},
// 	// {, enType::QUEUE},
// 	// {, enType::DEQUE},
// 	// {, enType::PRIORITY_QUEUE},
// 	// {, enType::LINKED_LIST},
// 	// {, enType::DOUBLY_LINKED_LIST},
// 	// {, enType::CIRCULAR_LINKED_LIST},
// 	// {, enType::CIRCULAR_DOUBLY_LINKED_LIST},
// 	// {, enType::BINARY_TREE},
// 	// {, enType::BINARY_SEARCH_TREE},
// 	// {, enType::AVL_TREE},
// 	// {, enType::RED_BLACK_TREE},
// 	// {, enType::HEAP},
// 	// {, enType::HASH_TABLE},
// 	// {, enType::REVERSE},
// 	// {, enType::ASSIGNMENT},

// 	// {"@", TokenType::}, // At sign
// 	// {"&", TokenType::}, // Ampersand
// 	// {"|", TokenType::}, // Vertical bar
// 	// {"_", TokenType::UNDEFINED},
// 	// {":", TokenType::}, // Colon
// 	// {"\"",  TokenType::}, // Single quote
// 	// {"/", TokenType::}, // Division
// 	// {";", TokenType::}, // Semicolon
// 	// {"°", TokenType::}, // Degree
// 	// {"∞", TokenType::}, // Infinity
// 	// {"∅", TokenType::}, // Empty set
// 	// {"∀", TokenType::}, // For all
// 	// {"∃", TokenType::}, // There exists
// 	// {"∄", TokenType::}, // There does not exist
// 	// {"§", TokenType::}, // Section
// 	// {"¶", TokenType::}, // Paragraph
// 	// {"±", TokenType::}, // Plus or minus
// 	// {"∞", TokenType::}, // Infinity
// 	// {"%", TokenType::}, // Percent
// 	// {"∫", TokenType::}, // Integral
// 	// {"≈", TokenType::}, // Approximately equal
// 	// {"≠", TokenType::}, // Not equal
// 	// {"≤", TokenType::}, // Less than or equal
// 	// {"≥", TokenType::}, // Greater than or equal
// 	// {"∈", TokenType::}, // Element of
// 	// {"∉", TokenType::}, // Not an element of
// 	// {"∋", TokenType::}, // Contains as member
// 	// {"∌", TokenType::}, // Does not contain as member
// 	// {"∠", TokenType::}, // Angle
// 	// {"∩", TokenType::}, // Intersection
// 	// {"∪", TokenType::}, // Union
// 	// {"∼", TokenType::}, // Tilde
// 	// {"≅", TokenType::}, // Congruent to
// 	// {"≡", TokenType::}, // Identical to
// 	// {"≢", TokenType::}, // Not identical to
// 	// {"⊂", TokenType::}, // Subset of
// 	// {"⊃", TokenType::}, // Superset of
// 	// {"⊄", TokenType::}, // Not a subset of
// 	// {"⊆", TokenType::}, // Subset of or equal to
// 	// {"⊇", TokenType::}, // Superset of or equal to
// 	// {"⊥", TokenType::}, // Perpendicular
// 	// {"⌈", TokenType::}, // Left Ceiling
// 	// {"⌉", TokenType::}, // Right Ceiling
// 	// {"⌊", TokenType::}, // Left Floor
// 	// {"⌋", TokenType::}, // Right Floor
// 	// {"□", TokenType::}, //
// 	// {"◦", TokenType::}, // Degree
// 	// {"⟨", TokenType::}, // Bra
// 	// {"⟩", TokenType::}, // Ket
// };
// std::map<TokenType, std::string> KEYWORDS_ML_R = {
// 	{TokenType::AND, "∧"}, 			// Logical and
// 	{TokenType::COMMA, ","},
// 	{TokenType::CONTROL, "\\"},
// 	{TokenType::DIVISION, "÷"}, // Division
// 	{TokenType::END, "}"},			// End C++
// 	{TokenType::EQUAL, "="}, 		// Equal
// 	{TokenType::DOT, "."},			// Dot
// 	{TokenType::ELSE, "¿"},			// Else
// 	{TokenType::END, "\n"},		  // End of line
// 	{TokenType::EOF_, "\0"},		// End of file
// 	{TokenType::GREATER, ">"},	// Greater than
// 	{TokenType::IF, "?"},				// If
// 	{TokenType::LEFT_PAREN, "("},
// 	{TokenType::LESS, "<"},			// Less than
// 	{TokenType::MINUS, "-"},		// Minus
// 	{TokenType::NOT, "¬"},			// Logical not
// 	{TokenType::OR, "∨"}, 			// Logical or
// 	{TokenType::PLUS, "+"}, 		// Plus
// 	{TokenType::RIGHT_PAREN, ")"},
// 	{TokenType::SQRT, "√"},			// Square root
// 	{TokenType::START, "{"},		// Start C++
// 	{TokenType::TENSOR, "["},		// Tensor
// 	{TokenType::TENSOR, "]"},		// Tensor
// 	{TokenType::THEN, "∴"},			// Therefore
// 	{TokenType::WHILE, "~"},		// While
// 	//{TokenType::VARIABLE, ""}, // Omega
// };
