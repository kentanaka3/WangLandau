#include "utils.hpp"

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

void printMtx(const std::string& filename, const int& rows, const int& cols,
							std::vector<std::vector<double>> mtx) {
	std::ofstream file_out;
	file_out.open(filename.c_str());
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) file_out << mtx[i][j] << "\t";
		file_out << std::endl;
	}
	file_out.close();
}

double relu(const double x) {
	return (x >= 0.) ? x : 0.;
}

template <typename T>
int sgn(const T& val) {
	return (T(0) < val) - (val <= T(0));
}

void set_act(const int& actnum, double (**act) (double)) {
	std::cout << "Activation Function: ";
  switch (actnum) {
    case 0:
			*act = &tanh;
			std::cout << "tanh";
			break;
    case 1:
			*act = &relu;
			std::cout << "relu";
			break;
    default:
			std::cerr << "CRITICAL: INVALID ACTIVATION FUNCTION" << std::endl;
			exit(EXIT_FAILURE);
  }
	std::cout << std::endl;
}

// TODO: Consider reference instead of value
std::map<std::string, std::string> readParam(const std::string line,
               std::map<std::string, std::string> params) {
  // RegEx group         0,   1,      2,     3,   4,    5,      6 =
  //              original, key, double, float, int, bool, string
  std::regex expr("^# (\\w+)[\\t ]*" \
                  "(?:(0*\\.\\d+|\\de-*\\d+)|" \
                  "(\\d+\\.\\d+|\\d+\\.)|" \
                  "(-*\\d+)|" \
                  "(TRUE|FALSE)|" \
                  "((\\w|\\.| )*))$", std::regex_constants::icase);
  std::smatch matches;
  if (std::regex_search(line, matches, expr)) {
		std::map<std::string, std::string>::iterator it = params.find(matches[1]);
		if (it != params.end()) {
			params[matches[1]] = matches[std::stoi(params[matches[1]])];
		} else {
			#if DEBUG_UTIL >= 1
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

void param2vec(const std::string& param, std::vector<int>& vec, const int& N) {
	std::istringstream iss(param);
	std::string line;
	int i = 0;
	while (std::getline(iss, line, ' ') && i < N) {
		vec[i] = std::stoi(line);
		i++;
	}
}

void param2vec(const std::string& param, std::vector<double>& vec,
							 const int& N) {
	std::istringstream iss(param);
	std::string line;
	size_t i = 0;
	while (std::getline(iss, line, ' ') && i < N) {
		vec[i] = std::stod(line);
		i++;
	}
}