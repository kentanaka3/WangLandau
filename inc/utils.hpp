/*
 * utils.hpp
 * Created on: 2024. 1. 1.
 * Author: @kentanaka3
 * Summary: Utility functions for neural network
 *
 */
#ifndef UTILS_HPP
#define UTILS_HPP
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

#define EPS 1e-8

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

enum class Activation {LINEAR, TANH, SIGMOID, RELU, LEAKY_RELU, SOFTMAX};

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

enum class Loss {MSE, BCE, CE};

double meanSquaredError(const std::vector<double>& y_true,
												const std::vector<double>& y_pred);

double binaryCrossEntropy(const std::vector<double>& y_true,
													const std::vector<double>& y_pred);

double crossEntropy(const std::vector<double>& y_true,
										const std::vector<double>& y_pred);


enum class Optimizer {SGD, Momentum, MomentumNesterov, RMSprop, RMSpropGraves,
	Adam, AdaMax, nAdam, AdaGrad, AdaDelta,
};

std::vector<double> stochasticGradientDescent(
	const std::vector<double>& gradient, const double& learning_rate);
std::vector<double> stochasticGradientDescent(
	const std::vector<double>& gradient, const double& learning_rate,
	const double& momentum, std::vector<double>& weights);

std::vector<double> momentum(const std::vector<double>& gradient,
	const double& learning_rate, const double& b, const std::vector<double>& v);
std::vector<double> momentum(const std::vector<double>& gradient,
	const double& learning_rate, const double& b, const std::vector<double>& v,
	const double& decay, std::vector<double>& weights);

std::vector<double> momentumNesterov(const std::vector<double>& gradient,
	const double& learning_rate, const double& beta, std::vector<double>& v);
std::vector<double> momentumNesterov(const std::vector<double>& gradient,
	const double& learning_rate, const double& beta, std::vector<double>& v,
	const double& momentum, std::vector<double>& weights);

std::vector<double> RMSprop(const std::vector<double>& gradient,
	const double& learning_rate, const double& beta, std::vector<double>& cache);
std::vector<double> RMSprop(const std::vector<double>& gradient,
	const double& learning_rate, const double& beta, std::vector<double>& cache,
	const double& momentum, std::vector<double>& weights);

std::vector<double> RMSpropGraves(const std::vector<double>& gradient,
	const double& learning_rate, const double& beta, std::vector<double>& cache);
std::vector<double> RMSpropGraves(const std::vector<double>& gradient,
	const double& learning_rate, const double& beta, std::vector<double>& cache,
	const double& momentum, std::vector<double>& weights);

std::vector<double> Adam(const std::vector<double>& gradient,
	const double& learning_rate, const double& beta1, const double& beta2,
	std::vector<double>& m, std::vector<double>& v, const int& t);
std::vector<double> Adam(const std::vector<double>& gradient,
	const double& learning_rate, const double& beta1, const double& beta2,
	std::vector<double>& m, std::vector<double>& v, const int& t,
	const double& momentum, std::vector<double>& weights);

std::vector<double> AdaMax(const std::vector<double>& gradient,
	const double& learning_rate, const double& beta1, const double& beta2,
	std::vector<double>& m, std::vector<double>& u, const int& t);
std::vector<double> AdaMax(const std::vector<double>& gradient,
	const double& learning_rate, const double& beta1, const double& beta2,
	std::vector<double>& m, std::vector<double>& u, const int& t,
	const double& momentum, std::vector<double>& weights);

std::vector<double> nAdam(const std::vector<double>& gradient,
	const double& learning_rate, const double& beta1, const double& beta2,
	std::vector<double>& m, std::vector<double>& v, const int& t);
std::vector<double> nAdam(const std::vector<double>& gradient,
	const double& learning_rate, const double& beta1, const double& beta2,
	std::vector<double>& m, std::vector<double>& v, const int& t,
	const double& momentum, std::vector<double>& weights);

std::vector<double> AdaGrad(const std::vector<double>& gradient,
	const double& learning_rate, std::vector<double>& cache);
std::vector<double> AdaGrad(const std::vector<double>& gradient,
	const double& learning_rate, std::vector<double>& cache,
	const double& momentum, std::vector<double>& weights);

std::vector<double> AdaDelta(const std::vector<double>& gradient,
	const double& learning_rate, const double& beta, std::vector<double>& cache,
	std::vector<double>& update);
std::vector<double> AdaDelta(const std::vector<double>& gradient,
	const double& learning_rate, const double& beta, std::vector<double>& cache,
	std::vector<double>& update,
	const double& momentum, std::vector<double>& weights);

enum class LayerType {DENSE, CONV2D, POOL2D, FLATTEN, ACTIVATION, DROPOUT};
enum class PoolType {MAX, AVG};
enum class Padding {SAME, VALID};
enum class Stride {STRIDE, NO_STRIDE};

std::vector<double> readVec(const std::string& filename, const int& N);

std::vector<std::vector<double>> readMtx(const std::string& filename,
																				 const int& rows, const int& cols);

void printVec(const std::string& filename, const std::vector<double>& vec,
              const int& N, const int& ax);
void printVec(const std::string& filename, const std::vector<double>& vec,
              const int& N);
void printVec(const std::string& filename, const std::vector<double>& vec);
void printVec(const std::vector<double>& vec, const int& N, const int& ax);
void printVec(const std::vector<double>& vec, const int& N);
void printVec(const std::vector<double>& vec);
void printMtx(const std::string& filename, const size_t& rows,
							const size_t& cols, const std::vector<std::vector<double>>& mtx);

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

// Token types
enum class TokenType {
	WHILE,
	CONTROL,
	ASSIGNMENT,
	START, END,
	IDENTIFIER,
	TRUE, FALSE,
	AND, OR, NOT,																// Task 1
	IF, THEN, ELSE,
	// INPUT, OUTPUT,
	LESS, GREATER, EQUAL,
	LEFT_PAREN, RIGHT_PAREN,
	ACTIVATION, LAYER, LOSS, OPTIMIZER,
	PLUS, MINUS, DIVISION, POWER, OPERATOR,
	DOT, COMMA, SPACE, TAB, SEMICOLON, EOF_,
	BOOLEAN, INTEGER, VARIABLE, TENSOR, UNDEFINED,
	PI, E, TAU, PHI, SQRT, LN2, LN10, LOG2E, LOG10E,
	REVERSE
};

/*
 * Mathematical operators	: +, -, *, /, ^, %
 * Logical operators			: &, |, !
 * Comparison operators		: <, >
 * Assignment operators		: =
 * Delimiters							: (, ), [, ]
 * Keywords								: if(?), then(:), else(¿), while(~), start({), end(})
 * Constants							: true(), false()
 * Identifiers						: boolean, integer, tensor, variable, complex
 * End of file						: \0
 * Newline								: "\n"
 * Undefined							: Undefined
 */
extern std::map<std::string, TokenType> KEYWORDS_ML;
extern std::map<TokenType, std::string> KEYWORDS_ML_R;

const std::map<std::string, TokenType> RESERVED_KEYWORDS = {
	{"if", KEYWORDS_ML["?"]},
	{"IF", KEYWORDS_ML["?"]},
	{"If", KEYWORDS_ML["?"]},

	{"then", KEYWORDS_ML["∴"]},
	{"THEN", KEYWORDS_ML["∴"]},
	{"Then", KEYWORDS_ML["∴"]},

	{"else", KEYWORDS_ML["¿"]},
	{"ELSE", KEYWORDS_ML["¿"]},
	{"Else", KEYWORDS_ML["¿"]},

	{"while", KEYWORDS_ML["~"]},
	{"WHILE", KEYWORDS_ML["~"]},
	{"While", KEYWORDS_ML["~"]},

	{"start", KEYWORDS_ML["{"]},
	{"START", KEYWORDS_ML["{"]},
	{"Start", KEYWORDS_ML["{"]},

	{"end", KEYWORDS_ML["}"]},
	{"END", KEYWORDS_ML["}"]},
	{"End", KEYWORDS_ML["}"]},

	{"and", KEYWORDS_ML["∧"]},
	{"AND", KEYWORDS_ML["∧"]},
	{"And", KEYWORDS_ML["∧"]},

	{"or", KEYWORDS_ML["∨"]},
	{"OR", KEYWORDS_ML["∨"]},
	{"Or", KEYWORDS_ML["∨"]},

	{"not", KEYWORDS_ML["¬"]},
	{"NOT", KEYWORDS_ML["¬"]},
	{"Not", KEYWORDS_ML["¬"]},
};


// Token
class Token {
private:
public:
	std::string value;
	TokenType type;
	std::string text;
	int pos;
	char current_char;
	Token() : value(""), type(TokenType::UNDEFINED) {}
	Token(const std::string& value, const TokenType& type) : value(value), \
																													 type(type) {}
	Token(const TokenType& type, const std::string& value) : value(value), \
																													 type(type) {}
	std::string get_value() const { return value; }
	TokenType get_type() const { return type; }
	void set_value(const char& value) { this->value = std::string(1, value); }
	void set_value(const std::string& value) { this->value = value; }
	void set_type(TokenType type) { this->type = type; }
};

class Lexer {
private:
public:
	std::string text;
	int pos;
	char current_char;
	std::string current_str;
	Lexer(const std::string& text) : text(text), pos(0), current_char(text[0]),
																	 current_str(std::string(1, text[0])) {}
	void error() {
		throw std::invalid_argument("Invalid character " + current_char);
	}
	void advance() {
		pos++;
		// Indicates end of input
		current_char = (pos > (text.size() - 1)) ? '\0' : text[pos];
		current_str = std::string(1, current_char);
	}
	void skip_whitespace() {
		while (KEYWORDS_ML[current_str] != TokenType::EOF_ \
					 && std::isspace(current_char)) advance();
	}
	int integer() {
		std::string result = "";
		while (KEYWORDS_ML[current_str] != TokenType::EOF_ \
					 && std::isdigit(current_char)) {
			result += current_char;
			advance();
		}
		return std::stoi(result);
	}
	std::string variable() {
		std::string result = "";
		while (KEYWORDS_ML[current_str] != TokenType::EOF_ \
					 && isalnum(current_char)) {
			result += current_char;
			advance();
		}
		return result;
	}

	Token get_next_token() {
		// Is there a start of input?
		while (KEYWORDS_ML[current_str] != TokenType::EOF_) {
			if (std::isspace(current_char)) {
				skip_whitespace();
				continue;
			} else if (std::isdigit(current_char)) {
				return Token(std::to_string(integer()), TokenType::INTEGER);
			} else if (std::isalpha(current_char)) {
				return Token(variable(), TokenType::VARIABLE);
			} else {
				if (KEYWORDS_ML.find(current_str) == KEYWORDS_ML.end())
					return Token(current_str, TokenType::UNDEFINED);
				else {
					return Token(current_str, KEYWORDS_ML[current_str]);
				}
			}
		}
		return Token();
	}
};


class Interpreter {
private:
public:
  Lexer lexer;
  Token current_token;
  std::map<std::string, double> GLOBAL_SCOPE;
  std::map<std::string, double> LOCAL_SCOPE;
  std::map<std::string, double> TEMP_SCOPE;
  std::vector<std::string> GLOBAL_VARS;
  std::vector<std::string> LOCAL_VARS;
  std::vector<std::string> TEMP_VARS;
  std::vector<std::string> GLOBAL_TENSORS;
  std::vector<std::string> LOCAL_TENSORS;
  std::vector<std::string> TEMP_TENSORS;
  std::vector<std::string> GLOBAL_FUNCS;
  std::vector<std::string> LOCAL_FUNCS;
  std::vector<std::string> TEMP_FUNCS;
  std::vector<std::string> GLOBAL_OBJS;
  std::vector<std::string> LOCAL_OBJS;

	Interpreter(Lexer lexer) : lexer(lexer),
		current_token(lexer.get_next_token()) {}

	void error() {throw std::invalid_argument("Invalid syntax");}
  void translate() {
    while (current_token.get_type() != TokenType::EOF_) {
      if (current_token.get_type() == TokenType::VARIABLE) {
        std::string var_name = current_token.get_value();
        current_token = lexer.get_next_token();
        if (current_token.get_type() != TokenType::ASSIGNMENT) error();
        current_token = lexer.get_next_token();
        if (current_token.get_type() != TokenType::INTEGER) error();
        std::string value = current_token.get_value();
        current_token = lexer.get_next_token();
        std::cout << var_name << " = " << value << std::endl;
      }
    }
  }
};


/*
 * std::string text;
 * Lexer lexer(text);
 * Interpreter interpreter(lexer);
 * std::cout << interpreter.expr() << std::endl;
 *
 * Example:
 *	1. 2 + 3
 *	2. 2 + 3 * 4
 * 	3. 2 + 3 * 4 / 5
 * 	4. 2 + 3 * 4 / 5 - 6
 *  5. 2 + 3 * 4 / 5 - 6 ^ 7
 *  6. 2 + 3 * 4 / 5 - 6 ^ 7 % 8
 *  7. 2 + 3 * 4 / 5 - 6 ^ 7 % 8 & 9
 *  8. 2 + 3 * 4 / 5 - 6 ^ 7 % 8 & 9 | 10
 *  9. 2 + 3 * 4 / 5 - 6 ^ 7 % 8 & 9 | 10 xor 11
 * 10. 2 + 3 * 4 / 5 - 6 ^ 7 % 8 & 9 | 10 xor 11 and 12
 * 11. 2 + 3 * 4 / 5 - 6 ^ 7 % 8 & 9 | 10 xor 11 and 12 or 13
 * 12. 2 + 3 * 4 / 5 - 6 ^ 7 % 8 & 9 | 10 xor 11 and 12 or 13 not 14
 */
#endif // UTILS_HPP