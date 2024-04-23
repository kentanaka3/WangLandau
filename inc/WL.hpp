#ifndef WangLandau
#define WangLandau

#include <random>
#include <string>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "utils.hpp"

std::random_device rndm;
std::mt19937_64 gen(rndm());
std::uniform_int_distribution<> unf_dist(0., 1.);

class WL {
private:
  /* data */
public:
  // (Required) Filepath of the Input Parameters
  std::string filepath;

  // Minimal number of Monte Carlo (MC) steps/updates
  int min_MCS;

  // Number of bins for Histogram and log_DoS
  int N_bins;

  // Number of Threads (OMP)
  int N_cores;

  // (Optional) Flatness coefficient
  double flatness;

  // (Optional) TODO: Change name of the variable
  int check_histo;

  // (Optional) Delta X
  double delX;

  // (Optional) Single-weight moves in a single MC step/update
  int N_rand;

  // (Optional) Final precision of F factor for Wang-Landau implementation
  double log_F_end;

  // (Optional) Probability of choosing single moves
  double prob_single;

  // Histogram for the Wang-Landau sampling
  std::vector<unsigned int> hist;
  std::vector<bool> bins_visited;

  // Log Density of States (Entropy): For Wang-Landau sampling
  std::vector<double> log_DoS;

  // Explorable regime
  std::vector<int> explorable{2, 0};

  WL(/* args */);
  ~WL();
  
  void paramRead(const std::string& filepath);
  void paramPrint();
  bool is_flat(const double& log_F);
  double min_DoS();
  void run();
  void printHist(const double& log_F, const int& MC_step, const int& check,
                 const int& count);

  virtual double compEnergy() = 0;
  virtual void moveAccepted() = 0;
  virtual void moveSingleProposed() = 0;
  virtual double moveProposed() = 0;
  virtual void moveRejected() = 0;
};

WL::WL(/* args */) {
  min_MCS = 50;
  log_F_end = 1e-6;
  N_cores = 1;
  N_rand = 200;
  delX = 1.;
  check_histo = 100;
  flatness = 0.95;
}

WL::~WL() {
}

void WL::paramRead(const std::string& filepath) {
  std::ifstream ifs(filepath.c_str());
  if (ifs.is_open()) {
    // WL Parameter succesfull
    std::map<std::string, std::string> params;
    params["min_MCS"]       = "4";
    params["log_F_end"]     = "2";
    params["N_cores"]       = "4";
    params["N_rand"]        = "4";
    params["delX"]          = "3";
    params["check_histo"]   = "4";
    params["flatness"]      = "2";
    params["prob_single"]   = "3";
    params["explorable"]    = "6";
    std::string line;
    /*
    *key = std::stof((*it)[3]);
    if ((*it)[6] == "FALSE" || \
        (*it)[6] == "False" || \
        (*it)[6] == "false") {
      *key = false;
    } else if ((*it)[6] == "TRUE" || \
                (*it)[6] == "True" || \
                (*it)[6] == "true") {
      *key = true;
    }
	  */
    while (std::getline(ifs, line)) {
      // Read Parameters by parsing with RegEx
      params = paramMap(line, params);
    }
    ifs.close();
    if (params["log_F_end"] != "2") log_F_end = std::stod(params["log_F_end"]);
    if (params["flatness"] != "2") flatness = std::stod(params["flatness"]);
    if (params["delX"] != "3") delX = std::stod(params["delX"]);
    check_histo = std::stoi(params["check_histo"]);
    min_MCS = std::stoi(params["min_MCS"]);
    N_cores = std::stoi(params["N_cores"]);
    N_rand = std::stoi(params["N_rand"]);
    if (params["prob_single"] != "3")
      prob_single = std::stod(params["prob_single"]);
		param2vec(params["explorable"], explorable, 2);
  } else {
    std::cerr << "CRITICAL: Cannot open WL parameter file: " << filepath \
              << ", aborting." << std::endl;
		exit(EXIT_FAILURE);
  }
  return;
}

void WL::paramPrint() {
  std::cout << "- WL Parameters -" << std::endl << std::endl;
  #ifdef _OPENMP
  std::cout << "WARNING: OpenMP enabled with " << N_cores << " cores." \
            << std::endl;
  #endif
  std::cout << "min_MCS       : " << min_MCS << std::endl \
            << "log_F_end     : " << log_F_end << std::endl \
            << "delX          : " << delX << std::endl \
            << "flatness      : " << flatness << std::endl \
            << "check_histo   : " << check_histo << std::endl \
            << "N_rand        : " << N_rand << std::endl \
            << "prob_single   : " << prob_single << std::endl \
            << "explorable    : " << explorable[0] << " - " << explorable[1] \
                                  << std::endl \
            << "filepath      : " << filepath << std::endl << std::endl;
}

bool WL::is_flat(const double& log_F) {
  bool flat = true;

  return flat;
}

double WL::min_DoS() {
  double DoS = 0.;

  return DoS;
}

void WL::run() {

}

void WL::printHist(const double& log_F, const int& MC_step, const int& check,
                   const int& count) {

}


#endif