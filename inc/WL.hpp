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

  // Number of Processing Elements (MPI)
  int N_PEs;

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

  WL(/* args */);
  ~WL();
  double compEnergy();
  void paramRead(const std::string& filepath);
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

double WL::compEnergy() {
  double energy = 0.;

  return 0;
}

void WL::paramRead(const std::string& filepath) {
  std::ifstream ifs(filepath.c_str());
  if (ifs.is_open()) {
    // WL Parameter succesfull
    std::map<std::string, std::string> params;
    params["min_MCS"]       = "4";
    params["log_F_end"]     = "2";
    params["N_cores"]       = "4";
    params["N_PEs"]         = "4";
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
      params = readParam(line, params);
    }
    ifs.close();
    if (params["log_F_end"] != "2") log_F_end = std::stod(params["log_F_end"]);
    if (params["flatness"] != "2") flatness = std::stod(params["flatness"]);
    if (params["delX"] != "3") delX = std::stod(params["delX"]);
    check_histo = std::stoi(params["check_histo"]);
    min_MCS = std::stoi(params["min_MCS"]);
    N_cores = std::stoi(params["N_cores"]);
    N_rand = std::stoi(params["N_rand"]);
    N_PEs = std::stoi(params["N_PEs"]);
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

#endif