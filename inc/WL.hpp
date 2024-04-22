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

}

#endif