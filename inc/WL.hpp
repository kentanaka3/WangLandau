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
public:
  // (Required) Filepath of the Input Parameters
  std::string filepath;

  // Minimal number of Monte Carlo (MC) steps/updates
  int min_MCS = 50;

  // Number of bins for Histogram and log_DoS
  int N_bins;

  #ifdef _OPENMP
  // Number of Threads (OMP)
  int N_cores = 1;
  #endif

  // (Optional) Flatness coefficient
  double flatness = 0.95;

  // (Optional) TODO: Change name of the variable
  int check_histo = 100;

  // (Optional) Delta X
  double delX = 1.;

  // (Optional) Single-weight moves in a single MC step/update
  int N_rand = 200;

  // (Optional) Final precision of F factor for Wang-Landau implementation
  double log_F_end = 1e-6;

  // (Optional) Probability of choosing single moves
  double prob_single = 1.;

  // Histogram for the Wang-Landau sampling
  std::vector<unsigned int> hist;
  std::vector<bool> bins_visited;

  // Log Density of States (Entropy): For Wang-Landau sampling
  std::vector<double> log_DoS;

  // Explorable regime
  std::vector<int> explorable{2, 0};

  virtual ~WL() {};

  void paramRead(const std::string& filepath);
  void paramPrint();
  bool is_flat(const double& log_F, const int& MC_step, const int& check,
               const int& count);
  double min_DoS();
  void run();

  virtual double compEnergy() = 0;
  virtual void moveAccepted() = 0;
  virtual double moveProposed() = 0;
  virtual void moveRejected() = 0;
};

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
    N_rand = std::stoi(params["N_rand"]);
    if (params["prob_single"] != "3")
      prob_single = std::stod(params["prob_single"]);
		param2vec(params["explorable"], explorable, 2);
    #ifdef _OPENMP
    N_cores = std::stoi(params["N_cores"]);
    #endif
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

/*
 * Determine the flatness of the Histogram up to a predefined value
 */
bool WL::is_flat(const double& log_F, const int& MC_step, const int& check,
                 const int& count) {
  int count_bins = 0;
  double avg_hist = 0.;
	bool flat = true;

  // Computing the average of an Histogram
  #pragma omp parallel for reduction(+:avg_hist)
  for (int bin = 0; bin < N_bins; bin++) avg_hist += hist[bin] * bins_visited[bin];
  #pragma omp parallel for reduction(+:count_bins)
  for (int bin = 0; bin < N_bins; bin++) count_bins += bins_visited[bin];

  avg_hist /= count_bins;

  #if DEBUG >= 2
  std::cout << "Average of Histogram: " << avg_hist \
            << ", on a total number of bins: " << count_bins << std::endl;
  #endif

  // For an histogram to be considered "flat", the count in each bin must not
  // be lower than flatness times its average or higher than (2 - flatness)
  // times the average.
  // Note: OMP parallelization not available due to break.
  for (int bin = 0; bin < N_bins; bin++) {
    double tmp = hist[bin] / avg_hist;
    if (bins_visited[bin] && (tmp < flatness || (2. - tmp) < flatness)) {
      flat = false;
      break;
    }
  }
  std::ostringstream filename;
  filename.str("");
  filename << filepath << "Output/Hist" << (flat ? "f" : "o") \
           << "_" << std::setfill('0') << std::setw(10) \
           << std::setprecision(8) << log_F;
  if (!flat) {
    filename << "_" << std::setfill('0') << std::setw(5) << MC_step \
             << "_" << std::setfill('0') << std::setw(5) << check;
    std::cout << "Flatness condition NOT satisfied." << std::endl;
  } else {
    std::cout << "Flatness condition satisfied" << std::endl \
              << "Iteration with F factor = " << log_F << " completed. " \
              << "Histogram flattened." << std::endl;
  }
  filename << ".dat";
  std::ofstream file_out(filename.str());
	for (int bin = 0; bin < N_bins; bin++) {
		file_out << delX*bin << " " << (double)hist[bin] / (double)count \
						 << " " << log_DoS[bin] << std::endl;
	}
	file_out.close();
  return flat;
}

double WL::min_DoS() {
  double precision = log_F_end / 1000.;
  int bin = 0;
  for (bin; bin < N_bins - 1; bin++)
    if ((log_DoS[bin] < precision) && (precision < log_DoS[bin + 1])) break;
  double min = log_DoS[++bin];
  for (++bin; bin < N_bins; bin++)
    if ((min > log_DoS[bin]) && (log_DoS[bin] > precision)) min = log_DoS[bin];
  return min;
}

void WL::run() {
  N_bins = int((double)(explorable[1] - explorable[0]) / delX) + 1;
  hist.resize(N_bins, 0);
  log_DoS.resize(N_bins, 0.);
  bins_visited.resize(N_bins, false);
  // Energy of the system
  double energy = compEnergy();
  //
  int bin = int(energy / delX);
  std::ofstream file_out;
	std::ostringstream filename;
  filename.str("");
	filename << filepath << "Output/H.dat";
	file_out.open(filename.str());
  file_out << bin << ", " << energy << std::endl;
  file_out.close();

  // Boolean which determines that the bin associated to the next mapping in
  // the sequence has been visited
  bool visited = true;
  bins_visited[bin] = visited;

  // F factor for Wang-Landau implementation
  double log_F = 1.;
  log_DoS[bin] += log_F;

  // Counter for checking the flatness of the Histogram
  int check = 0;

  #if DEBUG >= 0
  std::cout << "- Running Wang-Landau (WL) -" << std::endl;
  #endif
  while (log_F >= log_F_end) { // Repeat until Convergence
    #if DEBUG >= 1
    std::cout << "Iteration of Wang-Landau with F factor = " << log_F \
              << std::endl;
    #endif
    int MC_step = 0;
    int count = 0;
		check++;

    // Flag which determines that Histogram (hist) is NOT flat.
    bool flat = false;

    // Reset Histogram
    #pragma omp parallel for shared(hist, N_bins)
    for (int i = 0; i < N_bins; i++) hist[i] = 0;

    while (!flat) { // Repeat until Histogram (hist) is flat
      for (int i = 0; i < N_rand; i++) {
        #if DEBUG >= 3
        std::cout << "DEBUG(3): MCS = " << i << ", ";
        #endif
        // Proposed energy of the system after a move
        double proposed_energy = moveProposed();
        int proposed_bin = int(proposed_energy / delX);
        if ((proposed_energy >= explorable[0]) && \
            (proposed_energy <= explorable[1]) && \
            log(unf_dist(gen)) < (log_DoS[bin] - log_DoS[proposed_bin])) {
          // Accept move
          #if DEBUG >= 3
          std::cout << " accepted" << std::endl;
          #endif
          moveAccepted();
          // Update bin and energy
          energy = proposed_energy;
          bin = proposed_bin;
          visited = bins_visited[bin];
        } else {
          // Reject move
          #if DEBUG >= 3
          std::cout << " rejected" << std::endl;
          #endif
          moveRejected();
        }
        hist[bin]++;
        count++;
        if (!visited) {
          #if DEBUG >= 0
          std::cout << "DEBUG(0): New visited " << bin << "w/ loss " \
                    << delX*bin << std::endl;
          #endif
          bins_visited[bin] = true;
          log_DoS[bin] += min_DoS();
          break;
        } else {
          log_DoS[bin] += log_F;
        }
      } // for i < N_rand

      // If a NEW bin has been visited, restart the Histogram but do NOT update
      // F factor.
      if (!visited) break;
      if ((++check >= check_histo) && (++MC_step >= min_MCS)) {
				std::cout << "Checking histo of f = " << logf \
									<< ", MC_step = " << MC_step \
									<< ", checkbox = " << check << std::endl;
				check = 0; //reset the counter for checking the histogram flatness
				flat = !is_flat(log_F, MC_step, check, count); //checking the histogram flatness
			}
    } // Flat Histogram
    log_F *= 0.5;
  } // log_F < log_F_end
	std::cout << "Done." << std::endl;
}
#endif