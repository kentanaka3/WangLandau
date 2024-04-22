#include "FFNN.hpp"

MLP::MLP(const std::string& file) {
  filepath = std::filesystem::path(file).remove_filename().string();
  paramRead(file);
  #if DEBUG >= 3
  paramPrint();
  #endif
  confRead(file);
  #if DEBUG >= 3
  confPrint();
  #endif
}

MLP::~MLP() {
}

void MLP::confRead(const std::string& file) {
	std::ifstream ifs(file.c_str());
  if (ifs.is_open()) {
		// WL Parameter succesfull
		std::map<std::string, std::string> params;
		params["N_inputs"] 		= "4";
		params["actID"]    		= "4";
		params["N_layers"] 		= "4";
		params["output_file"]	= "6";
		params["input_file"]  = "6";
		params["N_units"]     = "6";
		params["maxNorm"]     = "4";
    params["layerProb"]   = "6";
    std::string line;
		while (std::getline(ifs, line)) {
      // Read Parameters by parsing with RegEx
      params = paramMap(line, params);
    }
    ifs.close();
		N_inputs = std::stoi(params["N_inputs"]);
		N_layers = std::stoi(params["N_layers"]);
		output_file = params["output_file"];
		input_file = params["input_file"];
		actID = std::stoi(params["actID"]);
		N_units.resize(N_layers + 1, 0);
		param2vec(params["N_units"], N_units, N_layers + 1);
		layerProb.resize(N_layers, 0);
		param2vec(params["layerProb"], layerProb, N_layers);
		std::ostringstream filename;
		filename.str("");
		filename << filepath << input_file;
		input = readMtx(filename.str(), N_inputs, N_units[0]);
		filename.str("");
		filename << filepath << output_file;
		output = readVec(filename.str(), N_inputs);
  } else {
    std::cerr << "CRITICAL: Cannot open WL parameter file: " << file \
              << ", aborting." << std::endl;
		exit(EXIT_FAILURE);
  }
	return;
}

void MLP::confPrint() {
	std::cout << "Number of Input patterns (N_inputs): " << N_inputs \
						<< std::endl \
						<< "Number of Layers (N_layers): " << N_layers << std::endl \
						<< "Input file path (input_file): " << input_file << std::endl \
						<< "Output file path (output_file): " << output_file << std::endl \
						<< "Activation function ID (actID): " << actID << std::endl \
						<< "Number of Units per Layer (N_units): ";
	for (int i = 0; i <= N_layers; i++) std::cout << N_units[i] << ", ";
	std::cout << std::endl \
						<< std::endl;
}

double MLP::compEnergy() {
  double energy = 0.;

  return 0;
}