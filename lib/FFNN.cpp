#include "FFNN.hpp"

MLP::MLP(const std::string& file) {
  filepath = std::filesystem::path(file).remove_filename().string();
  #if DEBUG >= 3
  std::cout << " - Reading WL Parameters - " << std::endl;
  #endif
  paramRead(file);
  #if DEBUG >= 3
  paramPrint();
  std::cout << " - Reading System Configuration - " << std::endl;
  #endif
  confRead(file);
  #if DEBUG >= 3
  confPrint();
  #endif
	#ifdef _OPENMP
  omp_set_num_threads(N_cores);
  #endif
	moveProposedPtr = (prob_single == 1.) ? &MLP::moveSingleProposed : \
	 																				&MLP::moveMultipleProposed;
  W.resize(N_layers);
	B.resize(N_layers);
	std::ostringstream filename;
  for (int layer = 0; layer < N_layers; layer++) {
		int n_from = N_units[layer], n_to = N_units[layer + 1];
		filename.str("");
		filename << filepath << "Network/W" << std::setfill('0') << std::setw(3) \
						 << layer << ".dat";
		#if DEBUG >= 1
		std::cout << "DEBUG(1): Reading Matrix file: " << filename.str() \
							<< std::endl;
		#endif
		W[layer] = readMtx(filename.str(), n_to, n_from);
		filename.str("");
		filename << filepath << "Network/B" << std::setfill('0') << std::setw(3) \
						 << layer << ".dat";
		#if DEBUG >= 1
		std::cout << "DEBUG(1): Reading Vector file: " << filename.str() \
							<< std::endl;
		#endif
		B[layer] = readVec(filename.str(), n_to);
  }
	filename.str("");
	filename << filepath << input_file;
	input = readMtx(filename.str(), N_inputs, N_units[0]);
	filename.str("");
	filename << filepath << output_file;
	output = readVec(filename.str(), N_inputs);
	nodeLedger.resize(N_layers - 1);
	tosamples.resize(N_layers - 1);
	for (int l = 0; l < N_layers - 1 - 1; l++) {
		tosamples[l].resize(N_units[l + 1], 0);
		#pragma omp parallel for shared(N_units, tosample)
		for (int i = 0; i < N_units[l + 1]; i++) tosamples[l][i] = i;
	}
  nodeState.resize(N_layers);
	#pragma omp parallel for shared(nodeState, N_layers)
  for (int l = 0; l < (N_layers - 1); l++) {
    nodeState[l].resize(N_units[l + 1], true);
	}
	hist.resize(N_bins, 0);
  bins_visited.resize(N_bins, false);
	// Set activation Function
  set_act(actID, &act);
}

MLP::~MLP() {
}


double MLP::compEnergy() {
	std::vector<double> losses_mu(N_inputs, 0.);
	#if DEBUG >= 2
	std::ostringstream filename;
	std::vector<std::ofstream> H_files(N_layers);
	for (int layer = 0; layer < N_layers; layer++) {
		filename.str("");
		filename << filepath << "Output/H_l" << std::setfill('0') << std::setw(3) \
						 << layer << ".dat";
		H_files[layer].open(filename.str());
	}
	#endif

	std::vector<double> H_prev, H;
	for (int mu = 0; mu < N_inputs; mu++) {
		for (int layer = 0; layer < N_layers; layer++) { // Loop over Layers except last
			int n_from = N_units[layer], n_to = N_units[layer + 1];
			H.resize((n_to, 0.));
			for (int node_i = 0; node_i < n_to; node_i++) {
				if ((layer < N_layers - 1) && !nodeState[layer][node_i]) continue;
				H[node_i] = B[layer][node_i];
				// TODO: Use reduce(+: sum)
				#pragma omp parallel for
				for (int node_j = 0; node_j < n_from; node_j++) {
					#if DEBUG >= 3
					std::cout << "DEBUG(3): H^{L_{" << layer + 1 << "}} <- " \
															<< "B^{L_{" << layer << "}}_{" << node_i \
															<< "} + "
															<< "W^{L_{" << layer << "}}_{" << node_i \
															<< ", " << node_j << "} * ";
					#endif
					if (layer == 0) {
						std::cout << "V^{" << mu << "}_{" << node_j << "}";
						H[node_i] += W[layer][node_i][node_j] * input[mu][node_j];
					} else if (nodeState[layer - 1][node_j]) {
						std::cout << "\\sigma(H^{L_{" << layer << "}} \\dot "
											<< "\\hat{e}^{L_{" << layer - 1 << "}}_{" \
											<< node_j << "})" << std::endl;
						H[node_i] += W[layer][node_i][node_j] * act(H_prev[node_j]);
					}
				}
				#if DEBUG >= 2
				H_files[layer] << H[node_i] << " ";
				#endif
			}
			#if DEBUG >= 2
			H_files[layer] << std::endl;
			#endif
			if (layer == N_layers - 1) {
				if (N_units[N_layers] > 1) {
					losses_mu[mu] = (argMax(H, N_units[N_layers]) != output[mu]);
				} else {
					losses_mu[mu] = (H[0] * output[mu]) <= 0;
				}
			} else {
				H_prev.swap(H);
			}
		} // layer < N_layers
	}
	#if DEBUG >= 2
	for (int layer = 0; layer < N_layers; layer++) H_files[layer].close();
	#endif
	double loss = 0.;
	#pragma omp parallel for reduction(+:loss)
	for (int mu = 0; mu < N_inputs; mu++) loss += losses_mu[mu];
	return loss;
}

void MLP::moveAccepted() {
}

void MLP::moveSingleProposed(int layer, int node) {
	#if DEBUG >= 3
	std::cout << "DEBUG(3): Selecting Layer_" << layer << ", node_" << node
						<< std::endl;
	#endif
	nodeState[layer][node] = !nodeState[layer][node];
	nodeLedger[layer].push_back(node);
}

void MLP::moveMultipleProposed(int layer, int node) {
	if (unf_dist(gen) <= prob_single) {
		moveSingleProposed(layer, node);
	} else {
		for (layer = 0; layer < N_layers; layer++) {
			// Decide whether to change layer
			if (unf_dist(gen) > layerProb[layer]) continue;
			int N_change = int(unf_dist(gen) * N_units[layer + 1]) + 1;
			std::shuffle(tosamples[layer].begin(), tosamples[layer].end(), gen);
			#if DEBUG >= 3
			std::cout << "DEBUG(3): Changing the following " << N_change \
								<< " nodes." << std::endl;
			#endif
			for (int i = 0; i < N_change; i++) {
				moveSingleProposed(layer, tosamples[layer][i]);
			}
		}
	}
}

double MLP::moveProposed() {
	#pragma omp parallel for shared(nodeLedger, N_layers)
	for (int l = 0; l < N_layers - 1; l++) nodeLedger[l].clear();
	int layer = int(unf_dist(gen) * (N_layers - 1));
	(this->*moveProposedPtr)(layer, int(unf_dist(gen) * N_units[layer + 1]));
  return compEnergy();
}

void MLP::moveRejected() {
	for (int layer = 0; layer < N_layers - 1; layer++) {
		int N_change = nodeLedger[layer].end() - nodeLedger[layer].begin();
		#pragma omp parallel for shared(nodeState, nodeLedger, layer)
		for (int c = 0; c < N_change; c++) {
			int node = nodeLedger[layer][c];
			// TODO: Revise Vectorization protocol
			nodeState[layer][node] = !nodeState[layer][node];
		}
	}
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
	std::cout << "- System Configuration -" << std::endl << std::endl \
						<< "Number of Input patterns (N_inputs): " << N_inputs \
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
