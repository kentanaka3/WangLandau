all: veryclean build

build: openmp
	cmake --build build/

openmp:
	cmake -S . -B build -DUSE_OMP=ON

test: veryclean 
	cmake -S . -B build -DTEST=ON -DUSE_OMP=ON && cmake --build build/ && ./build/test_utils; ./build/test_WL; ./build/test_neuron;

run_test:
	./test_utils.x

brain: veryclean openmp
	cmake --build build/ && ./build/Brain.x test/data/Neuron/FNN.init

brain_mpi: veryclean
	cmake -S . -B build -DUSE_MPI=ON -DTEST=ON -DUSE_OMP=ON && cmake --build build/ && mpirun -np 4 ./build/Brain.x test/data/Neuron/FNN.init

debug_brain: veryclean openmp
	cmake --build build/ && gdb --args ./build/Brain.x test/data/Neuron/FNN.init

veryclean:
	rm -rf build