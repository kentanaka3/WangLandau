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
	cmake --build build/ && ./build/Brain.x data/Brain.init

veryclean:
	rm -rf build