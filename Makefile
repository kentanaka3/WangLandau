all: veryclean build

build: openmp
	cmake --build build/

openmp:
	cmake -S . -B build -DUSE_OMP=ON

test: veryclean
	cmake -S . -B build -DTEST=ON && cmake --build build/ && ./build/test_utils; ./build/test_WL; ./build/Brain.x

run_test:
	./test_utils.x

brain: veryclean openmp
	cmake --build build/ && ./build/Brain.x 1 /home/ken/Desktop/WangLandau/data/Brain.init

veryclean:
	rm -rf build