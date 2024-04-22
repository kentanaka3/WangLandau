all: veryclean build

build: openmp
	cmake --build build/

openmp:
	cmake -S . -B build -DUSE_OMP=ON

test: veryclean
	cmake -S . -B build -DTEST=ON && cmake --build build/

run_test:
	./test_utils.x

veryclean:
	rm -rf build