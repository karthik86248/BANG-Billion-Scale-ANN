bang: parANN.cu
	nvcc  parANN.cu -Xcompiler -fopenmp -std=c++14 -I../../utils --compiler-options '-fPIC' --shared -o libbang.so -O3
 
 driver: test_driver.cpp
	 g++ test_driver.cpp -ldl -L./ -lbang -std=c++14 -fopenmp -o bang_search

clean:
	rm -f bang.so driver

