main: main.cu parANN.cu parANN.h
	nvcc main.cu parANN.cu -Xcompiler -fopenmp -std=c++14 -I../../utils  -o bang_exact -O3
 

clean:
	rm -f parANN 


