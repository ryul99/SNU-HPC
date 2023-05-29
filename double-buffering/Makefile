all: main

main: main.cu
	/usr/local/cuda/bin/nvcc -o $@ $^

clean: main
	rm main
