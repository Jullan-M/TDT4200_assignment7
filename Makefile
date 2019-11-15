
.PHONY: clean

main: libs/bitmap.h libs/bitmap.c main.cu
	nvcc libs/bitmap.c main.cu -rdc=true -o main

clean:
	rm main

