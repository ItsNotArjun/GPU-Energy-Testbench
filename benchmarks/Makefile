ARCH ?= sm_89
OPT  := -O3

all: ld_benchmark st_benchmark

ld_benchmark: ld.cu
	nvcc $(OPT) -arch=$(ARCH) ld.cu -o ld_benchmark

st_benchmark: st.cu
	nvcc $(OPT) -arch=$(ARCH) st.cu -o st_benchmark

clean:
	rm -f ld_benchmark st_benchmark ld_benchmark.exe st_benchmark.exe

.PHONY: all clean
