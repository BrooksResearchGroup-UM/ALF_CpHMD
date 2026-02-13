NVCC     ?= nvcc
OUT_DIR  ?= ..
WHAM_TARGET   := $(OUT_DIR)/libwham.so
NL_TARGET := $(OUT_DIR)/libnonlinear.so

# Build fat binary for all supported architectures (sm_60+ for double atomicAdd).
# -arch=all-major embeds SASS for every major arch and PTX for forward compat.
NVCCFLAGS := -shared -Xcompiler -fPIC -arch=all-major -Wno-deprecated-gpu-targets

.PHONY: all clean

all: $(WHAM_TARGET) $(NL_TARGET)

$(WHAM_TARGET): wham.cu wham.h
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(NL_TARGET): nonlinear.cu nonlinear.h
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(WHAM_TARGET) $(NL_TARGET)
