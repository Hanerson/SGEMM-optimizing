# Build configuration
BUILD_DIR ?= build
# Target architecture. Ada / RTX 4090 is 89; Turing is 75.
CUDA_ARCH ?= 89
# Resolve nvcc even when it is not on PATH (common in container images).
NVCC ?= $(shell command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)

# Kernel used by `make run`; 0 is cuBLAS.
KERNEL ?= 10

.PHONY: build run bench clean submit

# Configure and build an optimised binary for $(CUDA_ARCH).
build:
	cmake -S . -B $(BUILD_DIR) \
		-DCMAKE_CUDA_COMPILER=$(NVCC) \
		-DCUDA_COMPUTE_CAPABILITY=$(CUDA_ARCH) \
		-DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) -j

# Run one kernel across every size.
run: build
	./$(BUILD_DIR)/sgemm $(KERNEL)

# Run every kernel in turn; this is what the results table was produced with.
bench: build
	@for k in 0 1 2 3 4 5 6 10; do \
		echo "=== kernel $$k ==="; \
		./$(BUILD_DIR)/sgemm $$k | grep -E "performance|dimensions"; \
	done

clean:
	rm -rf $(BUILD_DIR) target matrixValidationFailure.txt

# ===== Auto-generated push target =====
# Usage: make submit MSG="your message"
MSG ?= auto commit
submit:
	@if [ -z "$(MSG)" ]; then \
		echo "Error: missing commit message. Use: make submit MSG=\"xxx\""; \
		exit 1; \
	fi
	git add -A
	git commit -m "$(MSG)" || echo "No changes to commit"
	git push origin HEAD
	@echo "Push complete"
