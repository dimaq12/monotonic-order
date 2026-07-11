CC ?= gcc
PYTHON ?= python3
CFLAGS ?= -O3 -DNDEBUG -std=c11 -fPIC -march=native -fopenmp -Wall -Wextra -Wpedantic
LDFLAGS ?= -shared

.PHONY: all clean test package-test wheel asan benchmark benchmark-warmed benchmark-argsort benchmark-lexargsort benchmark-strings benchmark-spatial benchmark-hilbert

all: libmonotonic_order.so

libmonotonic_order.so: monotonic_order.c monotonic_order.h
	$(CC) $(CFLAGS) $(LDFLAGS) monotonic_order.c -lm -o $@

test: package-test

package-test:
	$(PYTHON) setup.py build_ext --inplace --force
	PYTHONPATH=src $(PYTHON) -m unittest -v test_monotonic_order.py tests/test_package.py

wheel:
	$(PYTHON) -m pip wheel . --no-build-isolation --no-deps --wheel-dir dist

asan:
	$(CC) -O1 -g -std=c11 -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer \
		monotonic_order.c test_core.c -lm -o /tmp/monotonic_order_asan
	ASAN_OPTIONS=detect_leaks=0 /tmp/monotonic_order_asan

benchmark: all
	$(PYTHON) setup.py build_ext --inplace
	OMP_NUM_THREADS=3 OMP_PROC_BIND=close OMP_PLACES=cores PYTHONPATH=src $(PYTHON) benchmark.py

benchmark-warmed: all
	$(PYTHON) setup.py build_ext --inplace
	OMP_NUM_THREADS=3 OMP_PROC_BIND=close OMP_PLACES=cores PYTHONPATH=src $(PYTHON) benchmark_warmed.py

benchmark-argsort:
	$(PYTHON) setup.py build_ext --inplace
	PYTHONPATH=src $(PYTHON) benchmark_argsort.py

benchmark-lexargsort:
	$(PYTHON) setup.py build_ext --inplace
	PYTHONPATH=src $(PYTHON) benchmark_lexargsort.py

benchmark-strings:
	$(PYTHON) setup.py build_ext --inplace
	PYTHONPATH=src $(PYTHON) benchmark_strings.py

benchmark-spatial:
	$(PYTHON) setup.py build_ext --inplace
	PYTHONPATH=src $(PYTHON) benchmark_spatial.py

benchmark-hilbert:
	$(PYTHON) setup.py build_ext --inplace
	PYTHONPATH=src $(PYTHON) benchmark_hilbert.py

clean:
	$(PYTHON) setup.py clean --all
	rm -f libmonotonic_order.so
