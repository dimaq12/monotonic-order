CC ?= gcc
CFLAGS ?= -O3 -DNDEBUG -std=c11 -fPIC -march=native -fopenmp -Wall -Wextra -Wpedantic
LDFLAGS ?= -shared

.PHONY: all clean test package-test wheel asan benchmark benchmark-warmed benchmark-argsort benchmark-lexargsort

all: libidealorder.so

libidealorder.so: ideal_order.c ideal_order.h
	$(CC) $(CFLAGS) $(LDFLAGS) ideal_order.c -lm -o $@

test: package-test

package-test:
	python3 setup.py build_ext --inplace --force
	PYTHONPATH=src python3 -m unittest -v test_ideal_order.py tests/test_package.py

wheel:
	python3 -m pip wheel . --no-build-isolation --no-deps --wheel-dir dist

asan:
	$(CC) -O1 -g -std=c11 -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer \
		ideal_order.c test_core.c -lm -o /tmp/ideal_order_asan
	ASAN_OPTIONS=detect_leaks=0 /tmp/ideal_order_asan

benchmark: all
	python3 setup.py build_ext --inplace
	OMP_NUM_THREADS=3 OMP_PROC_BIND=close OMP_PLACES=cores PYTHONPATH=src python3 benchmark.py

benchmark-warmed: all
	python3 setup.py build_ext --inplace
	OMP_NUM_THREADS=3 OMP_PROC_BIND=close OMP_PLACES=cores PYTHONPATH=src python3 benchmark_warmed.py

benchmark-argsort:
	python3 setup.py build_ext --inplace
	PYTHONPATH=src python3 benchmark_argsort.py

benchmark-lexargsort:
	python3 setup.py build_ext --inplace
	PYTHONPATH=src python3 benchmark_lexargsort.py

clean:
	rm -f libidealorder.so
