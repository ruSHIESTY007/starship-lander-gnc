# Flight software build. Strict C99, single precision, no dynamic memory.
CC      ?= gcc
CSTD     = -std=c99 -pedantic
WARN     = -Wall -Wextra -Werror -Wshadow -Wconversion -Wdouble-promotion -Wfloat-conversion \
           -Wstrict-prototypes -Wmissing-prototypes -Wcast-qual -Wundef
OPT     ?= -O2
CFLAGS   = $(CSTD) $(WARN) $(OPT) -Iinclude -fPIC -fsingle-precision-constant
LDLIBS   = -lm

SRC   := $(wildcard src/*.c)
OBJ   := $(patsubst src/%.c,build/%.o,$(SRC))
TSRC  := $(wildcard tests/test_*.c)
TOBJ  := $(patsubst tests/%.c,build/tests/%.o,$(TSRC))

.PHONY: all lib test bench size clean
all: lib test

lib: build/libfsw.so build/libfsw.a

build/%.o: src/%.c $(wildcard include/*.h) | build
	$(CC) $(CFLAGS) -c $< -o $@

build/tests/%.o: tests/%.c $(wildcard include/*.h) tests/test_framework.h | build
	@mkdir -p build/tests
	$(CC) $(CFLAGS) -Wno-double-promotion -Wno-conversion -Wno-float-conversion -Itests -c $< -o $@

build/libfsw.so: $(OBJ)
	$(CC) -shared -o $@ $^ $(LDLIBS)

build/libfsw.a: $(OBJ)
	ar rcs $@ $^

build/test_fsw: $(TOBJ) build/libfsw.a
	$(CC) -o $@ $(TOBJ) build/libfsw.a $(LDLIBS)

test: build/test_fsw
	./build/test_fsw

# Static memory / code footprint of each module at -Os (proxy for MCU flash/RAM)
size: | build
	@mkdir -p build/os
	@for f in $(SRC); do $(CC) $(CSTD) -Os -Iinclude -fsingle-precision-constant -c $$f -o build/os/$$(basename $$f .c).o; done
	@size build/os/*.o

build:
	@mkdir -p build

clean:
	rm -rf build

bench: build/libfsw.a
	$(CC) $(CSTD) -O2 -Iinclude -fsingle-precision-constant tests/bench.c build/libfsw.a -lm -o build/bench
	./build/bench
