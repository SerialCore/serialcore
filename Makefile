CC = gcc
CPPC = g++

BUILD_DIR = build/
OBJ_DIR = $(BUILD_DIR)obj/
BIN_DIR = $(BUILD_DIR)bin/

INC_DIR = include/
SRC_DIR = src/
MATH_DIR = $(SRC_DIR)math/
SONN_DIR = $(SRC_DIR)sonn/
FFNN_DIR = $(SRC_DIR)ffnn/
TEST_DIR = test/

EXECUTABLE = $(BIN_DIR)serialcore

INCLUDES = -I$(INC_DIR)
CFLAGS = -O2 -Wall -Wno-unknown-pragmas -Wfatal-errors -fPIC -Ofast
CPPFLAGS = $(CFLAGS)
LDFLAGS = -lm -pthread

C_SOURCES = $(wildcard $(SRC_DIR)*.c $(MATH_DIR)*.c $(SONN_DIR)*.c $(FFNN_DIR)*.c)

OBJECTS = $(patsubst %.c,$(OBJ_DIR)%.o,$(C_SOURCES)) \
          $(patsubst %.cc,$(OBJ_DIR)%.o,$(CPP_SOURCES)) \
          $(patsubst %.cu,$(OBJ_DIR)%.o,$(CU_SOURCES))

TEST_SOURCES = $(wildcard $(TEST_DIR)test_*.c)
TEST_BINS = $(patsubst $(TEST_DIR)%.c,$(BIN_DIR)%,$(TEST_SOURCES))
TEST_OBJS = $(patsubst $(TEST_DIR)%.c,$(OBJ_DIR).o,$(TEST_SOURCES))

# Main executable's own translation unit; excluded from test binaries so
# each test can provide its own main().
MAIN_OBJ = $(OBJ_DIR)$(SRC_DIR)main.o

all: $(EXECUTABLE) $(TEST_BINS)

$(EXECUTABLE): $(OBJECTS)
	@mkdir -p $(@D)
	$(CC) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR)%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)%.o: %.cc
	@mkdir -p $(@D)
	$(CPPC) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(BIN_DIR)test_%: $(OBJ_DIR)$(TEST_DIR)test_%.o $(filter-out $(MAIN_OBJ),$(OBJECTS))
	@mkdir -p $(@D)
	$(CC) $< $(filter-out $(MAIN_OBJ),$(OBJECTS)) -o $@ $(LDFLAGS)

$(OBJ_DIR)$(TEST_DIR)%.o: $(TEST_DIR)%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

run: $(EXECUTABLE)
	./$(EXECUTABLE)

test: $(TEST_BINS)
	@set -e; for bin in $(TEST_BINS); do \
		echo "--- $$bin ---"; \
		./$$bin; \
	done

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all run test clean
