CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -Isrc -Ilib/onnx/include
LDFLAGS = -Llib/onnx/lib -Wl,-rpath,'$$ORIGIN/../lib/onnx/lib'
LDLIBS = -lonnxruntime -lpthread -lm

SRC_DIR = src
BUILD_DIR = build
TARGET = $(BUILD_DIR)/worker

SRCS = $(shell find $(SRC_DIR) -name '*.c')
OBJS = $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(OBJS) $(LDFLAGS) $(LDLIBS) -o $(TARGET)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
