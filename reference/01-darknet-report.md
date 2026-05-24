# Darknet Report

## Repository
- Path: `reference/darknet`
- Upstream: `https://github.com/pjreddie/darknet.git`

## Purpose
Darknet is a C/CUDA neural network framework centered on image models. In this checkout it provides training and inference for image classification, object detection, segmentation, and several recurrent architectures. The codebase is config-driven and exposes a CLI, a C library, and a Python `ctypes` wrapper.

Key files:
- `reference/darknet/examples/darknet.c`
- `reference/darknet/src/network.c`
- `reference/darknet/include/darknet.h`

## Functional Scope
- Classification workflows in `reference/darknet/examples/classifier.c`
- Detection workflows in `reference/darknet/examples/detector.c`
- Legacy YOLO entrypoints in `reference/darknet/examples/yolo.c`
- Image loading/augmentation in `reference/darknet/src/data.c` and `reference/darknet/src/image.c`
- Python API in `reference/darknet/python/darknet.py`

The repo includes cfg families for AlexNet, ResNet, Darknet-53, YOLOv1, YOLOv2, YOLOv3, and RNN/GRU variants under `reference/darknet/cfg/`.

## Main Architecture
The central runtime model is a `network` made of a sequence of `layer` structs. The `layer` type in `reference/darknet/include/darknet.h` is a large tagged struct containing function pointers plus fields for all supported layer families. The network runtime in `reference/darknet/src/network.c` is responsible for:
- network construction and resizing
- forward/backward/update passes
- prediction helpers
- detection box extraction
- multi-GPU training helpers

Configuration parsing is handled by `reference/darknet/src/parser.c`, which turns cfg sections like `[net]`, `[convolutional]`, `[route]`, `[shortcut]`, `[yolo]` into instantiated layers.

## Algorithm Implementations

### Convolution Core
The main convolution implementation is in `reference/darknet/src/convolutional_layer.c`. It uses the classic `im2col + GEMM` approach:
- lowering with `reference/darknet/src/im2col.c`
- matrix multiply with `reference/darknet/src/gemm.c`
- backward input reconstruction with `reference/darknet/src/col2im.c`

This is the main numeric kernel behind CNN inference and training.

### Fully Connected and BatchNorm
- Dense layers: `reference/darknet/src/connected_layer.c`
- Batch normalization: `reference/darknet/src/batchnorm_layer.c`

These implement manual forward and backward passes rather than delegating to a general tensor engine.

### Detection Heads
Three detector styles are implemented:
- `reference/darknet/src/detection_layer.c`: older YOLOv1-style grid detector
- `reference/darknet/src/region_layer.c`: anchor-based region layer, closer to YOLOv2
- `reference/darknet/src/yolo_layer.c`: YOLOv3-style detection head

Important detection logic includes:
- bounding-box decoding in `get_yolo_box()`
- target assignment and delta construction in `delta_yolo_box()` and `forward_yolo_layer()`
- inference-time box decoding in `get_yolo_detections()`
- IoU and NMS in `reference/darknet/src/box.c`

### Recurrent Models
The repo also includes explicit implementations for:
- `reference/darknet/src/rnn_layer.c`
- `reference/darknet/src/gru_layer.c`
- `reference/darknet/src/lstm_layer.c`
- `reference/darknet/src/crnn_layer.c`

## Training and Inference Flow
Typical detector flow:
1. Parse cfg in `reference/darknet/src/parser.c`
2. Load weights with `load_weights()` from `reference/darknet/src/parser.c`
3. Load and augment data with `reference/darknet/src/data.c`
4. Run `forward_network()`, `backward_network()`, `update_network()` from `reference/darknet/src/network.c`
5. Save checkpoints from the example command path in `reference/darknet/examples/detector.c`

Inference flow:
1. Preprocess image
2. Run `network_predict()` / `network_predict_image()`
3. Extract detections with `get_network_boxes()`
4. Run NMS with `do_nms_sort()` or `do_nms_obj()`

## Build and Runtime Model
Build is controlled by `reference/darknet/Makefile` with toggles for `GPU`, `CUDNN`, `OPENCV`, `OPENMP`, and `DEBUG`. Outputs include:
- `darknet`
- `libdarknet.so`
- `libdarknet.a`

## Strengths
- End-to-end training/inference stack in C/CUDA
- Clear layer-by-layer implementation
- Config-driven experimentation
- Includes detection, classification, and recurrent families in one codebase

## Limitations and Quirks
- The README markets newer YOLO generations, but the checked-out code is fundamentally classic Darknet with older cfg families.
- `.data` files referenced by examples are not present in this snapshot, so some example commands rely on external assets.
- The `layer` struct is highly monolithic, which makes extension practical but messy.
- The Makefile assumes older OpenCV/CUDA conventions.
- Several detector losses are old hand-coded formulations rather than more modern IoU-loss families.

## Overall Assessment
This repo is a classic Darknet codebase: config-driven, layer-oriented, and explicit about the mechanics of CNN and YOLO training. It is useful as a reference implementation for convolutional networks and early-to-mid generation YOLO detectors, especially when the goal is to study how the algorithms are implemented directly in C.
