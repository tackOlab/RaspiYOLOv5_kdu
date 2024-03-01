#pragma once

// Define the size of model
// 1. width and height shall be equal to each ohter
// 2. shall be equal to 160 or 320 or 640
// Corresponding .onnx is required!
constexpr float MODEL_WIDTH  = 160.0f;
constexpr float MODEL_HEIGHT = 160.0f;

// Constants
constexpr float SCORE_THRESHOLD      = 0.5f;
constexpr float NMS_THRESHOLD        = 0.45f;
constexpr float CONFIDENCE_THRESHOLD = 0.45f;
