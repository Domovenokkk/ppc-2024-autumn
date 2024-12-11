#include "seq/mezhuev_m_sobel_edge_detection/include/seq.hpp"

#include <cmath>
#include <iostream>

namespace mezhuev_m_sobel_edge_detection {

bool SobelEdgeDetectionSeq::validation() {
    if (!taskData || taskData->inputs.empty() || taskData->outputs.empty()) {
        return false;
    }

    if (taskData->inputs.size() != 1 || taskData->outputs.size() != 1) {
        return false;
    }

    if (!taskData->inputs[0] || !taskData->outputs[0]) {
        return false;
    }

    if (taskData->inputs_count.empty() || taskData->outputs_count.empty() ||
        taskData->inputs_count[0] != taskData->outputs_count[0]) {
        return false;
    }

    return true;
}

bool SobelEdgeDetectionSeq::pre_processing(TaskData* task_data) {
    if (!validation()) {
        return false;
    }

    gradient_x.resize(task_data->width * task_data->height);
    gradient_y.resize(task_data->width * task_data->height);

    taskData = task_data;
    return true;
}

bool SobelEdgeDetectionSeq::run() {
    if (!taskData) {
        std::cerr << "Error: Task data is null." << std::endl;
        return false;
    }

    size_t width = taskData->width;
    size_t height = taskData->height;

    if (taskData->inputs[0] == nullptr || taskData->outputs[0] == nullptr) {
        std::cerr << "Error: Input or output buffer is null." << std::endl;
        return false;
    }

    for (size_t y = 1; y < height - 1; ++y) {
        for (size_t x = 1; x < width - 1; ++x) {
            int16_t gx = 0, gy = 0;

            gx += -1 * taskData->inputs[0][(y - 1) * width + (x - 1)];
            gx += 1 * taskData->inputs[0][(y - 1) * width + (x + 1)];
            gx += -2 * taskData->inputs[0][y * width + (x - 1)];
            gx += 2 * taskData->inputs[0][y * width + (x + 1)];
            gx += -1 * taskData->inputs[0][(y + 1) * width + (x - 1)];
            gx += 1 * taskData->inputs[0][(y + 1) * width + (x + 1)];

            gy += -1 * taskData->inputs[0][(y - 1) * width + (x - 1)];
            gy += -2 * taskData->inputs[0][(y - 1) * width + x];
            gy += -1 * taskData->inputs[0][(y - 1) * width + (x + 1)];
            gy += 1 * taskData->inputs[0][(y + 1) * width + (x - 1)];
            gy += 2 * taskData->inputs[0][(y + 1) * width + x];
            gy += 1 * taskData->inputs[0][(y + 1) * width + (x + 1)];

            gradient_x[y * width + x] = gx;
            gradient_y[y * width + x] = gy;

            int magnitude = std::sqrt(gx * gx + gy * gy);

            if (magnitude > 255) {
                magnitude = 255;
            }

            taskData->outputs[0][y * width + x] = static_cast<uint8_t>(magnitude);
        }
    }

    return true;
}

bool SobelEdgeDetectionSeq::post_processing() {
    if (!taskData || taskData->outputs[0] == nullptr) {
        std::cerr << "Error: Invalid output buffer." << std::endl;
        return false;
    }

    for (size_t i = 0; i < taskData->outputs_count[0]; ++i) {
        if (taskData->outputs[0][i] == 0) {
            std::cerr << "Error: Invalid output value at index " << i << "." << std::endl;
            return false;
        }
    }

    return true;
}

}  // namespace mezhuev_m_sobel_edge_detection