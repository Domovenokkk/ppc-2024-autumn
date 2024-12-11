#include <gtest/gtest.h>

#include "seq/mezhuev_m_sobel_edge_detection/include/seq.hpp"
#include <vector>
#include <chrono>
#include <cmath>

namespace mezhuev_m_sobel_edge_detection {

TEST(SobelEdgeDetectionSeqPerfTest, PreProcessingPerformance) {
    size_t width = 1920;
    size_t height = 1080;

    SobelEdgeDetectionSeq sobel_edge_detection_seq;
    SobelEdgeDetectionSeq::TaskData task_data;

    task_data.width = width;
    task_data.height = height;
    task_data.inputs_count.push_back(width * height);
    task_data.outputs_count.push_back(width * height);
    task_data.inputs.push_back(new uint8_t[width * height]());
    task_data.outputs.push_back(new uint8_t[width * height]());

    auto start = std::chrono::high_resolution_clock::now();
    EXPECT_TRUE(sobel_edge_detection_seq.pre_processing(&task_data));
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "PreProcessing duration: " << duration.count() << " ms" << std::endl;

    delete[] task_data.inputs[0];
    delete[] task_data.outputs[0];
}

TEST(SobelEdgeDetectionSeqPerfTest, RunPerformance) {
    size_t width = 1920;
    size_t height = 1080;

    SobelEdgeDetectionSeq sobel_edge_detection_seq;
    SobelEdgeDetectionSeq::TaskData task_data;

    task_data.width = width;
    task_data.height = height;
    task_data.inputs_count.push_back(width * height);
    task_data.outputs_count.push_back(width * height);
    task_data.inputs.push_back(new uint8_t[width * height]());
    task_data.outputs.push_back(new uint8_t[width * height]());

    sobel_edge_detection_seq.pre_processing(&task_data);

    auto start = std::chrono::high_resolution_clock::now();
    EXPECT_TRUE(sobel_edge_detection_seq.run());
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Run duration: " << duration.count() << " ms" << std::endl;

    delete[] task_data.inputs[0];
    delete[] task_data.outputs[0];
}

TEST(SobelEdgeDetectionSeqPerfTest, PostProcessingPerformance) {
    size_t width = 1920;
    size_t height = 1080;

    SobelEdgeDetectionSeq sobel_edge_detection_seq;
    SobelEdgeDetectionSeq::TaskData task_data;

    task_data.width = width;
    task_data.height = height;
    task_data.inputs_count.push_back(width * height);
    task_data.outputs_count.push_back(width * height);
    task_data.inputs.push_back(new uint8_t[width * height]());
    task_data.outputs.push_back(new uint8_t[width * height]());

    sobel_edge_detection_seq.pre_processing(&task_data);
    sobel_edge_detection_seq.run();

    auto start = std::chrono::high_resolution_clock::now();
    EXPECT_TRUE(sobel_edge_detection_seq.post_processing());
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "PostProcessing duration: " << duration.count() << " ms" << std::endl;

    delete[] task_data.inputs[0];
    delete[] task_data.outputs[0];
}

}  // namespace mezhuev_m_sobel_edge_detection