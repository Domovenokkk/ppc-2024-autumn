#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <vector>

#include "seq/mezhuev_m_sobel_edge_detection/include/seq.hpp"

namespace mezhuev_m_sobel_edge_detection {

TEST(mezhuev_m_sobel_edge_detection, ValidationTestValidData) {
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq sobel_edge_detection_seq;

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionSeq::TaskData task_data;
  task_data.width = 5;
  task_data.height = 5;

  task_data.inputs_count.push_back(25);
  task_data.outputs_count.push_back(25);

  task_data.inputs.push_back(new uint8_t[25]{0});
  task_data.outputs.push_back(new uint8_t[25]{0});

  sobel_edge_detection_seq.setTaskData(&task_data);

  EXPECT_TRUE(sobel_edge_detection_seq.validation());

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, ValidationTest_NullInputOrOutputBuffer) {
  SobelEdgeDetectionSeq sobel_edge_detection_seq;

  SobelEdgeDetectionSeq::TaskData task_data;
  task_data.width = 5;
  task_data.height = 5;
  task_data.inputs.push_back(nullptr);
  task_data.outputs.push_back(new uint8_t[25]{0});

  sobel_edge_detection_seq.setTaskData(&task_data);

  EXPECT_FALSE(sobel_edge_detection_seq.validation());

  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, ValidationTest_MultipleInputsOrOutputs) {
  SobelEdgeDetectionSeq sobel_edge_detection_seq;

  SobelEdgeDetectionSeq::TaskData task_data;
  task_data.width = 5;
  task_data.height = 5;
  task_data.inputs_count.push_back(25);
  task_data.outputs_count.push_back(25);

  task_data.inputs.push_back(new uint8_t[25]{0});
  task_data.outputs.push_back(new uint8_t[25]{0});
  task_data.inputs.push_back(new uint8_t[25]{0});

  sobel_edge_detection_seq.setTaskData(&task_data);

  EXPECT_FALSE(sobel_edge_detection_seq.validation());

  delete[] task_data.inputs[0];
  delete[] task_data.inputs[1];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, PreProcessingInvalidData) {
  SobelEdgeDetectionSeq sobel_edge_detection_seq;

  SobelEdgeDetectionSeq::TaskData task_data;
  task_data.width = 5;
  task_data.height = 5;

  sobel_edge_detection_seq.setTaskData(&task_data);

  EXPECT_FALSE(sobel_edge_detection_seq.pre_processing(&task_data));
}

TEST(mezhuev_m_sobel_edge_detection, PreProcessingMismatchedInputsAndOutputs) {
  SobelEdgeDetectionSeq sobel_edge_detection_seq;

  SobelEdgeDetectionSeq::TaskData task_data;
  task_data.width = 5;
  task_data.height = 5;
  task_data.inputs_count.push_back(25);
  task_data.outputs_count.push_back(24);

  task_data.inputs.push_back(new uint8_t[25]{0});
  task_data.outputs.push_back(new uint8_t[25]{0});

  sobel_edge_detection_seq.setTaskData(&task_data);

  EXPECT_FALSE(sobel_edge_detection_seq.pre_processing(&task_data));

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, PreProcessingNullInputOrOutputBuffer) {
  SobelEdgeDetectionSeq sobel_edge_detection_seq;

  SobelEdgeDetectionSeq::TaskData task_data;
  task_data.width = 5;
  task_data.height = 5;
  task_data.inputs.push_back(nullptr);
  task_data.outputs.push_back(new uint8_t[25]{0});

  sobel_edge_detection_seq.setTaskData(&task_data);

  EXPECT_FALSE(sobel_edge_detection_seq.pre_processing(&task_data));

  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, PreProcessingMultipleInputsOrOutputs) {
  SobelEdgeDetectionSeq sobel_edge_detection_seq;

  SobelEdgeDetectionSeq::TaskData task_data;
  task_data.width = 5;
  task_data.height = 5;
  task_data.inputs_count.push_back(25);
  task_data.outputs_count.push_back(25);

  task_data.inputs.push_back(new uint8_t[25]{0});
  task_data.outputs.push_back(new uint8_t[25]{0});
  task_data.inputs.push_back(new uint8_t[25]{0});

  sobel_edge_detection_seq.setTaskData(&task_data);

  EXPECT_FALSE(sobel_edge_detection_seq.pre_processing(&task_data));

  delete[] task_data.inputs[0];
  delete[] task_data.inputs[1];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, RunNullInputOrOutputBuffer) {
  SobelEdgeDetectionSeq sobel_edge_detection_seq;

  SobelEdgeDetectionSeq::TaskData task_data;
  task_data.width = 3;
  task_data.height = 3;
  task_data.inputs.push_back(nullptr);
  task_data.outputs.push_back(nullptr);

  sobel_edge_detection_seq.setTaskData(&task_data);

  EXPECT_FALSE(sobel_edge_detection_seq.run());
}

TEST(mezhuev_m_sobel_edge_detection, PostProcessingInvalidValue) {
  SobelEdgeDetectionSeq sobel_edge_detection_seq;

  SobelEdgeDetectionSeq::TaskData task_data;
  task_data.width = 5;
  task_data.height = 5;
  task_data.inputs_count.push_back(25);
  task_data.outputs_count.push_back(25);

  task_data.inputs.push_back(
      new uint8_t[25]{0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0});
  task_data.outputs.push_back(new uint8_t[25]{0});

  sobel_edge_detection_seq.setTaskData(&task_data);

  EXPECT_FALSE(sobel_edge_detection_seq.post_processing());

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, PostProcessingNullOutputBuffer) {
  SobelEdgeDetectionSeq sobel_edge_detection_seq;

  SobelEdgeDetectionSeq::TaskData task_data;
  task_data.width = 5;
  task_data.height = 5;
  task_data.inputs_count.push_back(25);
  task_data.outputs_count.push_back(25);

  task_data.inputs.push_back(
      new uint8_t[25]{0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0});
  task_data.outputs.push_back(nullptr);

  sobel_edge_detection_seq.setTaskData(&task_data);

  EXPECT_FALSE(sobel_edge_detection_seq.post_processing());

  delete[] task_data.inputs[0];
}

}  // namespace mezhuev_m_sobel_edge_detection