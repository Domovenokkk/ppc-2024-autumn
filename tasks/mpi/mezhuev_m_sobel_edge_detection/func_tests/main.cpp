#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "mpi/mezhuev_m_sobel_edge_detection/include/ops_mpi.hpp"

TEST(mezhuev_m_sobel_edge_detection, PreProcessingInvalidData) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs.push_back(nullptr);
  task_data.outputs.push_back(new uint8_t[10000]{0});
  task_data.inputs_count.push_back(1);
  task_data.outputs_count.push_back(1);

  EXPECT_FALSE(sobel_edge_detection.pre_processing(&task_data));

  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, PostProcessingInvalidData) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.outputs.push_back(nullptr);
  task_data.outputs_count.push_back(1);

  EXPECT_FALSE(sobel_edge_detection.post_processing());
}

TEST(mezhuev_m_sobel_edge_detection, ValidationWithEmptyData) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs.push_back(nullptr);
  task_data.outputs.push_back(nullptr);
  task_data.inputs_count.push_back(0);
  task_data.outputs_count.push_back(0);

  EXPECT_FALSE(sobel_edge_detection.validation());
}

TEST(mezhuev_m_sobel_edge_detection, PostProcessingZeroOutput) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.outputs.push_back(new uint8_t[10000]{0});
  task_data.outputs_count.push_back(10000);

  EXPECT_FALSE(sobel_edge_detection.post_processing());

  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, PreProcessingInvalidInputCount) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs.push_back(new uint8_t[10000]{0});
  task_data.outputs.push_back(new uint8_t[10000]{0});
  task_data.inputs_count.push_back(1);
  task_data.outputs_count.push_back(10000);

  EXPECT_FALSE(sobel_edge_detection.pre_processing(&task_data));

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, PreProcessingInvalidDimensions) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs.push_back(new uint8_t[10000]{0});
  task_data.outputs.push_back(new uint8_t[10000]{0});
  task_data.inputs_count.push_back(10000);
  task_data.outputs_count.push_back(10000);
  task_data.width = 0;
  task_data.height = 100;

  EXPECT_FALSE(sobel_edge_detection.pre_processing(&task_data));

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, PreProcessingEmptyImage) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs.push_back(new uint8_t[0]);
  task_data.outputs.push_back(new uint8_t[0]);
  task_data.inputs_count.push_back(0);
  task_data.outputs_count.push_back(0);
  task_data.width = 0;
  task_data.height = 0;

  EXPECT_FALSE(sobel_edge_detection.pre_processing(&task_data));

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, PreProcessingEmptyInputAndOutput) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs.push_back(nullptr);
  task_data.outputs.push_back(nullptr);
  task_data.inputs_count.push_back(0);
  task_data.outputs_count.push_back(0);
  task_data.width = 0;
  task_data.height = 0;

  EXPECT_FALSE(sobel_edge_detection.pre_processing(&task_data));
}

TEST(mezhuev_m_sobel_edge_detection, InvalidDataBetweenProcesses) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs.push_back(new uint8_t[10000]{0});
  task_data.outputs.push_back(new uint8_t[10000]{0});
  task_data.inputs_count.push_back(10000);
  task_data.outputs_count.push_back(5000);  // Ошибка в количестве данных для выходных данных
  task_data.width = 100;
  task_data.height = 100;

  EXPECT_FALSE(sobel_edge_detection.pre_processing(&task_data));

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, InvalidImageDimensions) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs.push_back(new uint8_t[10000]{0});
  task_data.outputs.push_back(new uint8_t[10000]{0});
  task_data.inputs_count.push_back(10000);
  task_data.outputs_count.push_back(10000);
  task_data.width = 1;  // Недопустимая ширина изображения
  task_data.height = 100;

  EXPECT_FALSE(sobel_edge_detection.pre_processing(&task_data));

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, NullPointerInputData) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs.push_back(nullptr);  // Пустой указатель на входные данные
  task_data.outputs.push_back(new uint8_t[10000]{0});
  task_data.inputs_count.push_back(10000);
  task_data.outputs_count.push_back(10000);

  EXPECT_FALSE(sobel_edge_detection.pre_processing(&task_data));

  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, NullPointerOutputData) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs.push_back(new uint8_t[10000]{0});
  task_data.outputs.push_back(nullptr);
  task_data.inputs_count.push_back(10000);
  task_data.outputs_count.push_back(10000);

  EXPECT_FALSE(sobel_edge_detection.pre_processing(&task_data));

  delete[] task_data.inputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, InvalidProcessingData) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs.push_back(new uint8_t[10000]{0});
  task_data.outputs.push_back(new uint8_t[10000]{0});
  task_data.inputs_count.push_back(10000);
  task_data.outputs_count.push_back(10000);
  task_data.width = 100;
  task_data.height = 100;

  std::fill(task_data.inputs[0], task_data.inputs[0] + 10000, 255);

  EXPECT_FALSE(sobel_edge_detection.run());

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, InvalidSizeRun) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.width = 0;
  task_data.height = 0;
  task_data.inputs.push_back(nullptr);
  task_data.outputs.push_back(nullptr);
  task_data.inputs_count.push_back(0);
  task_data.outputs_count.push_back(0);

  EXPECT_FALSE(sobel_edge_detection.run());
}

TEST(mezhuev_m_sobel_edge_detection, ValidRun) {
  boost::mpi::communicator world;

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.width = 5;
  task_data.height = 5;
  task_data.inputs.push_back(new uint8_t[25]{255});
  task_data.outputs.push_back(new uint8_t[25]{0});
  task_data.inputs_count.push_back(25);
  task_data.outputs_count.push_back(25);

  std::fill(task_data.inputs[0], task_data.inputs[0] + 25, 128);

  sobel_edge_detection.setTaskData(&task_data);

  std::cout << "Running Sobel edge detection..." << std::endl;
  bool result = sobel_edge_detection.run();
  std::cout << "Sobel edge detection completed with result: " << result << std::endl;

  EXPECT_TRUE(result);

  std::cout << "Output image values after processing:" << std::endl;
  for (size_t i = 0; i < 25; ++i) {
    std::cout << (int)task_data.outputs[0][i] << " ";
  }
  std::cout << std::endl;

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, TaskDataIsNull) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  bool result = sobel_edge_detection.run();
  EXPECT_FALSE(result);
}

TEST(mezhuev_m_sobel_edge_detection, CorrectProcessingWithMultipleProcesses) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.width = 5;
  task_data.height = 5;
  task_data.inputs.push_back(new uint8_t[25]{255});
  task_data.outputs.push_back(new uint8_t[25]{0});
  task_data.inputs_count.push_back(25);
  task_data.outputs_count.push_back(25);
  std::fill(task_data.inputs[0], task_data.inputs[0] + 25, 128);

  sobel_edge_detection.setTaskData(&task_data);

  bool result = sobel_edge_detection.run();
  EXPECT_TRUE(result);

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, CorrectEdgeDetection) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.width = 5;
  task_data.height = 5;
  task_data.inputs.push_back(new uint8_t[25]{255});
  task_data.outputs.push_back(new uint8_t[25]{0});
  task_data.inputs_count.push_back(25);
  task_data.outputs_count.push_back(25);
  std::fill(task_data.inputs[0], task_data.inputs[0] + 25, 128);

  sobel_edge_detection.setTaskData(&task_data);

  bool result = sobel_edge_detection.run();
  EXPECT_TRUE(result);

  uint8_t expected_output[25] = {0};
  for (int i = 0; i < 25; ++i) {
    EXPECT_EQ(task_data.outputs[0][i], expected_output[i]);
  }

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, MPIBarrierWorks) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.width = 5;
  task_data.height = 5;
  task_data.inputs.push_back(new uint8_t[25]{255});
  task_data.outputs.push_back(new uint8_t[25]{0});
  task_data.inputs_count.push_back(25);
  task_data.outputs_count.push_back(25);
  std::fill(task_data.inputs[0], task_data.inputs[0] + 25, 128);

  sobel_edge_detection.setTaskData(&task_data);

  bool result = sobel_edge_detection.run();
  EXPECT_TRUE(result);

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, SmallImage) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.width = 3;
  task_data.height = 3;
  task_data.inputs.push_back(new uint8_t[9]{255});
  task_data.outputs.push_back(new uint8_t[9]{0});
  task_data.inputs_count.push_back(9);
  task_data.outputs_count.push_back(9);

  std::fill(task_data.inputs[0], task_data.inputs[0] + 9, 128);

  sobel_edge_detection.setTaskData(&task_data);

  bool result = sobel_edge_detection.run();
  EXPECT_TRUE(result);

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, EmptyImage) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  size_t width = 5;
  size_t height = 5;

  std::vector<uint8_t> input_image(width * height, 0);
  std::vector<uint8_t> output_image(width * height, 0);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.width = width;
  task_data.height = height;
  task_data.inputs.push_back(input_image.data());
  task_data.outputs.push_back(output_image.data());

  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel(world);
  ASSERT_TRUE(sobel.setTaskData(&task_data));

  ASSERT_TRUE(sobel.run());

  for (size_t i = 0; i < output_image.size(); ++i) {
    EXPECT_EQ(output_image[i], 0);
  }
}

TEST(mezhuev_m_sobel_edge_detection, PostProcessingAllZeros) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.outputs.push_back(new uint8_t[10]{0});
  task_data.outputs_count.push_back(10);

  sobel_edge_detection.setTaskData(&task_data);

  EXPECT_FALSE(sobel_edge_detection.post_processing())
      << "Expected post_processing to return false for all-zero output.";

  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, PostProcessingEmptyOutput) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.outputs.push_back(new uint8_t[0]);
  task_data.outputs_count.push_back(0);

  sobel_edge_detection.setTaskData(&task_data);

  EXPECT_FALSE(sobel_edge_detection.post_processing()) << "Expected post_processing to return false for empty output.";

  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, PreProcessingInvalidTaskData) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  EXPECT_FALSE(sobel_edge_detection.pre_processing(nullptr)) << "Pre-processing should fail for null task data.";
}

TEST(mezhuev_m_sobel_edge_detection, PreProcessingZeroWidthHeight) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.width = 0;
  task_data.height = 0;
  task_data.inputs.push_back(new uint8_t[1]{1});
  task_data.outputs.push_back(nullptr);
  task_data.inputs_count.push_back(1);
  task_data.outputs_count.push_back(1);

  EXPECT_FALSE(sobel_edge_detection.pre_processing(&task_data))
      << "Pre-processing should fail for zero width and height.";

  delete[] task_data.inputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, ValidationInvalidInputOutputSize) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs.resize(2);
  task_data.outputs.resize(1);
  sobel_edge_detection.setTaskData(&task_data);

  EXPECT_FALSE(sobel_edge_detection.validation())
      << "Validation should fail when inputs.size() != 1 or outputs.size() != 1.";
}

TEST(mezhuev_m_sobel_edge_detection, ValidationNullInputOutput) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs.push_back(nullptr);
  task_data.outputs.push_back(new uint8_t[10]{0});
  sobel_edge_detection.setTaskData(&task_data);

  EXPECT_FALSE(sobel_edge_detection.validation()) << "Validation should fail when inputs[0] is nullptr.";

  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, ValidationEmptyInputOutputCount) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs.push_back(new uint8_t[10]{0});
  task_data.outputs.push_back(new uint8_t[10]{0});
  sobel_edge_detection.setTaskData(&task_data);

  EXPECT_FALSE(sobel_edge_detection.validation())
      << "Validation should fail when inputs_count or outputs_count is empty.";

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, ValidationMismatchedInputOutputCount) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs.push_back(new uint8_t[10]{0});
  task_data.outputs.push_back(new uint8_t[10]{0});
  task_data.inputs_count.push_back(10);
  task_data.outputs_count.push_back(5);

  sobel_edge_detection.setTaskData(&task_data);

  EXPECT_FALSE(sobel_edge_detection.validation()) << "Validation should fail when inputs_count[0] != outputs_count[0].";

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, ValidationValidTaskData) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel_edge_detection(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs.push_back(new uint8_t[10]{0});
  task_data.outputs.push_back(new uint8_t[10]{0});
  task_data.inputs_count.push_back(10);
  task_data.outputs_count.push_back(10);

  sobel_edge_detection.setTaskData(&task_data);

  EXPECT_TRUE(sobel_edge_detection.validation()) << "Validation should succeed with valid task data.";

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}