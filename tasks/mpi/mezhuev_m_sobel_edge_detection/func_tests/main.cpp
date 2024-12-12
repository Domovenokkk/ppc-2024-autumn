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

TEST(mezhuev_m_sobel_edge_detection, PreProcessingValidData) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::GridTorusTopologyParallel grid_topology(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs_count.push_back(100);
  task_data.outputs_count.push_back(100);
  task_data.inputs.push_back(new uint8_t[100]{0});
  task_data.outputs.push_back(new uint8_t[100]{0});

  grid_topology.pre_processing();

  EXPECT_NE(task_data.inputs[0], nullptr);
  EXPECT_NE(task_data.outputs[0], nullptr);

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, PreProcessingInvalidData) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::GridTorusTopologyParallel grid_topology(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs_count.push_back(100);
  task_data.outputs_count.push_back(100);
  task_data.inputs.push_back(nullptr);
  task_data.outputs.push_back(new uint8_t[100]{0});

  EXPECT_FALSE(grid_topology.pre_processing());

  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, PostProcessingInvalidData) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::GridTorusTopologyParallel grid_topology(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.outputs_count.push_back(100);
  task_data.outputs.push_back(new uint8_t[100]{0});

  EXPECT_FALSE(grid_topology.post_processing());

  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, PostProcessingWithoutOutputs) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::GridTorusTopologyParallel grid_topology(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.outputs_count.push_back(0);

  EXPECT_FALSE(grid_topology.post_processing());
}

TEST(mezhuev_m_sobel_edge_detection, PostProcessingWithNullData) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::GridTorusTopologyParallel grid_topology(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.outputs_count.push_back(100);
  task_data.outputs.push_back(nullptr);

  EXPECT_FALSE(grid_topology.post_processing());
}

TEST(mezhuev_m_sobel_edge_detection, PostProcessingNullOutputBuffer) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::GridTorusTopologyParallel grid_topology(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.outputs_count.push_back(100);
  task_data.outputs.push_back(nullptr);

  grid_topology.pre_processing();

  EXPECT_FALSE(grid_topology.post_processing());
}

TEST(mezhuev_m_sobel_edge_detection, PostProcessingZeroInOutput) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::GridTorusTopologyParallel grid_topology(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.outputs_count.push_back(100);
  uint8_t* output_data = new uint8_t[100]{255};
  output_data[10] = 0;
  task_data.outputs.push_back(output_data);

  grid_topology.pre_processing();

  EXPECT_FALSE(grid_topology.post_processing());

  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, PostProcessingEmptyOutput) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::GridTorusTopologyParallel grid_topology(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.outputs_count.push_back(0);
  task_data.outputs.push_back(new uint8_t[0]);

  grid_topology.pre_processing();

  EXPECT_FALSE(grid_topology.post_processing());

  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, post_processing_zero_output) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::GridTorusTopologyParallel grid_topology(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.outputs_count.push_back(100);
  task_data.outputs.push_back(new uint8_t[100]{0});

  EXPECT_FALSE(grid_topology.post_processing());  // Ожидаем, что не произойдет завершение

  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, memory_deallocation) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::GridTorusTopologyParallel grid_topology(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs_count.push_back(100);
  task_data.outputs_count.push_back(100);
  task_data.inputs.push_back(new uint8_t[100]{0});
  task_data.outputs.push_back(new uint8_t[100]{0});

  grid_topology.pre_processing();

  for (size_t i = 0; i < task_data.inputs.size(); ++i) {
    delete[] task_data.inputs[i];
  }
  for (size_t i = 0; i < task_data.outputs.size(); ++i) {
    delete[] task_data.outputs[i];
  }

  task_data.inputs.clear();
  task_data.outputs.clear();

  EXPECT_EQ(task_data.inputs.size(), static_cast<size_t>(0));
  EXPECT_EQ(task_data.outputs.size(), static_cast<size_t>(0));
}

TEST(mezhuev_m_sobel_edge_detection, boundary_conditions) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::GridTorusTopologyParallel grid_topology(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs_count.push_back(1);
  task_data.outputs_count.push_back(1);
  task_data.inputs.push_back(new uint8_t[1]{128});
  task_data.outputs.push_back(new uint8_t[1]{0});

  grid_topology.pre_processing();

  EXPECT_EQ(task_data.inputs[0][0], 128);
  EXPECT_EQ(task_data.outputs[0][0], 0);

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, incorrect_output_buffer_count) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::GridTorusTopologyParallel grid_topology(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs_count.push_back(100);
  task_data.outputs_count.push_back(0);
  task_data.inputs.push_back(new uint8_t[100]{0});

  grid_topology.pre_processing();

  EXPECT_EQ(task_data.outputs.size(), static_cast<size_t>(0));

  delete[] task_data.inputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, EmptyInputData) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::GridTorusTopologyParallel grid_topology(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs_count.push_back(0);
  task_data.outputs_count.push_back(100);
  task_data.outputs.push_back(new uint8_t[100]{0});

  grid_topology.pre_processing();

  EXPECT_EQ(task_data.inputs.size(), static_cast<size_t>(0));
  EXPECT_NE(task_data.outputs[0], nullptr);

  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, SinglePixelProcessing) {
  boost::mpi::communicator world;
  mezhuev_m_sobel_edge_detection::GridTorusTopologyParallel grid_topology(world);

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs_count.push_back(1);
  task_data.outputs_count.push_back(1);
  task_data.inputs.push_back(new uint8_t[1]{255});
  task_data.outputs.push_back(new uint8_t[1]{0});

  grid_topology.pre_processing();

  EXPECT_EQ(task_data.inputs[0][0], 255);
  EXPECT_EQ(task_data.outputs[0][0], 0);

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}