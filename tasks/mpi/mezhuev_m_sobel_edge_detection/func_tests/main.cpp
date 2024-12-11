#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <iostream>
#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

#include "mpi/mezhuev_m_sobel_edge_detection/include/mpi.hpp"

namespace mezhuev_m_sobel_edge_detection {

TEST(mezhuev_m_sobel_edge_detection, ValidData) {
  boost::mpi::communicator world;
  GridTorusTopologyParallel grid_topology(world);

  TaskData task_data;
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

TEST(mezhuev_m_sobel_edge_detection, NullInputBuffer) {
  boost::mpi::communicator world;
  GridTorusTopologyParallel grid_topology(world);

  TaskData task_data;
  task_data.inputs.push_back(nullptr);
  task_data.outputs.push_back(new uint8_t[100]{0});

  grid_topology.pre_processing();

  EXPECT_EQ(task_data.inputs[0], nullptr);

  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, PostProcessingZeroOutput) {
  boost::mpi::communicator world;
  GridTorusTopologyParallel grid_topology(world);

  TaskData task_data;
  task_data.outputs_count.push_back(100);
  task_data.outputs.push_back(new uint8_t[100]{0});

  EXPECT_FALSE(grid_topology.post_processing());

  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, MemoryDeallocation) {
  boost::mpi::communicator world;
  GridTorusTopologyParallel grid_topology(world);

  TaskData task_data;
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

TEST(mezhuev_m_sobel_edge_detection, BoundaryConditions) {
  boost::mpi::communicator world;
  GridTorusTopologyParallel grid_topology(world);

  TaskData task_data;
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

TEST(mezhuev_m_sobel_edge_detection, IncorrectOutputBufferCount) {
  boost::mpi::communicator world;
  GridTorusTopologyParallel grid_topology(world);

  TaskData task_data;
  task_data.inputs_count.push_back(100);
  task_data.outputs_count.push_back(0);
  task_data.inputs.push_back(new uint8_t[100]{0});

  grid_topology.pre_processing();

  EXPECT_EQ(task_data.outputs.size(), static_cast<size_t>(0));

  delete[] task_data.inputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, MismatchedInputOutputSizes) {
  boost::mpi::communicator world;
  GridTorusTopologyParallel grid_topology(world);

  TaskData task_data;
  task_data.inputs_count.push_back(100);
  task_data.outputs_count.push_back(50);
  task_data.inputs.push_back(new uint8_t[100]{0});
  task_data.outputs.push_back(new uint8_t[50]{0});

  grid_topology.pre_processing();

  EXPECT_NE(task_data.inputs[0], nullptr);
  EXPECT_NE(task_data.outputs[0], nullptr);

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, DoubleMemoryDeallocation) {
  boost::mpi::communicator world;
  GridTorusTopologyParallel grid_topology(world);

  TaskData task_data;
  task_data.inputs_count.push_back(100);
  task_data.outputs_count.push_back(100);
  task_data.inputs.push_back(new uint8_t[100]{0});
  task_data.outputs.push_back(new uint8_t[100]{0});

  grid_topology.pre_processing();

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];

  task_data.inputs.clear();
  task_data.outputs.clear();

  EXPECT_EQ(task_data.inputs.size(), static_cast<size_t>(0));
  EXPECT_EQ(task_data.outputs.size(), static_cast<size_t>(0));
}

TEST(mezhuev_m_sobel_edge_detection, MPIDistribution) {
  boost::mpi::communicator world;
  GridTorusTopologyParallel grid_topology(world);

  TaskData task_data;
  task_data.inputs_count.push_back(100);
  task_data.outputs_count.push_back(100);
  task_data.inputs.push_back(new uint8_t[100]{0});
  task_data.outputs.push_back(new uint8_t[100]{0});

  grid_topology.pre_processing();

  int rank = world.rank();
  int size = world.size();

  if (size > 1) {
    EXPECT_GE(rank, 0);
    EXPECT_LT(rank, size);
  }

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, LargeDataSetHandling) {
  boost::mpi::communicator world;
  GridTorusTopologyParallel grid_topology(world);

  TaskData task_data;
  size_t large_size = 100'000;
  task_data.inputs_count.push_back(large_size);
  task_data.outputs_count.push_back(large_size);
  task_data.inputs.push_back(new uint8_t[large_size]{0});
  task_data.outputs.push_back(new uint8_t[large_size]{0});

  EXPECT_NO_THROW(grid_topology.pre_processing());

  EXPECT_NE(task_data.inputs[0], nullptr);
  EXPECT_NE(task_data.outputs[0], nullptr);

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, TaskDataWithoutOutputBuffer) {
  boost::mpi::communicator world;
  GridTorusTopologyParallel grid_topology(world);

  TaskData task_data;
  task_data.inputs_count.push_back(100);
  task_data.outputs_count.push_back(0);
  task_data.inputs.push_back(new uint8_t[100]{0});

  grid_topology.pre_processing();

  EXPECT_EQ(task_data.outputs.size(), static_cast<size_t>(0));
  EXPECT_EQ(task_data.inputs.size(), static_cast<size_t>(1));

  delete[] task_data.inputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, IncorrectBufferSizes) {
  boost::mpi::communicator world;
  GridTorusTopologyParallel grid_topology(world);

  TaskData task_data;
  task_data.inputs_count.push_back(50);
  task_data.outputs_count.push_back(100);
  task_data.inputs.push_back(new uint8_t[50]{0});
  task_data.outputs.push_back(new uint8_t[100]{0});

  grid_topology.pre_processing();

  EXPECT_NE(task_data.inputs[0], nullptr);
  EXPECT_NE(task_data.outputs[0], nullptr);

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, RandomDataProcessing) {
  boost::mpi::communicator world;
  GridTorusTopologyParallel grid_topology(world);

  TaskData task_data;
  size_t data_size = 100;
  task_data.inputs_count.push_back(data_size);
  task_data.outputs_count.push_back(data_size);

  auto* input_data = new uint8_t[data_size];
  auto* output_data = new uint8_t[data_size]{0};

  std::generate(input_data, input_data + data_size, []() { return std::rand() % 256; });

  task_data.inputs.push_back(input_data);
  task_data.outputs.push_back(output_data);

  grid_topology.pre_processing();

  EXPECT_NE(task_data.inputs[0], nullptr);
  EXPECT_NE(task_data.outputs[0], nullptr);

  delete[] input_data;
  delete[] output_data;
}

TEST(mezhuev_m_sobel_edge_detection, EmptyInputData) {
  boost::mpi::communicator world;
  GridTorusTopologyParallel grid_topology(world);

  TaskData task_data;
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
  GridTorusTopologyParallel grid_topology(world);

  TaskData task_data;
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

TEST(mezhuev_m_sobel_edge_detection, MPISynchronization) {
  boost::mpi::communicator world;
  GridTorusTopologyParallel grid_topology(world);

  TaskData task_data;
  size_t data_size = 100;
  task_data.inputs_count.push_back(data_size);
  task_data.outputs_count.push_back(data_size);
  task_data.inputs.push_back(new uint8_t[data_size]{0});
  task_data.outputs.push_back(new uint8_t[data_size]{0});

  grid_topology.pre_processing();

  if (world.size() > 1) {
    int rank = world.rank();
    EXPECT_GE(rank, 0);
    EXPECT_LT(rank, world.size());
  }

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

}  // namespace mezhuev_m_sobel_edge_detection