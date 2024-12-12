#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <chrono>

#include "mpi/mezhuev_m_sobel_edge_detection/include/ops_mpi.hpp"

TEST(mezhuev_m_sobel_edge_detection, RunPerformance) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  size_t width = 1920;
  size_t height = 1080;

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs.push_back(new uint8_t[width * height]());
  task_data.inputs_count.push_back(width * height);
  task_data.outputs.push_back(new uint8_t[width * height]());
  task_data.outputs_count.push_back(width * height);

  mezhuev_m_sobel_edge_detection::GridTorusTopologyParallel grid_topology(world);

  ASSERT_TRUE(grid_topology.pre_processing());
  ASSERT_TRUE(grid_topology.validation());

  auto start = std::chrono::high_resolution_clock::now();
  ASSERT_TRUE(grid_topology.run(task_data));
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Run duration: " << duration.count() << " ms" << std::endl;

  ASSERT_TRUE(grid_topology.post_processing());

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, PreProcessingPerformance) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  size_t width = 1920;
  size_t height = 1080;

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs.push_back(new uint8_t[width * height]());
  task_data.inputs_count.push_back(width * height);
  task_data.outputs.push_back(new uint8_t[width * height]());
  task_data.outputs_count.push_back(width * height);

  mezhuev_m_sobel_edge_detection::GridTorusTopologyParallel grid_topology(world);

  auto start = std::chrono::high_resolution_clock::now();
  ASSERT_TRUE(grid_topology.pre_processing());
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Pre-processing duration: " << duration.count() << " ms" << std::endl;

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, ValidationPerformance) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  size_t width = 1920;
  size_t height = 1080;

  mezhuev_m_sobel_edge_detection::TaskData task_data;
  task_data.inputs.push_back(new uint8_t[width * height]());
  task_data.inputs_count.push_back(width * height);
  task_data.outputs.push_back(new uint8_t[width * height]());
  task_data.outputs_count.push_back(width * height);

  mezhuev_m_sobel_edge_detection::GridTorusTopologyParallel grid_topology(world);

  ASSERT_TRUE(grid_topology.pre_processing());

  auto start = std::chrono::high_resolution_clock::now();
  ASSERT_TRUE(grid_topology.validation());
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Validation duration: " << duration.count() << " ms" << std::endl;

  ASSERT_TRUE(grid_topology.post_processing());

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}