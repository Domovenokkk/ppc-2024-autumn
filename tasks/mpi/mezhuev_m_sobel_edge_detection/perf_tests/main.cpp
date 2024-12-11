#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <chrono>
#include <iostream>

#include "mpi/mezhuev_m_sobel_edge_detection/include/mpi.hpp"

namespace mezhuev_m_sobel_edge_detection {

TEST(mezhuev_m_sobel_edge_detection, pre_processing_large_dSata) {
  boost::mpi::communicator world;
  GridTorusTopologyParallel grid_topology(world);

  TaskData task_data;
  size_t large_size = 10'000'000;
  task_data.inputs_count.push_back(large_size);
  task_data.outputs_count.push_back(large_size);
  task_data.inputs.push_back(nullptr);
  task_data.outputs.push_back(nullptr);

  grid_topology.setTaskData(&task_data);

  auto start = std::chrono::high_resolution_clock::now();
  bool result = grid_topology.pre_processing();
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = end - start;
  if (world.rank() == 0) {
    std::cout << "Pre-processing large data completed in " << duration.count() << " seconds." << std::endl;
  }

  EXPECT_TRUE(result);
  EXPECT_NE(task_data.inputs[0], nullptr);
  EXPECT_NE(task_data.outputs[0], nullptr);

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, validation_large_data) {
  boost::mpi::communicator world;
  GridTorusTopologyParallel grid_topology(world);

  TaskData task_data;
  size_t large_size = 10'000'000;
  task_data.inputs_count.push_back(large_size);
  task_data.outputs_count.push_back(large_size);
  task_data.inputs.push_back(new uint8_t[large_size]);
  task_data.outputs.push_back(new uint8_t[large_size]);

  grid_topology.setTaskData(&task_data);

  auto start = std::chrono::high_resolution_clock::now();
  bool result = grid_topology.validation();
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = end - start;
  if (world.rank() == 0) {
    std::cout << "Validation completed in " << duration.count() << " seconds." << std::endl;
  }

  EXPECT_TRUE(result);

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, run_large_data) {
  boost::mpi::communicator world;
  GridTorusTopologyParallel grid_topology(world);

  TaskData task_data;
  size_t large_size = 10'000;
  task_data.inputs_count.push_back(large_size);
  task_data.outputs_count.push_back(large_size);
  task_data.inputs.push_back(new uint8_t[large_size]{1});
  task_data.outputs.push_back(new uint8_t[large_size]{0});

  grid_topology.setTaskData(&task_data);

  auto start = std::chrono::high_resolution_clock::now();
  bool result = grid_topology.run();
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = end - start;
  if (world.rank() == 0) {
    std::cout << "Run completed in " << duration.count() << " seconds." << std::endl;
  }

  EXPECT_TRUE(result);
  EXPECT_NE(task_data.outputs[0][0], 0);

  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, post_processing_large_data) {
  boost::mpi::communicator world;
  GridTorusTopologyParallel grid_topology(world);

  TaskData task_data;
  size_t large_size = 10'000'000;
  task_data.outputs_count.push_back(large_size);
  task_data.outputs.push_back(new uint8_t[large_size]);

  for (size_t i = 0; i < large_size; ++i) {
    task_data.outputs[0][i] = 1;
  }

  grid_topology.setTaskData(&task_data);

  auto start = std::chrono::high_resolution_clock::now();
  bool result = grid_topology.post_processing();
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = end - start;
  if (world.rank() == 0) {
    std::cout << "Post-processing completed in " << duration.count() << " seconds." << std::endl;
  }

  EXPECT_TRUE(result);

  delete[] task_data.outputs[0];
}

}  // namespace mezhuev_m_sobel_edge_detection
