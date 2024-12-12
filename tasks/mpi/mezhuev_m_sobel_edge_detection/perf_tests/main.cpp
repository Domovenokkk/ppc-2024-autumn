#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <chrono>

#include "mpi/mezhuev_m_sobel_edge_detection/include/ops_mpi.hpp"

void prepareTaskData(TaskData& task_data, size_t width, size_t height) {
  task_data.width = width;
  task_data.height = height;
  task_data.inputs_count.push_back(width * height);
  task_data.outputs_count.push_back(width * height);
  task_data.inputs.push_back(new uint8_t[width * height]());
  task_data.outputs.push_back(new uint8_t[width * height]());
}

void cleanupTaskData(TaskData& task_data) {
  delete[] task_data.inputs[0];
  delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection, SobelEdgeDetectionTime) {
  mpi::communicator world;
  int rank = world.rank();
  int size = world.size();

  const std::vector<size_t> widths = {256, 512, 1024, 2048};
  const std::vector<size_t> heights = {256, 512, 1024, 2048};

  for (size_t width : widths) {
    for (size_t height : heights) {
      if (rank == 0) {
        std::cout << "Running performance test with image size: " << width << "x" << height << std::endl;
      }

      mezhuev_m_sobel_edge_detection::TaskData task_data;
      prepareTaskData(task_data, width, height);

      mezhuev_m_sobel_edge_detection::SobelEdgeDetectionMPI sobel(world);
      if (!sobel.pre_processing(&task_data)) {
        cleanupTaskData(task_data);
        return;
      }

      auto start = std::chrono::high_resolution_clock::now();
      if (!sobel.run()) {
        cleanupTaskData(task_data);
        return;
      }
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration = end - start;

      if (rank == 0) {
        std::cout << "Edge detection time for image size " << width << "x" << height
                  << " with " << size << " processes: " << duration.count() << " seconds." << std::endl;
      }

      cleanupTaskData(task_data);

      world.barrier();
    }
  }
}