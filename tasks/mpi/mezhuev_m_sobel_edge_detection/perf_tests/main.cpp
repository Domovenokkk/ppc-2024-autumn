#include <gtest/gtest.h>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <chrono>
#include <vector>
#include <iostream>

#include "mpi/mezhuev_m_sobel_edge_detection/include/mpi.hpp"

namespace mezhuev_m_sobel_edge_detection {

TEST(mezhuev_m_sobel_edge_detection_perf, PreProcessingLargeData) {
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
    grid_topology.pre_processing();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    if (world.rank() == 0) {
        std::cout << "Pre-processing large data completed in " << duration.count() << " seconds." << std::endl;
    }

    EXPECT_NE(task_data.inputs[0], nullptr);
    EXPECT_NE(task_data.outputs[0], nullptr);

    delete[] task_data.inputs[0];
    delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection_perf, RunScalingProcesses) {
    boost::mpi::communicator world;
    GridTorusTopologyParallel grid_topology(world);

    TaskData task_data;
    size_t data_size = 1'000;
    task_data.inputs_count.push_back(data_size);
    task_data.outputs_count.push_back(data_size);
    task_data.inputs.push_back(new uint8_t[data_size]{1});
    task_data.outputs.push_back(new uint8_t[data_size]{0});

    grid_topology.setTaskData(&task_data);

    auto start = std::chrono::high_resolution_clock::now();
    grid_topology.run();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    if (world.rank() == 0) {
        std::cout << "Run completed in " << duration.count() << " seconds with " << world.size() << " processes." << std::endl;
    }

    EXPECT_NE(task_data.outputs[0][0], 0);

    delete[] task_data.inputs[0];
    delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection_perf, PostProcessingLargeData) {
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
        std::cout << "Post-processing large data completed in " << duration.count() << " seconds." << std::endl;
    }

    EXPECT_TRUE(result);

    delete[] task_data.outputs[0];
}

TEST(mezhuev_m_sobel_edge_detection_perf, GridSizeScaling) {
    boost::mpi::communicator world;
    GridTorusTopologyParallel grid_topology(world);

    TaskData task_data;
    size_t base_size = 1'000;
    size_t grid_factor = world.size();
    task_data.inputs_count.push_back(base_size * grid_factor);
    task_data.outputs_count.push_back(base_size * grid_factor);
    task_data.inputs.push_back(new uint8_t[base_size * grid_factor]{1});
    task_data.outputs.push_back(new uint8_t[base_size * grid_factor]{0});

    grid_topology.setTaskData(&task_data);

    auto start = std::chrono::high_resolution_clock::now();
    grid_topology.run();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    if (world.rank() == 0) {
        std::cout << "Run with grid scaling completed in " << duration.count() << " seconds for grid size " << grid_factor << "." << std::endl;
    }

    EXPECT_NE(task_data.outputs[0][0], 0);

    delete[] task_data.inputs[0];
    delete[] task_data.outputs[0];
}

}  // namespace mezhuev_m_sobel_edge_detection
