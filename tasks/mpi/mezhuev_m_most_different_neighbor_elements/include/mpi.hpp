#pragma once

#include <mpi.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/serialization/serialization.hpp>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace mezhuev_m_most_different_neighbor_elements_mpi {

template <typename DataType>
std::vector<DataType> getVector(int sizz);

template <typename DataType>
int findLargestNeighborDifferenceIndex(std::vector<DataType> vector);

template <typename DataType>
class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<DataType> input_;
  DataType res[2];
};

template <typename DataType>
class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<DataType> input_, local_input_;
  int local_input_size;
  DataType res[2];
  boost::mpi::communicator world;
};
}  // namespace mezhuev_m_most_different_neighbor_elements_mpi