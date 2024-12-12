#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace mezhuev_m_sobel_edge_detection {

struct TaskData {
  std::vector<uint8_t*> inputs;
  std::vector<size_t> inputs_count;
  std::vector<uint8_t*> outputs;
  std::vector<size_t> outputs_count;
};

class GridTorusTopologyParallel {
 public:
  GridTorusTopologyParallel(boost::mpi::communicator& comm) : world(comm) {}

  bool pre_processing();
  bool validation();
  bool run(TaskData& taskData);
  bool post_processing();

 private:
  boost::mpi::communicator& world;
  TaskData* taskData = nullptr;
};

}  // namespace mezhuev_m_sobel_edge_detection