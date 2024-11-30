#include "mpi/mezhuev_m_lattice_torus/include/mpi.hpp"

#include <boost/mpi.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace mezhuev_m_lattice_torus {

bool GridTorusTopologyParallel::pre_processing() {
  return validation();
}

bool GridTorusTopologyParallel::validation() {
  if (taskData->inputs.empty() || taskData->inputs_count.empty()) {
    return false;
  }

  for (size_t i = 0; i < taskData->inputs.size(); ++i) {
    if (taskData->inputs_count[i] <= 0 || taskData->inputs[i] == nullptr) {
      return false;
    }
  }

  if (taskData->inputs_count[0] != taskData->outputs_count[0]) {
    return false;
  }

  int size = boost::mpi::communicator().size();
  int grid_dim = static_cast<int>(std::sqrt(size));
  return grid_dim * grid_dim == size;

  return true;
}

bool GridTorusTopologyParallel::run() {
  int rank = world.rank();
  int size = world.size();
  int grid_dim = std::sqrt(size);

  world.barrier();

  auto compute_neighbors = [grid_dim, size](int rank) -> std::vector<int> {
    int x = rank % grid_dim;
    int y = rank / grid_dim;

    int left = (x - 1 + grid_dim) % grid_dim + y * grid_dim;
    int right = (x + 1) % grid_dim + y * grid_dim;
    int up = x + ((y - 1 + grid_dim) % grid_dim) * grid_dim;
    int down = x + ((y + 1) % grid_dim) * grid_dim;

    return {left, right, up, down};
  };

  auto neighbors = compute_neighbors(rank);
  std::vector<uint8_t> send_buffer(taskData->inputs_count[0]);
  std::copy(taskData->inputs[0], taskData->inputs[0] + taskData->inputs_count[0], send_buffer.begin());

  std::vector<uint8_t> combined_buffer;
  combined_buffer.reserve(taskData->inputs_count[0] * neighbors.size());

  for (int neighbor : neighbors) {
    try {

      world.send(neighbor, 0, send_buffer);

      std::vector<uint8_t> recv_buffer(taskData->inputs_count[0]);
      world.recv(neighbor, 0, recv_buffer);

      combined_buffer.insert(combined_buffer.end(), recv_buffer.begin(), recv_buffer.end());
    } catch (const boost::mpi::exception& ex) {
      std::cerr << "Error communicating with neighbor " << neighbor << ": " << ex.what() << std::endl;
      return false;
    }
  }

  if (taskData->outputs_count[0] >= combined_buffer.size()) {
    std::copy(combined_buffer.begin(), combined_buffer.end(), taskData->outputs[0]);
  } else {
    std::cerr << "Output buffer is too small to hold received data!" << std::endl;
    return false;
  }

  world.barrier();
  return true;
}


bool GridTorusTopologyParallel::post_processing() {
  return true;
}

}  // namespace mezhuev_m_lattice_torus