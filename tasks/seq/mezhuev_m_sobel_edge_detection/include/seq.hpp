#include <cstdint>
#include <iostream>
#include <vector>

namespace mezhuev_m_sobel_edge_detection {

class SobelEdgeDetectionSeq {
public:
  struct TaskData {
    std::vector<uint8_t*> inputs;
    std::vector<size_t> inputs_count;
    std::vector<uint8_t*> outputs;
    std::vector<size_t> outputs_count;
    size_t width;
    size_t height;
  };

  SobelEdgeDetectionSeq() : taskData(nullptr) {}

  TaskData* getTaskData() { return taskData; }

  void setTaskData(TaskData* data) { taskData = data; }

  bool pre_processing(TaskData* task_data);
  bool run();
  bool validation();
  bool post_processing();

private:
  TaskData* taskData;
  std::vector<int16_t> gradient_x;
  std::vector<int16_t> gradient_y;
};

}  // namespace mezhuev_m_sobel_edge_detection