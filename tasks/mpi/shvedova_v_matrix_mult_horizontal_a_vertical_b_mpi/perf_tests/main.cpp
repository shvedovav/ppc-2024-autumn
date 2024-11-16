#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi/include/ops_mpi.hpp"

namespace shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi {

void matrix_multiply_square_result(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& result,
                                   int rows_a, int cols_b) {
  result.resize(rows_a * cols_b, 0);

  for (int i = 0; i < rows_a; ++i) {
    for (int j = 0; j < cols_b; ++j) {
      int sum = 0;
      for (int k = 0; k < cols_b; ++k) {
        sum += a[i * cols_b + k] * b[k * rows_a + j];
      }
      result[i * cols_b + j] = sum;
    }
  }
}

std::vector<int> getRandomMatrix(int rows, int cols) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> dist(0, 1000);
  std::vector<int> matrix(rows * cols);
  for (int i = 0; i < rows * cols; i++) {
    matrix[i] = dist(gen);
  }
  return matrix;
}

}  // namespace shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi

TEST(svedova_v_matrix_mult_horizontal_a_vertical_b_mpi, pipeline_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> global_matrix_a;
  std::vector<int> global_matrix_b;
  std::vector<int> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int rowA = 100;
  int colA = 100;
  int colB = 100;

  if (world.rank() == 0) {

    global_matrix_a = shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::getRandomMatrix(rowA, colA);
    global_matrix_b = shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::getRandomMatrix(colA, colB);

    global_result.resize(rowA * colB, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix_a.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix_a.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix_b.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix_b.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rowA));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&colA));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&colB));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel =
      std::make_shared<shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::MatrixMultiplicationTaskParallel>(
          taskDataPar);
  ASSERT_EQ(taskParallel->validation(), true);
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    std::vector<int> expected_result;
    shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::matrix_multiply_square_result(global_matrix_a, global_matrix_b,
                                                                                      expected_result, rowA, colB);
    ASSERT_EQ(global_result, expected_result);
  }
}

TEST(svedova_v_matrix_mult_horizontal_a_vertical_b_mpi, task_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> global_matrix_a;
  std::vector<int> global_matrix_b;
  std::vector<int> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int rowA = 100;
  int colA = 100;
  int colB = 100;

  if (world.rank() == 0) {

    global_matrix_a = shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::getRandomMatrix(rowA, colA);
    global_matrix_b = shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::getRandomMatrix(colA, colB);

    global_result.resize(rowA * colB, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix_a.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix_a.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix_b.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix_b.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rowA));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&colA));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&colB));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel =
      std::make_shared<shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::MatrixMultiplicationTaskParallel>(
          taskDataPar);
  ASSERT_EQ(taskParallel->validation(), true);
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    std::vector<int> expected_result;
    shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::matrix_multiply_square_result(global_matrix_a, global_matrix_b,
                                                                                      expected_result, rowA, colB);
    ASSERT_EQ(global_result, expected_result);
  }
}
