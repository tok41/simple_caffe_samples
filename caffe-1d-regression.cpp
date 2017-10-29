#include <iostream>
#include <memory>
#include <random>
#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>
#include <glog/logging.h>

int main(int argc, char** argv) {
  // glogの初期化
  google::InitGoogleLogging(argv[0]);

  // パラメータ指定
  const int batch_size = 10;
  const int data_size = 1000;

  

  constexpr auto kMinibatchSize = 32;
  constexpr auto kDataSize = kMinibatchSize * 10;
  
  // 教師データとして用いる入力データと目標データをfloat配列として準備する．
  // 入力データ：2次元
  // 目標データ：1次元
  std::array<float, kDataSize * 2> input_data;
  std::array<float, kDataSize> target_data;
  std::mt19937 random_engine; // MT法に夜乱数生成器
  std::uniform_real_distribution<> dist(0.0, 1.0); // 一様分布のクラス

  // 3*x_1 - 2*x_2 + 4 = target に従ってデータを生成する．
  for (auto i = 0; i < kDataSize; ++i) {
    const auto x_1 = dist(random_engine); //一様分布に従った乱数を生成
    const auto x_2 = dist(random_engine);
    const auto target = 3 * x_1 - 2 * x_2 + 4;
    input_data[i * 2] = x_1;
    input_data[i * 2 + 1] = x_2;
    target_data[i] = target;
  }

  // データをファイルに出力
  std::ofstream ofs("sample_data.csv");
  for (auto i = 0; i < kDataSize; ++i) {
    ofs<<input_data[i * 2]<<","<<input_data[i*2+1]<<","<<target_data[i]<<std::endl;
  }

  // MemoryDataLayerはメモリ上の値を出力できるDataLayer．
  // 各MemoryDataLayerには入力データとラベルデータ（1次元の整数）の2つを与える必要があるが，
  // ここでは回帰を行いたいので，入力データと目標データそれぞれを別のMemoryDataLayerで出力し，
  // ラベルデータの代わりに使用されないダミーの値を与えておく．
  std::array<float, kDataSize> dummy_data; // ダミーのラベル
  std::fill(dummy_data.begin(), dummy_data.end(), 0.0);

  // Solverの設定をテキストファイルから読み込む
  caffe::SolverParameter solver_param;
  caffe::ReadSolverParamsFromTextFileOrDie("solver.prototxt", &solver_param);
  std::shared_ptr<caffe::Solver<float>> solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
  const auto net = solver->net();

  // 入力データをMemoryDataLayer"input"にセットする
  const auto input_layer =
    boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(net->layer_by_name("input"));
  assert(input_layer);
  input_layer->Reset(input_data.data(), dummy_data.data(), kDataSize);

  // 目標データをMemoryDataLayer"target"にセットする
  const auto target_layer =
    boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(net->layer_by_name("target"));
  assert(target_layer);
  target_layer->Reset(target_data.data(), dummy_data.data(), kDataSize);

  // Solverの設定通りに学習を行う
  solver->Solve();

  // 学習されたパラメータを出力してみる
  // ax + by + c = target
  const auto ip_blobs = net->layer_by_name("ip")->blobs();
  const auto learned_a = ip_blobs[0]->cpu_data()[0];
  const auto learned_b = ip_blobs[0]->cpu_data()[1];
  const auto learned_c = ip_blobs[1]->cpu_data()[0];
  std::cout << learned_a << "x_1 + " << learned_b << "x_2 + " << learned_c
	    << " = target" << std::endl;

  // 学習されたモデルを使って予測してみる
  // x = 10, y = 20
  std::array<float, kDataSize * 2> sample_input;
  sample_input[0] = 10;
  sample_input[1] = 20;
  input_layer->Reset(sample_input.data(), dummy_data.data(), kDataSize);
  net->ForwardPrefilled(nullptr);
  std::cout << "10a + 20b + c = " << net->blob_by_name("ip")->cpu_data()[0] << std::endl;
}
