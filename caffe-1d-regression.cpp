#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <memory>
#include <random>
#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>
#include <glog/logging.h>


void split(const std::string &s, char delim, std::function< void(int, std::string& )> f) {
  std::string token;
  std::istringstream stream(s);
  int i=0;
  while(getline(stream, token, ',')){
    f(i, token);
    //std::cout << i << ", " << token << std::endl;
    i++;
  }
}


int main(int argc, char** argv) {
  // glogの初期化
  google::InitGoogleLogging(argv[0]);

  // データをロード
  //std::ifstream ifs("data/sampledata_1d_linear_short.csv");
  std::ifstream ifs("data/sampledata_discrete_target.csv");
  if (ifs.fail())
    {
      std::cerr << "failed : file open" << std::endl;
      return -1;
    }
  std::string line;
  // ヘッダ部分のロード
  std::vector<std::string> headerList;
  getline(ifs, line);
  split(line, ',', [&](int i, std::string & header) {
      headerList.push_back(header);
    });
  // for (auto h: headerList)
  //   std::cout << h << std::endl;

  // データ部分のロード
  std::map<int, std::vector<float>> dataMap;
  while (getline(ifs, line))
    {
      split(line, ',', [&](int i, std::string & token) {
	  dataMap[i].push_back(std::stof(token));
	});
    }
  // for(auto itr=dataMap.begin(); itr != dataMap.end(); ++itr) {
  //   std::cout << "key=" << itr->first << std::endl;
  //   for(auto d: dataMap[itr->first])
  //     std::cout << d << std::endl;
  // }
  std::cout << "data_size : " << dataMap[0].size() << std::endl;


  // パラメータ指定
  const int batch_size = 10;


  // 読み込んだデータをcaffeのMemoryDataLayerに入力したい
  // この問題は回帰を解きたいので、ラベルデータが存在しない
  // だから、ダミーの値を用意しておく
  std::vector<float> dummy_data( dataMap[0].size(), 0.0 );

  // Solverの設定をテキストファイルから読み込む
  caffe::SolverParameter solver_param;
  caffe::ReadSolverParamsFromTextFileOrDie("solver.prototxt", &solver_param);
  std::shared_ptr<caffe::Solver<float>> solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
  const auto net = solver->net();

  // 入力データをMemoryDataLayer"input"にセットする
  const auto input_layer =
    boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(net->layer_by_name("input"));
  assert(input_layer);
  input_layer->Reset(dataMap[0].data(), dummy_data.data(), dataMap[0].size());
  
  // 目標データをMemoryDataLayer"target"にセットする
  const auto target_layer =
    boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(net->layer_by_name("target"));
  assert(target_layer);
  target_layer->Reset(dataMap[1].data(), dummy_data.data(), dataMap[1].size());
  
  // Solverの設定通りに学習を行う
  solver->Solve();
  

  // 学習されたモデルを使って予測
  // 結果をファイルに出力
  std::ofstream ofs("data/predicted_result.csv");
  // x ~ [0, 1.0]
  std::random_device rand_dev; // 乱数のシードを決めるために
  std::mt19937 mt(rand_dev());
  std::uniform_real_distribution<> dist(0.0, 9.0); // 一様分布のクラス

  int n_test = 100; // テストの回数
  for(int n=0 ; n<(int)(n_test/batch_size) ; n++){
    std::vector<float> sample_input;
    for(int i=0 ; i<batch_size ; i++) { // バッチ分のデータを入力
      sample_input.push_back(dist(mt));
    }
    input_layer->Reset(sample_input.data(), dummy_data.data(), sample_input.size());
    const auto result = net->Forward();
    for(int i=0 ; i<sample_input.size() ; i++){
      //std::cout << "x=" << sample_input[i] << ", y=" << net->blob_by_name("ip")->cpu_data()[i] << std::endl;
      ofs << sample_input[i] << "," << net->blob_by_name("ip")->cpu_data()[i] << std::endl;
    }
  }
  
}
