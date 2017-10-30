# simple_caffe_test

- caffeを使ったアプリケーションのコンパイルを練習する

## Sources
- caffe-1d-regression.cpp
  - main
- net.prototxt
  - definition file of Network
- solver.prototxt
  - definition file of Solver
- ipynb/generate-sample-data.ipynb
  - generate data
- ipynb/visualize_regression_result.ipynb
  - visualization of predicted result

## Build
```
$ g++ -std=c++11 -I${HOME}/caffe/include -L$HOME/caffe/build/lib caffe-sample.cpp -lcaffe -lglog -lboost_system -lhdf5 -lhdf5_hl
```

## Referrences
- http://d.hatena.ne.jp/muupan/20141010/1412895321
- https://github.com/BVLC/caffe

