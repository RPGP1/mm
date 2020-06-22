行列積 with CUDA
===

CUDAプログラミングモデルを用いて行列積を高速化した実装例。システム要件は以下の通りである。

|項目|設定した必要条件|動作確認した環境|
|:-|:-|:-|
|GPU|CUDA Compute Capability 7.x のGPUが2つ|NVIDIA Quadro GV100 ([データシートはこちら](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/productspage/quadro/quadro-desktop/quadro-volta-gv100-data-sheet-us-nvidia-704619-r3-web.pdf))が2つ|
|ビルドシステム|cmake (>=3.12) + make|cmake (=3.17) + make (=4.1)|
|C++コンパイラ|C++17への対応|g++-8 (Ubuntu 8.3.0-6ubuntu1~18.04.1) 8.3.0|
|NVCC|10.0以上|10.0.130|

## ビルド

[トップにあるCMakeLists.txt](./CMakeLists.txt)を参照してビルドシステムを起動すればよい。例えば次のようになる。

```sh
git clone https://github.com/RPGP1/mm.git --recursive
mkdir mm/build && cd mm/build
cmake .. && make
```

## 実行

実行バイナリは、CMakeのバイナリディレクトリから見て`./cuda/mm_cuda`に生成される。実行にあたってはオプション`-p`に計算問題を記録したファイルを指定する。例えば[ビルド](#ビルド)でのコマンドに続けて次を入力する。

```sh
./cuda/mm_cuda -p ../data/large.dat
```

## ファイル構成

ヘッダファイル(`*.hpp`, `*.ipp`)は[includeディレクトリ](./include)、ソースファイル(`*.cpp`, `*.cu`)は[srcディレクトリ](./src)に入っている。

それぞれのファイルの役割は次のようになっている。

*   `gemm.[hic]pp, gemm_fwd.hpp`: 計算する行列を分割してGPUそれぞれに割り振ったり、そのためのメモリをGPU側に確保したりといったCPU側の前準備をする。
*   `kernel.[hi]pp, kernel.cu`: 行列を転送し、計算し、結果を返送する。時間計測のほとんどである。`kernel.ipp`にはGPUカーネルの挙動を調整するパラメータも並んでいる。
*   `main.cpp`: 計算問題ファイルを扱う[`mm-tester`](https://github.com/RPGP1/mm-tester.git)を活用して、データの読み込みや時間計測、結果の表示をする。
*   パラメータ設定
    *   `definition.hpp`: 扱う浮動小数点数の型など
    *   `size.hpp`: 行列の大きさ
*   細々としたヘルパー
    *   `check_device_props.[hc]pp`: GPUに対する要件の一部を確認する。
    *   `error.[hc]pp`: CUDA APIの返り値をハンドリングする。
    *   `host_allocation.[hi]pp`: 行列を格納するCPU側のメモリを確保する。

## 実行結果の例

次のような出力を得た。22.5 TFLOPSである。

```
# Problem: "../data/large.dat"
# Size: lhs[ 105248 ][ 49152 ] * rhs[ 49152 ][ 52736 ]
# Elapsed [ms]: 24245.542196999998851
# FLOPS [Gflops]: 22504.031475279778533
# Standard (Strict): 96
# Standard (Loose): 768
# Difference count (Strict): 0
# Difference count (Loose): 0
# Max Wrongness: 63
# @ (0, 0)
# max_abs_answer = 0
```
