# DLRM (Naive vs CUDA)

DLRM(Deep Learning Recommendation Model)의
- PyTorch 기반 기본/참고 구현(`dlrm-naive.ipynb`)
- LibTorch + CUDA 최적화 구현 및 벤치마크(`dlrm-cuda.cu`)
을 포함

## 구성
- `dlrm-cuda.cu`: embedding + interaction 커널 fusion, CUDA Graph 등을 이용한 최적화 버전
- `CMakeLists.txt`: LibTorch(CUDA) 기반 빌드 설정
- `dlrm-naive.ipynb`: 동일 구조의 naive/reference 구현(성능 비교용)

## 요구 사항
- CUDA Toolkit
- CUDA enabled LibTorch
  - `CMakeLists.txt`의 `CMAKE_PREFIX_PATH`를 본인 환경의 libtorch 경로로 수정.
- CMake >= 3.18
- C++17 컴파일러

## 빌드
```bash
mkdir -p build
cd build
cmake ..
make -j
```

## 실행
```bash
./build/dlrm-cuda
```
실행 시 코드에 하드코딩된 설정(batch size, feature sizes, MLP 구조 등)으로
warmup 후 CUDA Graph 기반 벤치마크를 수행하고 평균 실행 시간을 출력.

