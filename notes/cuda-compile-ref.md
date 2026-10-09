# CUDA 编译快速参考

## 最简 CUDA 程序

```cuda
#include <cstdio>

__global__ void hello_kernel() {
    printf("Hello from thread (%d, %d)\n", threadIdx.x, blockIdx.x);
}

int main() {
    hello_kernel<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

## nvcc 编译命令

### 编译单个 `.cu` 文件

```bash
nvcc -o hello hello.cu
```

### 编译多个 `.cu` 文件

```bash
nvcc -o my_prog file1.cu file2.cu
```

### 指定 GPU 架构

RTX 2080 Ti (Compute Capability 7.5):

```bash
nvcc -arch=sm_75 -o hello hello.cu
```

### 链接外部库（如 cuBLAS）

```bash
nvcc -o my_prog my_prog.cu -lcublas
```

### 只编译到 PTX（中间汇编，不生成可执行文件）

```bash
nvcc -arch=sm_75 -ptx my_prog.cu
```

## 本项目

### 手动编译

```bash
cd /home/cuda_learning
nvcc -arch=sm_75 -o my_sgemm sgemm.cu src/runner.cu -lcublas
```

运行 naive kernel（编号 1）：

```bash
./my_sgemm 1
```

### CMake 编译

```bash
cd /home/cuda_learning
cmake -B build -S .
cmake --build build -j$(nproc)
./build/sgemm 1
```

## 常用参数速查

| 参数 | 含义 |
|------|------|
| `-o <file>` | 指定输出文件名 |
| `-arch=sm_XX` | 指定 GPU 架构（如 sm_75） |
| `-ptx` | 只生成 PTX 中间代码 |
| `-G` | 生成设备调试信息（debug 模式，会大幅降低性能） |
| `-l<name>` | 链接库（如 `-lcublas`、`-lcudart`） |
| `-I<path>` | 添加头文件搜索路径 |

## 查 GPU 架构

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

## 项目结构

```
/home/cuda_learning/
├── sgemm.cu              # 主入口
├── CMakeLists.txt         # CMake 构建配置
├── src/
│   ├── kernels.cuh        # kernel 头文件入口
│   ├── runner.cu          # 工具函数 + kernel launch 封装
│   ├── runner.cuh         # 工具函数头文件
│   └── kernels/
│       ├── 1-naive.cuh    # 最基础的 naive kernel
│       └── ...            # 其他优化版本
├── build/                 # 编译输出目录
└── notes/                 # 学习笔记
```

## Naive Kernel 说明

- 位置: `src/kernels/1-naive.cuh`
- 思路: 每个线程计算输出矩阵 C 的一个元素
- Grid: `(ceil(M/32), ceil(N/32))`，Block: `(32, 32)`
- 核心循环: 遍历 K 维度做累加 `tmp += A[x*K+i] * B[i*N+y]`
- 瓶颈: 无共享内存、无分块、无向量化，每个线程独立从全局内存反复读取数据

## Kernel 编号

| 编号 | kernel |
|:----:|:-------|
| 0 | cuBLAS (参考实现) |
| 1 | naive baseline |
| 2+ | 优化版本 |