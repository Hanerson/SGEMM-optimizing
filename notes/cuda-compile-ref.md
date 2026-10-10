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
│       ├── 1-naive.cuh                # 最基础的 naive kernel
│       └── 2_kernel_global_mem_coalesce.cuh  # 全局内存合并访存优化版
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
| 2 | global memory coalesce (一维 block, 按列连续访存) |

## 编译踩坑记录

### 1. 头文件必须用引号 + 相对路径

`#include <runner.cuh>` ❌ —— 尖括号只搜系统路径，找不到项目内的头文件
`#include "src/runner.cuh"` ✅ —— 引号会优先搜源文件所在目录

如果用 CMake 编译（设了 `-I src`），两种写法都可以；**直接 `nvcc` 必须用引号版本**。

### 2. 多文件编译不能缺 `runner.cu`

```bash
# ❌ 只编译 sgemm.cu 会报链接错误
nvcc sgemm.cu -lcublas -arch=sm_75 -o sgemm

# ✅ 必须同时编译 src/runner.cu
nvcc sgemm.cu src/runner.cu -lcublas -arch=sm_75 -o sgemm
```

`main()` 在 `sgemm.cu`，但 kernel launch 封装和辅助函数（`run_kernel`、`cudaCheck`、`randomize_matrix` 等）在 `src/runner.cu`。

### 3. `throw std::invalid_argument` 需要 `<stdexcept>`

`runner.cu` 里用了 `throw std::invalid_argument(...)`，必须 `#include <stdexcept>`，否则编译报错。

### 4. `cudaCheck` 宏与函数同名

```c
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))
```

宏名和函数名一样，利用 C 预处理器的"禁止自展开"特性工作——展开一次后自动解析为函数调用。功能正常但容易看晕，如果想更清晰可以改名：
```c
#define CUDA_CHECK(err) (cudaCheck(err, __FILE__, __LINE__))
```

### 5. `-arch` 必须匹配实际 GPU

RTX 2080 Ti 是 SM 7.5，用 `-arch=sm_75`。设错架构即使编译通过，运行时也会报错（`no kernel image is available for execution`）。

查 GPU 架构：
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```