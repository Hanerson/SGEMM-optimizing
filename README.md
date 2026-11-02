# SGEMM 优化实践：从 naive 到 warptiling

一个 FP32 SGEMM（单精度矩阵乘）的渐进优化项目。七个 CUDA kernel 从最朴素的实现出发，
一步步走到接近 cuBLAS 的水平，每一步只引入一个核心思想：合并访存、共享内存分块、
寄存器分块、向量化、warp 分块。

所有性能数据都是在 **NVIDIA GeForce RTX 4090** 上实测得到的，不是估算值，也不是从别处
抄来的参考数字。

## 测试环境

| 项目 | 配置 |
| --- | --- |
| GPU | **NVIDIA GeForce RTX 4090**（AD102，`sm_89`，128 SM） |
| 显存 | 24564 MiB（24 GB GDDR6X） |
| 驱动 / CUDA | 570.124.06 / 12.8（`nvcc` V12.8.61） |
| 主机 | Intel Xeon Platinum 8358P @ 2.60 GHz，128 线程，20 GB 内存 |
| 系统 | Ubuntu 22.04 容器（kernel 5.4.0-100-generic） |
| 编译 | `cmake 3.31.4` + `gcc 13.3.0`，`-DCUDA_COMPUTE_CAPABILITY=89` |

> 镜像里 `nvcc` 不在 `PATH` 上，`make` 会自动回退到 `/usr/local/cuda/bin/nvcc`，
> 也可以用 `make build NVCC=/path/to/nvcc` 指定。

## 性能结果

测试规模 M = N = K，FP32，`alpha = 0.5`、`beta = 3.0`，每个尺寸 50 次 `cudaEvent` 计时
取平均。单位 GFLOPS。

表中每个格子是 **三轮完整扫描的中位数**。同一格的多轮极差大多在 1% 以内，
少数格子更高（k10 在 4096 维为 4.4%），单轮采样不足以代表真实水平。

| kernel | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| k0 cuBLAS | 4.6 | 196.0 | 21174.8 | 42661.9 | 54333.9 | 53535.1 |
| k1 naive | 73.5 | 301.4 | 613.1 | 626.1 | 668.0 | 668.7 |
| k2 coalesce | 500.6 | 2243.8 | 4893.0 | 5068.9 | 5189.6 | 5230.5 |
| k3 shared mem | 604.6 | 2773.0 | 6122.5 | 6480.4 | 6556.1 | 6461.4 |
| k4 1D blocktiling | 379.2 | 1710.2 | 7298.5 | 18663.5 | 20074.6 | 18958.3 |
| k5 2D blocktiling | 176.9 | 847.6 | 3783.1 | 15530.7 | 34621.0 | 35940.0 |
| k6 vectorize | 226.3 | 1017.0 | 4299.1 | 17688.6 | 43898.8 | 42325.1 |
| k10 warptiling | 226.2 | 1085.7 | 4834.8 | 20429.4 | 48113.6 | **48247.3** |

### 4096³ 汇总

| kernel | GFLOPS | 占 cuBLAS | 占标称峰值 | 耗时 | 相对上一级 |
| --- | ---: | ---: | ---: | ---: | ---: |
| k0 cuBLAS | 53535.1 | 100.0% | 64.8% | 2.57 ms | — |
| k1 naive | 668.7 | 1.2% | 0.8% | 205.5 ms | — |
| k2 coalesce | 5230.5 | 9.8% | 6.3% | 26.3 ms | 7.82× |
| k3 shared mem | 6461.4 | 12.1% | 7.8% | 21.3 ms | 1.24× |
| k4 1D blocktiling | 18958.3 | 35.4% | 23.0% | 7.25 ms | 2.93× |
| k5 2D blocktiling | 35940.0 | 67.1% | 43.5% | 3.82 ms | 1.90× |
| k6 vectorize | 42325.1 | 79.1% | 51.2% | 3.25 ms | 1.18× |
| k10 warptiling | 48247.3 | 90.1% | 58.4% | 2.85 ms | 1.14× |

- **k10 达到 cuBLAS 的 90.1%**，相对 naive 提速 **72.2×**
- cuBLAS 跑到标称峰值的 64.8%，k10 到 58.4%
- 「占标称峰值」按 RTX 4090 FP32 标称 82.6 TFLOPS 计算
  （128 SM × 128 lane × 2 FLOP × 2.52 GHz 参考加速频率）

### 两条曲线

横向看（同一 kernel 随规模变化）：k1~k3 在 1024 维之后基本走平，说明它们已经撞到各自的
结构性上限；k4~k10 则一直涨到 2048 维才饱和。

纵向看（同一规模下各 kernel 的排名）：128 维时最优的是 k3（604.6），4096 维时最优的是
k10（48247.3）。**小尺寸和大尺寸的排名不同**，所以「哪个 kernel 最快」必须带上规模才有
意义。

## 快速开始

```bash
# 编译（默认按 sm_89 构建，可用 CUDA_ARCH 覆盖）
make build

# 跑单个 kernel 的完整尺寸扫描（0 是 cuBLAS，1/2/3/4/5/6/10 是自研 kernel）
make run KERNEL=10

# 跑全部 kernel，输出上表用到的数据
make bench

# 清理
make clean
```

不用 `make` 也可以直接走 CMake：

```bash
export CUDACXX=/usr/local/cuda/bin/nvcc
cmake -S . -B build -DCMAKE_CUDA_COMPILER=$CUDACXX \
      -DCUDA_COMPUTE_CAPABILITY=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/sgemm 10
```

程序参数是 kernel 编号：`0` 走 cuBLAS，`1 2 3 4 5 6 10` 走自研实现。
每次运行依次测试 128、256、512、1024、2048、4096 六个尺寸，每个尺寸先与 cuBLAS
校验正确性，再做 50 次计时。

## 优化路线

每一站只引入一个新思想，下面的倍数来自上表 4096³ 的实测。

### k1 → k2：合并访存（7.82×）

k1 的线程映射是 `x = threadIdx.x`（行）、`y = threadIdx.y`（列），一个 warp 的 32 个 lane
落在 A 的 32 个不同行上，地址相差 `K×4B`，一次 load 被拆成 32 次 32B 事务。

k2 把映射换成一维、lane 沿列连续：读 A 变成同一地址（广播），读 B 变成连续 128B
（一次事务）。计算量完全没变，只是地址变了，就快了 7.82 倍——不是算得更快，
是同样的数据少搬了约 8 倍。

### k2 → k3：共享内存分块（1.24×）

每个 block 先把 A、B 的 32×32 切片搬进 SMEM，再在 SMEM 上做点积，把重复的 GMEM 读取
变成一次搬运。

收益只有 1.24 倍，因为 k2 的重复读很多已经在 L1/L2 命中，SMEM 只是把隐式缓存换成显式
缓存；而且 32×32 的 tile 算术强度只有 8 FLOP/B，远低于这张卡的平衡点。

### k3 → k4：1D 寄存器分块（2.93×）

每个线程从「算 1 个 C」改成「算同一列的 8 个 C」。B 的一个值读一次、复用 8 次，
SMEM 读指令与 FMA 的比例从 2:1 降到 1.125:1。

这是第一个真正见效的寄存器级优化：把数据从 SMEM 再往寄存器搬一层。

### k4 → k5：2D 寄存器分块（1.90×）

把 8×1 的「柱子」换成 8×8 的方阵。每个 `dotIdx` 读 16 个操作数、做 64 次 FMA，
LDS/FMA 从 1.125 降到 0.25，A 和 B 同时被复用。

代价是寄存器压力：64 个累加器 + 16 个操作数，约 90 个寄存器/线程，occupancy 明显下降。
收益没有 LDS 降幅那么大，是典型的 Amdahl 现象——只优化了「SMEM 读」这一段。

### k5 → k6：向量化（1.18×）

用 `float4` 把 4 次标量加载合成一条 128-bit 指令，并把 As 在 SMEM 中转置成 `[BK][BM]`，
让 A 的 SMEM 读也能用 `LDS.128`。

收益有限：FMA 数量没变，SMEM 带宽没变，省下的只是指令条数。**向量化是锦上添花，
应该在「已经算得够多」之后再做。**

### k6 → k10：warp 分块（1.14×）

前六个 kernel 里 warp 是硬件隐式划分的；k10 把 warp 这一层显式写进代码：128 个线程 =
4 个 warp，每个 warp 负责 64×64 的 C 子块，每个线程管 128 个结果。

这时 occupancy 只有 17%~25%，但性能反而最高——因为每个线程有 128 个互不依赖的 FMA，
足够把流水线填满（ILP）。**occupancy 是手段，不是目标。**

## 这张卡上的几个观察

**1. 机器平衡点比上一代高得多。**
RTX 4090 的 FP32 标称峰值 82.6 TFLOPS、显存带宽 1008 GB/s，平衡点约 **82 FLOP/B**。
而 128×128 的 block tile 只能提供约 32 FLOP/B，理论上仍在内存受限区间。
k5/k6/k10 能跑到 cuBLAS 的 67%~90%，很大程度靠 **72 MB 的 L2** 兜住了跨 block 复用。

**2. 小尺寸严重失真。**
128 维时 cuBLAS 只有 4.6 GFLOPS，比 512 维低三个数量级——一次 128³ 的 GEMM 只有
4.2 MFLOP，50 次重复加起来还在毫秒级，冷启动、显存频率爬升、launch 开销全部被算进去。
表中 128/256 两列只适合看趋势，不要拿来下结论。

**3. 中间档存在非单调。**
k4 在 2048 维是 20074.6，到 4096 维反而降到 18958.3；k6 同样从 43898.8 降到 42325.1。
k5 和 k10 则继续上涨。这类回落通常来自 wave 量化（4096 维下每个 SM 分到的 block 数不整齐）
与寄存器/occupancy 的相互作用，三轮扫描可复现同一趋势，不是噪声。

## 测量口径与已知问题

计时逻辑在 `sgemm.cu`：

| 项目 | 设置 |
| --- | --- |
| 计时方式 | `cudaEvent`，包住 50 次 kernel launch |
| 重复次数 | 50 |
| 预热 | 正式计时前先跑一次正确性校验 |
| 尺寸 | 128 / 256 / 512 / 1024 / 2048 / 4096 |
| `alpha` / `beta` | 0.5 / 3.0 |
| 参考实现 | cuBLAS FP32 |
| FLOPs | `2·m·n·k` |
| 正确性判据 | 绝对误差 < 0.01 |

三点需要明确：

1. **`beta = 3.0` 且计时循环内不重置 C。** 每跑一次 `C = αAB + 3C`，50 次后放大
   `3^50 ≈ 7.2×10^23`。指令条数不变，计时仍然有效，但这不是好习惯，严谨做法是每次
   重置 C 或令 `beta = 0`。
2. **正确性用绝对误差 0.01，不随 K 缩放。** 4096 长度的点积用固定阈值，换一组数据
   分布可能误判。更稳妥的是相对误差，或用 double 参考。
3. **k5/k6/k10 没有边界检查**，要求 M、N、K 都是 tile 尺寸的整数倍。本项目只测 2 的幂，
   不受影响；作为通用实现它是不完整的。

表中数据由三轮完整扫描取中位数得到。测量期间 GPU 利用率 0%、核心频率 210 MHz（idle）、
功耗 14~17 W，没有其他计算任务竞争；显存中有约 20.7 GB 被已退出的进程占用，
不影响计算与带宽。

## 硬件基线

| 项目 | RTX 4090（AD102，`sm_89`） |
| --- | ---: |
| SM 数量 | 128 |
| FP32 CUDA Core / SM | 128 |
| FP32 标称峰值 | ≈ 82.6 TFLOPS |
| 显存 | 24 GB GDDR6X |
| 显存带宽 | 1008 GB/s |
| L2 Cache | 72 MB |
| 共享内存 / SM | 100 KB |
| 寄存器 / SM | 65536 × 4B = 256 KB |
| 机器平衡点 | 82.6 TFLOPS ÷ 1008 GB/s ≈ **82 FLOP/B** |

`nvidia-smi` 报告该卡最大 SM 时钟为 3105 MHz；若按此频率计算理论峰值会更高。
本表统一采用 2.52 GHz 参考加速频率，与公开规格一致，便于横向比较。

## 目录结构

```
cuda-info/
├── sgemm.cu                     # 主程序：解析 kernel 编号、分配矩阵、校验、计时
├── Makefile                     # build / run / bench / clean / submit
├── CMakeLists.txt               # 架构可用 -DCUDA_COMPUTE_CAPABILITY 覆盖
├── src/
│   ├── runner.cu                # kernel launch 封装、工具函数、run_kernel 总调度
│   ├── runner.cuh
│   ├── kernels.cuh              # 汇总所有 kernel 头文件
│   └── kernels/
│       ├── 1_naive.cuh
│       ├── 2_kernel_global_mem_coalesce.cuh
│       ├── 3_kernel_shared_mem_blocking.cuh
│       ├── 4_1D_blockTiling.cuh
│       ├── 5_2D_blockTiling.cuh
│       ├── 6_kernel_vectorize.cuh
│       └── 10_kernel_warptiling.cuh
└── notes/
    └── cuda-compile-ref.md      # CUDA 编译命令速查
```

kernel 编号从 6 跳到 10，沿用 Simon Boehm 原文的顺序：k7/k8 是消 bank conflict 的尝试
（消掉了但整体更慢），k9 是自动调参，k10 是 warptiling。本项目保留 k10 的编号，方便与
原文和笔记对照。

## 参考

1. Simon Boehm, [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
2. Simon Boehm, [SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA)
3. NVIDIA, [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
4. NVIDIA, [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
5. wangzyon, [NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE) —— 计时框架的来源
6. 完整推导与逐站讲解见配套笔记：[SGEMM 优化，从 naive 到 warptiling](https://blog.hanerson.com/2026/10/20/SGEMM-%E4%BC%98%E5%8C%96%E9%95%BF%E5%BE%81%E8%B7%AF/)
