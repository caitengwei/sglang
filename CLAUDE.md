# SGLang 项目开发指南

## 项目概述

SGLang 是一个高性能的大语言模型（LLM）和视觉语言模型（VLM）推理服务框架，为全球超过 400,000 个 GPU 提供推理服务。项目核心特点包括：

- **RadixAttention**: 提供 5 倍加速的前缀缓存技术
- **零开销调度器**: CPU 调度器，优化请求处理
- **预填充-解码分离**: 支持 PD 分离架构，提升吞吐量
- **多硬件支持**: NVIDIA、AMD、Intel、Google TPU、Ascend NPU 等
- **生产级部署**: 被 xAI、LinkedIn、Oracle Cloud 等组织使用

## 技术栈

- **编程语言**: Python（主要）、CUDA C++、Rust
- **深度学习框架**: PyTorch 2.9.1
- **推理优化库**: FlashInfer、Triton、Cutlass、DeepGEMM
- **服务框架**: FastAPI、gRPC
- **并行策略**: 张量并行、流水线并行、专家并行

## 项目结构

```
sglang/
├── python/sglang/                    # 主要 Python 代码
│   ├── srt/                          # SGLang Runtime（核心推理引擎）
│   │   ├── entrypoints/              # HTTP/gRPC 服务器入口
│   │   ├── managers/                 # 调度管理器
│   │   ├── models/                   # 模型实现（100+ 模型）
│   │   ├── layers/                   # 自定义层
│   │   │   ├── attention/            # 注意力机制优化
│   │   │   ├── moe/                  # MoE 专家并行
│   │   │   └── quantization/         # 量化（FP8/INT4/AWQ/GPTQ）
│   │   ├── distributed/              # 分布式计算
│   │   ├── disaggregation/           # PD 分离
│   │   └── speculative/              # 推测解码
│   ├── lang/                         # 前端语言接口
│   └── multimodal_gen/               # 多模态生成
├── sgl-kernel/                       # 自定义 CUDA 内核
├── sgl-model-gateway/                # Rust 模型网关
├── benchmark/                        # 性能基准测试
├── test/                             # 测试套件
└── docs/                             # 文档
```

## 开发环境搭建

### 前置要求

- Python 3.9+
- CUDA 12.0+（NVIDIA GPU）或 ROCm（AMD GPU）
- 至少 16GB 显存的 GPU（开发推荐 24GB+）

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/sgl-project/sglang.git
cd sglang

# 2. 创建虚拟环境
conda create -n sglang python=3.10
conda activate sglang

# 3. 安装 SGLang 运行时
cd python
pip install -e .

# 4. 安装 sgl-kernel
cd ../sgl-kernel
pip install -e .

# 5. 安装额外依赖（根据需求）
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/
```

### 验证安装

```bash
# 运行简单测试
python -m sglang.launch_server --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 --port 30000

# 在另一个终端
python -m sglang.test.send_one_request
```

## 常用开发命令

### 运行服务器

```bash
# 基础启动
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat --tp 2

# 带前缀缓存
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat --enable-mixed-chunk

# 启用推测解码
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat --speculative-algo EAGLE

# 多 LoRA 支持
python -m sglang.launch_server --model-path base-model --lora-paths lora1 lora2
```

### 运行测试

```bash
# 运行所有测试
cd python
pytest

# 运行特定测试文件
pytest tests/test_radix_attention.py

# 运行基准测试
cd benchmark
python json_completion_benchmark.py

# 运行 kernel 测试
cd sgl-kernel
cmake -S . -B build
cmake --build build
cd build && ctest
```

### 代码质量检查

```bash
# 格式化代码
cd python
black sglang/
isort sglang/

# 类型检查
mypy sglang/srt/

# Lint
cd sglang
flake8 sglang/
```

### 构建发布包

```bash
# 构建 wheel
cd python
pip install build
python -m build --wheel

# 构建 sgl-kernel
cd sgl-kernel
python setup.py build_ext --inplace
```

## 开发规范和最佳实践

### 代码风格

- 遵循 PEP 8 规范
- 使用 Black 格式化代码（行长度 88 字符）
- 使用 isort 排序导入
- 类型注解：对公共 API 必须使用，私有函数推荐使用

### 性能关键代码

- 性能敏感的代码放在 `sgl-kernel/` 中，使用 CUDA/Triton 实现
- Python 代码使用 `@torch.compile` 装饰器加速热点代码
- 内存分配：使用预分配的张量池避免频繁分配

### 添加新模型

1. 在 `python/sglang/srt/models/` 创建模型文件
2. 继承 `TpEmbedding`, `TpLinear`, `Attention` 等基础层
3. 实现 `forward()` 和 `load_weights()` 方法
4. 在 `test/` 添加模型测试用例
5. 更新 `supported_models.md` 文档

### 测试要求

- 新功能必须包含单元测试
- 性能优化需要基准测试对比
- 分布式功能需测试 TP > 1 的情况
- 使用 `pytest.mark.parametrize` 测试不同配置

## 关键配置文件

### 模型配置

模型量化配置：`python/sglang/srt/layers/quantization/configs/`
- 包含 200+ 模型的量化配置
- 支持 FP8、INT4、AWQ、GPTQ 等多种格式

### Kernel 配置

CUDA Kernel 源码：`sgl-kernel/csrc/`
- FlashAttention 变体
- MoE 相关 kernel
- 量化 kernel

### 部署配置

示例部署配置：`examples/` 目录
- Kubernetes 部署配置
- Docker 配置
- 性能调优配置

## 调试和性能分析

### 环境变量

```bash
# 启用详细日志
export SGLANG_LOG_LEVEL=DEBUG

# 禁用 FlashInfer
export SGLANG_DISABLE_FLASHINFER=1

# 启用手动测试模式
export SGLANG_ENABLE_MANUAL_TEST=1

# 限制 max batch size
export SGLANG_MAX_BATCH_SIZE=16
```

### 性能分析工具

```bash
# 使用 PyTorch Profiler
torchrun --nproc_per_node=2 benchmark/latency_throughput_benchmark.py

# 使用 Nsight Systems（NVIDIA）
nsys profile -o report python -m sglang.launch_server ...

# Kernel 基准测试
cd sgl-kernel/benchmark
python benchmark_attention.py
```

### 常见问题排查

1. **OOM（显存不足）**
   - 减小 `--mem-fraction-static`
   - 降低 `--max-running-requests`
   - 启用 `--enable-p2p-check`

2. **推理速度慢**
   - 检查是否启用 FlashInfer
   - 验证 CUDA kernel 是否编译正确
   - 使用 `--profile` 分析热点

3. **分布式问题**
   - 使用 `NCCL_DEBUG=INFO` 检查通信
   - 验证网络配置（InfiniBand vs Ethernet）
   - 检查 GPU P2P 访问

## 提交 PR 的最佳实践

1. **PR 前检查清单**
   - [ ] 代码通过现有测试
   - [ ] 新功能有对应的测试用例
   - [ ] 性能影响已评估（如有）
   - [ ] 文档已更新
   - [ ] 提交信息清晰描述了更改

2. **性能优化类 PR**
   - 提供基准测试前后对比数据
   - 说明优化原理和适用场景
   - 测试不同模型和 batch size

3. **新模型支持**
   - 提供模型权重加载测试
   - 验证输出与 HuggingFace 一致性
   - 添加到支持的模型列表

## 相关资源

- **文档**: https://docs.sglang.io/
- **Slack 社区**: https://slack.sglang.io/
- **开发会议**: 每周一次（见 Slack 公告）
- **性能基准**: https://github.com/sgl-project/sglang/tree/main/benchmark
- **学习资料**: https://github.com/sgl-project/sgl-learning-materials

## 项目统计

- **Stars**: 15,000+
- **贡献者**: 200+
- **模型支持**: 100+ 模型
- **日处理量**: 数万亿 token
- **采用组织**: xAI、LinkedIn、Oracle Cloud 等

---

*最后更新：2025-12-15*
