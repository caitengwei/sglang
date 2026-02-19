# SGLang Q1 Roadmap 详细分析 (Issue #12780)

> **注意**：Issue #12780 链接指向上游 sgl-project/sglang 仓库，可能需要验证可访问性：https://github.com/sgl-project/sglang/issues/12780

## 概述

**Issue**: #12780 - Development Roadmap (2026 Q1) *(链接待核实)*
**作者**: hnyls2002 (Liangsheng Yin)
**发布时间**: 2025-11-06
**最后更新**: 2026-01-19

**核心目标**: 特性完整性与生产级可靠性、易用性、新硬件优化、RL 框架集成、多模态支持

---

## 一、Base Engine Features (基础引擎特性)

### 1.1 Overlap Scheduler 默认开启 (Speculative Decoding)
- **PoC**: @hnyls2002
- **Slack**: #spec-decoding
- **Issue**: #11762

| 维度 | 分析 |
|------|------|
| **作用** | 在推测解码场景下默认启用重叠调度器，重叠计算与通信，减少调度开销 |
| **技术难点** | 1. 需要精确控制内存使用，避免 OOM<br>2. 与不同注意力后端的兼容性<br>3. 动态 batch size 下的稳定性 |
| **实现难度** | ⭐⭐⭐ (中等) |
| **价值评估** | ⭐⭐⭐⭐⭐ (极高) - 直接影响吞吐量和延迟，是生产环境关键优化 |
| **相关代码** | `python/sglang/srt/managers/scheduler.py` |
| **预估工作量** | 2-3 周 |

### 1.2 Prefill CUDA Graph 默认开启
- **PoC**: @Oasis-Git @ispobock @BBuf
- **Slack**: #piecewise-cuda-graph
- **Issue**: #11490

| 维度 | 分析 |
|------|------|
| **作用** | 对 prefill 阶段启用 CUDA Graph，减少 CPU 开销，降低 latency |
| **技术难点** | 1. 动态 sequence length 的处理<br>2. 与 CUDA Graph 的内存限制协调<br>3. 多模态输入的兼容性 |
| **实现难度** | ⭐⭐⭐⭐ (较高) |
| **价值评估** | ⭐⭐⭐⭐ (高) - 显著降低 TTFT (Time To First Token) |
| **相关代码** | `python/sglang/srt/layers/cuda_graph.py` |
| **预估工作量** | 3-4 周 |

### 1.3 通用内存池与前缀缓存 (Hybrid Models)
- **PoC**: @cctry @xiezhq-hermann
- **Slack**: #prefix-cache, #kv-cache-store
- **Issue**: #12587

| 维度 | 分析 |
|------|------|
| **作用** | 为多模态/混合模型提供统一的内存池和前缀缓存机制 |
| **技术难点** | 1. 不同模态的 KV cache 格式统一<br>2. 跨模态的 prefix matching 策略<br>3. 内存碎片化管理 |
| **实现难度** | ⭐⭐⭐⭐⭐ (很高) |
| **价值评估** | ⭐⭐⭐⭐⭐ (极高) - 多模态场景的核心基础设施 |
| **依赖项** | KV Cache System & Memory Pool (#12826) |
| **预估工作量** | 6-8 周 |

### 1.4 Mixed Chunked Prefill 重构
- **PoC**: @hzh0425 @yizhang2077
- **Issue**: #13626

| 维度 | 分析 |
|------|------|
| **作用** | 重构混合分块 prefill，支持更灵活的 chunk 策略 |
| **技术难点** | 1. 保持与现有调度逻辑的兼容<br>2. 优化 chunk 大小动态调整 |
| **实现难度** | ⭐⭐⭐ (中等) |
| **价值评估** | ⭐⭐⭐⭐ (高) - 提升长文本处理能力 |
| **预估工作量** | 2-3 周 |

### 1.5 Torch Compile Stack
- **PoC**: (寻找负责人)
- **Slack**: #torch-compile
- **相关**: PR #10987, Issue #10118

| 维度 | 分析 |
|------|------|
| **作用** | 全面支持 PyTorch 2.x compile 特性，加速模型执行 |
| **技术难点** | 1. 与 CUDA Graph 的兼容性<br>2. 动态形状的编译处理<br>3. 调试困难 |
| **实现难度** | ⭐⭐⭐⭐⭐ (很高) |
| **价值评估** | ⭐⭐⭐⭐ (高) - 长期性能提升潜力大 |
| **风险评估** | 高 - 可能影响稳定性，需要大量测试 |
| **预估工作量** | 8-10 周 |

### 1.6 SRT Core/Plugin 重构
| 维度 | 分析 |
|------|------|
| **作用** | 使核心可复用，便于外部定制化开发和维护 out-of-tree 代码 |
| **技术难点** | 1. 设计稳定的 API 边界<br>2. 向后兼容性保证<br>3. 插件生命周期管理 |
| **实现难度** | ⭐⭐⭐⭐⭐ (很高) |
| **价值评估** | ⭐⭐⭐⭐⭐ (极高) - 生态扩展的关键 |
| **预估工作量** | 10-12 周 |

### 1.7 DP Attention 与 Attention Backend 重构
| 维度 | 分析 |
|------|------|
| **作用** | 使注意力后端无状态化，统一 DP attention 同步点 |
| **技术难点** | 1. 状态管理与传递<br>2. 跨设备同步优化<br>3. 多种后端统一抽象 |
| **实现难度** | ⭐⭐⭐⭐⭐ (很高) |
| **价值评估** | ⭐⭐⭐⭐⭐ (极高) - 分布式推理的基础 |
| **预估工作量** | 8-10 周 |

---

## 二、Parallelism (并行策略)

### 2.1 Pipeline Parallelism 重构
- **PoC**: @ShangmingCai
- **Slack**: #pipeline-parallel
- **Issue**: #11857

| 维度 | 分析 |
|------|------|
| **作用** | 支持长上下文 prefill 和高吞吐 decoding 的流水线并行 |
| **技术难点** | 1. bubble 问题优化<br>2. 与 PD 分离的协调<br>3. 负载均衡策略 |
| **实现难度** | ⭐⭐⭐⭐⭐ (很高) |
| **价值评估** | ⭐⭐⭐⭐⭐ (极高) - 大规模部署的核心能力 |
| **预估工作量** | 6-8 周 |

### 2.2 Expert Parallelism 重构
- **PoC**: @ch-wan
- **Slack**: #expert-parallel
- **相关**: #8715, PRs #10423, #11837

| 维度 | 分析 |
|------|------|
| **作用** | 优化 MoE 模型的专家并行实现 |
| **技术难点** | 1. 专家路由效率<br>2. 负载不均衡处理<br>3. 与 TP/PP 的组合 |
| **实现难度** | ⭐⭐⭐⭐ (较高) |
| **价值评估** | ⭐⭐⭐⭐⭐ (极高) - DeepSeek 等 MoE 模型必需 |
| **预估工作量** | 4-6 周 |

### 2.3 Context Parallelism 候选
- **相关**: #16632, PR #12820, PR #14982, PR #14194

| 维度 | 分析 |
|------|------|
| **作用** | 长上下文处理的上下文并行 |
| **技术难点** | 1. Ring Attention 实现<br>2. 通信开销优化 |
| **实现难度** | ⭐⭐⭐⭐ (较高) |
| **价值评估** | ⭐⭐⭐⭐ (高) - 支持超长上下文 |
| **预估工作量** | 4-6 周 |

### 2.4 并行策略兼容性目标
| 目标 | 难度 | 价值 |
|------|------|------|
| All Parallelisms + Speculative Decoding | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| All Parallelisms + PD Disaggregation | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Multiple LB strategies for DP attention | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### 2.5 GB200/GB300 NVL72 优化
- **PoC**: @Fridge003 @fzyzcjy
- **Slack**: #deepseek-large-scale-serving

| 维度 | 分析 |
|------|------|
| **作用** | 针对新一代 NVIDIA 硬件的深度优化 |
| **技术难点** | 1. NVLink72 拓扑利用<br>2. FP4/FP8 精度调优<br>3. 新特性适配 |
| **实现难度** | ⭐⭐⭐⭐⭐ (很高) |
| **价值评估** | ⭐⭐⭐⭐⭐ (极高) - 硬件性能最大化 |
| **硬件依赖** | 需要 GB200/GB300 实机测试 |
| **预估工作量** | 8-12 周 |

---

## 三、Server Reliability (服务可靠性)

### 3.1 Illegal Memory Access 修复
- **Issue**: #11968

| 维度 | 分析 |
|------|------|
| **作用** | 修复非法内存访问导致的崩溃 |
| **技术难点** | 1. 难以复现的 race condition<br>2. 多线程内存管理 |
| **实现难度** | ⭐⭐⭐⭐ (较高) |
| **价值评估** | ⭐⭐⭐⭐⭐ (极高) - 生产稳定性关键 |
| **预估工作量** | 持续进行 |

### 3.2 Runtime Memory/Paging Checker
| 维度 | 分析 |
|------|------|
| **作用** | 运行时内存和分页检查 |
| **技术难点** | 1. 低开销实现<br>2. 准确的错误定位 |
| **实现难度** | ⭐⭐⭐ (中等) |
| **价值评估** | ⭐⭐⭐⭐ (高) - 提升调试效率 |
| **预估工作量** | 2-3 周 |

### 3.3 Grammar Crash Fault Tolerance
| 维度 | 分析 |
|------|------|
| **作用** | 结构化输出 grammar 崩溃容错 |
| **实现难度** | ⭐⭐⭐ (中等) |
| **价值评估** | ⭐⭐⭐⭐ (高) |
| **预估工作量** | 1-2 周 |

### 3.4 Server Crash Fault Tolerance
| 维度 | 分析 |
|------|------|
| **作用** | 服务端整体崩溃容错 |
| **实现难度** | ⭐⭐⭐⭐ (较高) |
| **价值评估** | ⭐⭐⭐⭐⭐ (极高) - 高可用性保障 |
| **预估工作量** | 4-6 周 |

---

## 四、Kernel (内核优化)

### 4.1 JIT Kernels
- **Roadmap**: #17035
- **PoC**: @DarkSharpness

| 维度 | 分析 |
|------|------|
| **作用** | 即时编译内核，针对具体 workload 优化 |
| **技术难点** | 1. 编译开销控制<br>2. 缓存策略<br>3. 与现有 kernel 系统整合 |
| **实现难度** | ⭐⭐⭐⭐⭐ (很高) |
| **价值评估** | ⭐⭐⭐⭐ (高) - 灵活性与性能并重 |
| **预估工作量** | 6-8 周 |

### 4.2 Integrate FlashInfer Kernels
- **Issue**: #17130
- **Slack**: #flashinfer-kernels

| 维度 | 分析 |
|------|------|
| **作用** | 集成 FlashInfer 的高性能注意力 kernel |
| **技术难点** | 1. API 适配<br>2. 版本同步<br>3. 多后端支持 |
| **实现难度** | ⭐⭐⭐ (中等) |
| **价值评估** | ⭐⭐⭐⭐⭐ (极高) - 直接性能提升 |
| **预估工作量** | 3-4 周 |

### 4.3 Tune FP8 GEMM in Cutlass
- **Slack**: #kernel-dev

| 维度 | 分析 |
|------|------|
| **作用** | 调优 FP8 GEMM 性能 |
| **技术难点** | 1. Cutlass 模板复杂<br>2. 精度-性能 tradeoff<br>3. 不同 GPU 架构适配 |
| **实现难度** | ⭐⭐⭐⭐ (较高) |
| **价值评估** | ⭐⭐⭐⭐ (高) - FP8 推理关键 |
| **预估工作量** | 4-6 周 |

### 4.4 Communication Kernel 工作
| 特性 | 难度 | 价值 |
|------|------|------|
| NCCL Symmetric Memory (PR #8238, #12572) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Overlap TP Communication with Compute (PR #9058) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Integrate A2A Kernels (e.g., pplx) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### 4.5 Automated Nightly Fusion Detection
- **Workflow**: actions/runs/19004823026
- **Slack**: #ci-cd-build-release

| 维度 | 分析 |
|------|------|
| **作用** | 自动化检测 kernel fusion 机会 |
| **实现难度** | ⭐⭐⭐⭐ (较高) |
| **价值评估** | ⭐⭐⭐ (中) - 长期优化工具 |
| **预估工作量** | 3-4 周 |

---

## 五、Speculative Decoding (推测解码)

### 5.1 通用推测算法抽象
| 维度 | 分析 |
|------|------|
| **作用** | 支持多种推测算法的统一抽象层 |
| **技术难点** | 1. 不同算法的统一接口<br>2. 性能开销最小化 |
| **实现难度** | ⭐⭐⭐⭐ (较高) |
| **价值评估** | ⭐⭐⭐⭐⭐ (极高) - 算法创新的基础 |
| **预估工作量** | 4-6 周 |

### 5.2 Eagle + Ngram 混合算法
| 维度 | 分析 |
|------|------|
| **作用** | 结合 Eagle 和 Ngram 的混合推测策略 |
| **实现难度** | ⭐⭐⭐⭐ (较高) |
| **价值评估** | ⭐⭐⭐⭐ (高) |
| **预估工作量** | 3-4 周 |

### 5.3 自适应算法
| 维度 | 分析 |
|------|------|
| **作用** | 运行时动态调整推测参数 |
| **技术难点** | 1. 参数搜索空间<br>2. 自适应策略设计<br>3. 收敛性保证 |
| **实现难度** | ⭐⭐⭐⭐⭐ (很高) |
| **价值评估** | ⭐⭐⭐⭐⭐ (极高) - 智能化推理 |
| **预估工作量** | 6-8 周 |

### 5.4 DLLM Draft Models 支持
- **相关**: SpecForge (#412)
- **贡献者**: @jinleic @yilian49 @xiaomin-D @sleepcoo

| 维度 | 分析 |
|------|------|
| **作用** | 支持扩散语言模型作为 draft model |
| **技术难点** | 1. DLLM 与传统自回归的差异<br>2. 验证机制适配 |
| **实现难度** | ⭐⭐⭐⭐⭐ (很高) |
| **价值评估** | ⭐⭐⭐⭐ (高) - 前沿研究方向 |
| **预估工作量** | 8-10 周 |

---

## 六、PD Disaggregation (Prefill-Decode 分离)

### 6.1 Decode Engine 上的 Radix Cache
| 维度 | 分析 |
|------|------|
| **作用** | 在 decode 引擎支持前缀缓存 |
| **技术难点** | 1. 跨引擎缓存同步<br>2. 一致性保证 |
| **实现难度** | ⭐⭐⭐⭐ (较高) |
| **价值评估** | ⭐⭐⭐⭐⭐ (极高) - PD 分离的核心优化 |
| **预估工作量** | 4-6 周 |

### 6.2 Scheduler Loop 重构
| 维度 | 分析 |
|------|------|
| **作用** | 重构调度循环以复用更多代码 |
| **实现难度** | ⭐⭐⭐ (中等) |
| **价值评估** | ⭐⭐⭐ (中) - 代码质量改进 |
| **预估工作量** | 2-3 周 |

### 6.3 Auto Scaling in OME
| 维度 | 分析 |
|------|------|
| **作用** | 在 Ocean Model Engine 中支持自动扩缩容 |
| **实现难度** | ⭐⭐⭐⭐ (较高) |
| **价值评估** | ⭐⭐⭐⭐⭐ (极高) - 云原生部署关键 |
| **预估工作量** | 4-6 周 |

### 6.4 NIXL and Dynamo Integration
| 维度 | 分析 |
|------|------|
| **作用** | 全面集成 NVIDIA 的 NIXL 和 Dynamo |
| **技术难点** | 1. 新框架学习成本<br>2. 架构适配 |
| **实现难度** | ⭐⭐⭐⭐⭐ (很高) |
| **价值评估** | ⭐⭐⭐⭐⭐ (极高) - NVIDIA 生态整合 |
| **预估工作量** | 8-10 周 |

---

## 七、KV Cache System & Memory Pool

- **PoC**: @xiezhq-hermann
- **Issue**: #12826
- **Slack**: #kv-cache-store

### 7.1 Sparse Attention and KV Cache Scheduler
- **PR**: #11191

| 维度 | 分析 |
|------|------|
| **作用** | GPU/CPU 上的稀疏注意力和 KV cache 调度 |
| **技术难点** | 1. 稀疏模式识别<br>2. 内存层级管理<br>3. CPU-GPU 数据传输优化 |
| **实现难度** | ⭐⭐⭐⭐⭐ (很高) |
| **价值评估** | ⭐⭐⭐⭐⭐ (极高) - 长上下文和内存优化关键 |
| **预估工作量** | 8-12 周 |

---

## 八、Diffusion (Multimodal Generation)

- **PoC**: @mickqian
- **Roadmap**: #12799
- **Slack**: #diffusion

| 维度 | 分析 |
|------|------|
| **作用** | 增强扩散模型支持，包括视频和图像生成 |
| **技术难点** | 1. 扩散模型与 LLM 的差异<br>2. 多步生成优化<br>3. 内存管理 |
| **实现难度** | ⭐⭐⭐⭐⭐ (很高) |
| **价值评估** | ⭐⭐⭐⭐⭐ (极高) - 多模态生成核心 |
| **预估工作量** | 10-12 周 |

---

## 九、Multimodal Models (多模态模型)

- **贡献者**: @mick @JustinTong0323 @yuan-luo
- **Slack**: #multi-modal

> **当前状态更新**：截至 2026 Q1，SGLang 已支持 **30+ 种视觉语言模型** (VL Models)，包括 LLaVA 系列、Qwen-VL 系列、InternVL、Phi-Vision、Gemma-VL 等。此外还支持**完整的扩散生成栈**（图像和视频生成）。

### 9.1 Day-0 Support & OCR Models
| 维度 | 分析 |
|------|------|
| **作用** | 主要模型的首日支持，增加 OCR 模型 |
| **当前状态** | ✅ 30+ VL 模型已支持 |
| **实现难度** | ⭐⭐⭐ (中等) |
| **价值评估** | ⭐⭐⭐⭐ (高) - 模型覆盖度 |
| **预估工作量** | 持续进行 |

### 9.2 Prefix & Embedding Cache 优化
| 维度 | 分析 |
|------|------|
| **作用** | 优化多模态前缀和 embedding 缓存 |
| **实现难度** | ⭐⭐⭐⭐ (较高) |
| **价值评估** | ⭐⭐⭐⭐ (高) - 性能优化 |
| **预估工作量** | 3-4 周 |

### 9.3 Faster CUDA IPC in MQ
- **PR**: #11917

| 维度 | 分析 |
|------|------|
| **作用** | 大视频/图像的更快 CUDA IPC |
| **实现难度** | ⭐⭐⭐⭐ (较高) |
| **价值评估** | ⭐⭐⭐⭐ (高) - 大文件传输优化 |
| **预估工作量** | 2-3 周 |

### 9.4 Omni Support
- **Issue**: #16546

| 维度 | 分析 |
|------|------|
| **作用** | 支持 Omni 模型（统一多模态模型） |
| **实现难度** | ⭐⭐⭐⭐⭐ (很高) |
| **价值评估** | ⭐⭐⭐⭐⭐ (极高) - 前沿方向 |
| **预估工作量** | 6-8 周 |

---

## 十、Quantization (量化)

- **Slack**: #quantization

> **当前状态更新**：截至 2026 Q1，SGLang 已支持 **34+ 种量化方法**，包括 GPTQ、AWQ、SqueezeLLM、FP8、INT8、GGUF、Marlin、Exl2 等主流格式。

### 10.1 通用量化格式支持
- **Issue**: #8180

| 维度 | 分析 |
|------|------|
| **作用** | 支持多种量化格式 |
| **当前状态** | ✅ 大部分已实现 (34+ 方法) |
| **实现难度** | ⭐⭐⭐⭐ (较高) |
| **价值评估** | ⭐⭐⭐⭐⭐ (极高) - 推理成本优化 |
| **预估工作量** | 持续维护 |

### 10.2 ModelOpt Support
- **PoC**: @Edwardf0t1
- **Slack**: #modelopt

| 维度 | 分析 |
|------|------|
| **作用** | 集成 NVIDIA ModelOpt |
| **实现难度** | ⭐⭐⭐ (中等) |
| **价值评估** | ⭐⭐⭐⭐ (高) |
| **预估工作量** | 2-3 周 |

### 10.3 Communication Quantization
| 维度 | 分析 |
|------|------|
| **作用** | FP4/FP8 的 allreduce/allgather/alltoall |
| **技术难点** | 1. 通信压缩算法<br>2. 精度保持 |
| **实现难度** | ⭐⭐⭐⭐⭐ (很高) |
| **价值评估** | ⭐⭐⭐⭐⭐ (极高) - 分布式推理关键优化 |
| **预估工作量** | 6-8 周 |

---

## 十一、Multi-LoRA Serving

- **Major Roadmap**: #2929
- **PoC**: @Fridge003
- **Slack**: #lora

### 11.1 LoRA for Speculative Decoding
- **Issue**: #12903
- **贡献者**: @ConnorLi96 @lifuhuang

| 维度 | 分析 |
|------|------|
| **作用** | 推测解码场景下的 LoRA 支持 |
| **技术难点** | 1. draft 和 target 模型的 LoRA 协调<br>2. 内存管理 |
| **实现难度** | ⭐⭐⭐⭐⭐ (很高) |
| **价值评估** | ⭐⭐⭐⭐ (高) |
| **预估工作量** | 6-8 周 |

### 11.2 Overlap Weight Loading with Compute
- **Issue**: #15512
- **贡献者**: @glenliu21 @ConnorLi96 @lifuhuang

| 维度 | 分析 |
|------|------|
| **作用** | 权重加载与计算重叠 |
| **实现难度** | ⭐⭐⭐⭐ (较高) |
| **价值评估** | ⭐⭐⭐⭐⭐ (极高) - 减少切换开销 |
| **预估工作量** | 4-6 周 |

### 11.3 LoRA for MoE Layers
- **Issue**: #14105
- **贡献者**: @ConnorLi96 @Jonahcb

| 维度 | 分析 |
|------|------|
| **作用** | MoE 层的 LoRA 支持 |
| **技术难点** | 1. 专家路由与 LoRA 结合<br>2. 稀疏更新 |
| **实现难度** | ⭐⭐⭐⭐⭐ (很高) |
| **价值评估** | ⭐⭐⭐⭐⭐ (极高) - MoE 模型必需 |
| **预估工作量** | 8-10 周 |

---

## 十二、Prefill-Only

- **Major Roadmap**: #15344
- **PoC**: @sundar24295s
- **Slack**: #prefill-only

| 维度 | 分析 |
|------|------|
| **作用** | 仅 prefill 模式（如 embedding 服务） |
| **实现难度** | ⭐⭐⭐ (中等) |
| **价值评估** | ⭐⭐⭐ (中) - 特定场景需求 |
| **预估工作量** | 2-4 周 |

---

## 十三、RL Framework Integration (强化学习框架集成)

- **Slack**: #reinforcement-learning, #slime-rl-framework

### 13.1 框架集成
| 框架 | 难度 | 价值 |
|------|------|------|
| AReaL | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| slime | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| verl | ⭐⭐⭐ | ⭐⭐⭐⭐ |

### 13.2 Customized Weight Refitting from RDMA
- **贡献者**: @zhaochenyang20 @JD-ETH

| 维度 | 分析 |
|------|------|
| **作用** | 从 RDMA 自定义权重重适配 |
| **实现难度** | ⭐⭐⭐⭐⭐ (很高) |
| **价值评估** | ⭐⭐⭐⭐⭐ (极高) - RL 训练-推理协同 |
| **预估工作量** | 6-8 周 |

### 13.3 Open Recipe for Large-scale MoE Training
- **贡献者**: @zhaochenyang20 @fzyzcjy @Fridge003 @zyzshishui

| 维度 | 分析 |
|------|------|
| **作用** | 大规模 MoE 训练+GRPO 的开源方案 |
| **实现难度** | ⭐⭐⭐⭐⭐ (很高) |
| **价值评估** | ⭐⭐⭐⭐⭐ (极高) - 行业影响力 |
| **预估工作量** | 10-12 周 |

### 13.4 Training-Inference Mismatch Mitigation
- **贡献者**: @zhaochenyang20 @fzyzcjy @Fridge003 @zyzshishui

| 维度 | 分析 |
|------|------|
| **作用** | 系统性缓解训练-推理不一致 |
| **技术难点** | 1. 差异分析<br>2. 缓解策略设计 |
| **实现难度** | ⭐⭐⭐⭐⭐ (很高) |
| **价值评估** | ⭐⭐⭐⭐⭐ (极高) - RL 效果保障 |
| **预估工作量** | 8-10 周 |

### 13.5 SGLang Gateway as DP Scheduler
| 维度 | 分析 |
|------|------|
| **作用** | 网关作为 RL 中 rollout 的 DP 调度器 |
| **实现难度** | ⭐⭐⭐⭐ (较高) |
| **价值评估** | ⭐⭐⭐⭐⭐ (极高) - RL 基础设施 |
| **预估工作量** | 4-6 周 |

### 13.6 Tinker-like Serverless RL APIs
- **贡献者**: @zhaochenyang20

| 维度 | 分析 |
|------|------|
| **作用** | 类 Tinker 的无服务器 RL API |
| **实现难度** | ⭐⭐⭐⭐ (较高) |
| **价值评估** | ⭐⭐⭐⭐ (高) - 易用性提升 |
| **预估工作量** | 4-6 周 |

### 13.7 Native NVFP8 Training
- **贡献者**: @GeLee-Q @xieck13 @fy1214

| 维度 | 分析 |
|------|------|
| **作用** | 原生 NVFP8 训练支持 |
| **实现难度** | ⭐⭐⭐⭐⭐ (很高) |
| **价值评估** | ⭐⭐⭐⭐⭐ (极高) - 训练效率提升 |
| **预估工作量** | 6-8 周 |

### 13.8 VLM RL with FSDP
- **贡献者**: @nanjiangwill @minleminzui

| 维度 | 分析 |
|------|------|
| **作用** | FSDP 下的视觉语言模型 RL |
| **实现难度** | ⭐⭐⭐⭐⭐ (很高) |
| **价值评估** | ⭐⭐⭐⭐ (高) - VLM RL 方案 |
| **预估工作量** | 8-10 周 |

### 13.9 Speculative Training
- **贡献者**: @guapisolo

| 维度 | 分析 |
|------|------|
| **作用** | 推测训练 |
| **实现难度** | ⭐⭐⭐⭐⭐ (很高) |
| **价值评估** | ⭐⭐⭐⭐ (高) - 训练加速前沿 |
| **预估工作量** | 8-12 周 |

---

## 十四、Diffusion Language Models (DLLMs)

- **PoC**: Zehuan Li, Jinwei Yao, Chenyang Zhao
- **RFC**: Block Diffusion Large Language Model (dLLM) Framework (#12766)
- **Roadmap**: #14199

| 维度 | 分析 |
|------|------|
| **作用** | 扩散语言模型框架 |
| **技术难点** | 1. 扩散模型与自回归的融合<br>2. 并行解码策略 |
| **实现难度** | ⭐⭐⭐⭐⭐ (极高) |
| **价值评估** | ⭐⭐⭐⭐⭐ (极高) - 下一代架构 |
| **预估工作量** | 12-16 周 |

---

## 十五、Hardware Support (硬件支持)

> **时间线说明**：原文档标注 "2025 Q4" 应为历史遗留，本文档为 2026 Q1 Roadmap，这些项目应理解为"进行中"或需核实当前状态。

| 平台 | Timeline | 负责人 | Issue | 难度 | 价值 | 当前状态 |
|------|----------|--------|-------|------|------|----------|
| AMD | 进行中 | @HaiShaw | #12890 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 待核实 |
| TPU | 进行中 | - | sglang-jax#190 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 待核实 |
| NPU | 进行中 | @iforgetmyname @ZhengdQin | #13664 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 待核实 |
| Intel CPU/XPU | 进行中 | - | #12802, #12806 | ⭐⭐⭐⭐ | ⭐⭐⭐ | 待核实 |
| Multi-backend Abstraction | - | @Alcanderian | - | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 核心方向 |

---

## 十六、Model Coverage (模型覆盖)

- **PoC**: @wisclmy0611 @JustinTong0323
- **Slack**: #dev

| 维度 | 分析 |
|------|------|
| **作用** | 所有主要模型的首日支持 |
| **实现难度** | ⭐⭐⭐ (中等，但持续) |
| **价值评估** | ⭐⭐⭐⭐⭐ (极高) - 用户采用关键 |
| **预估工作量** | 持续进行 |

---

## 十七、Model Gateway & API Layer

- **PoC**: @slin1237 @CatherineSue
- **Issue**: #13098
- **Slack**: #router-sig

### 17.1 功能列表与评估

| 功能 | 难度 | 价值 | 预估工作量 |
|------|------|------|-----------|
| 多模态和图像处理器 gRPC 支持 | ⭐⭐⭐ | ⭐⭐⭐⭐ | 2-3 周 |
| PII 和分类 API | ⭐⭐⭐ | ⭐⭐⭐ | 2-3 周 |
| 语义路由 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 3-4 周 |
| 网关监听 KV Cache 事件 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 4-6 周 |
| HTTP + gRPC 同时启动 | ⭐⭐⭐ | ⭐⭐⭐ | 1-2 周 |
| 网关终端 UI | ⭐⭐⭐ | ⭐⭐⭐ | 2-3 周 |
| Reactive UI 远程启动 workers | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 3-4 周 |
| Anthropic Message API 原生支持 | ⭐⭐⭐ | ⭐⭐⭐⭐ | 2-3 周 |
| Gateway SDK (Go/Python/Node.js) | ⭐⭐⭐ | ⭐⭐⭐⭐ | 4-6 周 |
| 指标增强 (TTFT, TPOT 等) | ⭐⭐⭐ | ⭐⭐⭐⭐ | 2-3 周 |

---

## 十八、Tracing and Profiling

- **Roadmap**: #13511

| 维度 | 分析 |
|------|------|
| **作用** | HiCache、PP、SD 的请求追踪 |
| **实现难度** | ⭐⭐⭐⭐ (较高) |
| **价值评估** | ⭐⭐⭐⭐ (高) - 可观测性 |
| **预估工作量** | 3-4 周 |

---

## 十九、Advanced Priority Scheduling

- **Issue**: #13526
- **PoC**: @harrisonlimh

| 维度 | 分析 |
|------|------|
| **作用** | 高级优先级调度 |
| **实现难度** | ⭐⭐⭐⭐ (较高) |
| **价值评估** | ⭐⭐⭐⭐ (高) - 服务质量保障 |
| **预估工作量** | 4-6 周 |

---

## 二十、CI / Release / Maintenance

- **PoC**: @alisonshao @Kangyan-Zhou
- **Slack**: #ci-cd-build-release, #help-desk

### 20.1 任务列表

| 任务 | 难度 | 价值 | 预估工作量 |
|------|------|------|-----------|
| CI Suites 重构 (#13808) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 4-6 周 |
| CI Monitor 工作流改进 | ⭐⭐⭐ | ⭐⭐⭐⭐ | 2-3 周 |
| Nightly Tests 增强 | ⭐⭐⭐ | ⭐⭐⭐⭐ | 持续进行 |
| Full Feature Coverage CI | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 4-6 周 |
| B300/GB200 硬件覆盖 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 2-3 周 |

---

## 补充：已发现但未列入 Roadmap 的特性

以下特性在代码库中已实现，但未在原 Roadmap 文档中明确列出：

### Prefill Delayer 系统

**代码路径**: `python/sglang/srt/managers/prefill_delayer.py`

**功能描述**：
- 智能延迟 Prefill 请求，优化整体吞吐量
- 基于 token 使用率水位线（`token_usage_low_watermark`）做决策
- 支持配置最大延迟轮数（`max_delay_passes`）
- 与 DP Attention 调度器集成

**启用方式**：
```bash
--enable-prefill-delayer
--prefill-delayer-max-delay-passes N
--prefill-delayer-token-usage-low-watermark 0.8
```

**相关集成**: `scheduler.py:767-778`, `schedule_policy.py:385-431`

### Batch Overlap 系统

**代码路径**: `python/sglang/srt/batch_overlap/`

**功能描述**：
- **Single Batch Overlap (SBO)**: 单批次内的计算重叠优化
  - 支持 `combine_down_gemm_two_stream_overlap`
  - 与 FlashInfer CuteDSL 和 DeepGEMM 后端集成
- **Two Batch Overlap (TBO)**: 双批次之间的流水线重叠
  - `TboDPAttentionPreparer` 用于 DP Attention 场景

**相关文件**：
- `single_batch_overlap.py` - SBO 标志位和配置
- `two_batch_overlap.py` - TBO 实现
- `operations.py` / `operations_strategy.py` - 重叠操作策略

### HiCache 多级存储后端

**代码路径**: `python/sglang/srt/mem_cache/`

**功能描述**：
- 支持 GPU → CPU → NVMe 的多级 KV Cache 存储
- 稀疏注意力优化（`mem_cache/sparsity/`）
- 与 PD Disaggregation 配合使用

### 量化方法统计

原 Roadmap 提到"通用量化格式支持"，实际代码库已支持 **34+ 种量化方法**（截至 2026 Q1）。

### 多模态支持统计

- **VL 模型**: 30+ 种视觉语言模型支持
- **扩散生成**: 完整的图像/视频生成栈（参见 Diffusion 章节）

---

## 总结与建议

### 按优先级分类

#### P0 (最高优先级 - 生产必需)
1. **Server Reliability** - 非法内存访问修复、崩溃容错
2. **Base Engine** - Overlap Scheduler、DP Attention 重构
3. **Parallelism** - PP/EP 重构、GB200/GB300 优化
4. **PD Disaggregation** - Radix Cache、NIXL/Dynamo 集成

#### P1 (高优先级 - 性能关键)
1. **Kernel** - FlashInfer 集成、Communication Kernel
2. **Speculative Decoding** - 通用抽象、自适应算法
3. **Multi-LoRA** - MoE LoRA、Weight Loading Overlap
4. **RL Framework** - 核心集成、NVFP8 Training

#### P2 (中优先级 - 功能扩展)
1. **Multimodal** - Day-0 支持、Omni
2. **Quantization** - 通用格式、Communication Quantization
3. **Gateway** - 核心功能、API 支持
4. **Hardware** - AMD/TPU/NPU 支持

#### P3 (长期方向 - 前沿探索)
1. **Diffusion** - 视频/图像生成
2. **DLLMs** - 扩散语言模型
3. **Torch Compile** - 长期性能优化
4. **Speculative Training** - 训练加速

### 关键技术依赖

```
Base Engine (SRT Core)
    ↓
Parallelism (PP/EP/CP/DP) → Kernel Optimization
    ↓
PD Disaggregation ↔ KV Cache System
    ↓
RL Framework ← → Gateway
    ↓
Multimodal + Diffusion
```

### 风险提醒

1. **高复杂度项目**: Torch Compile、DLLMs、RL Framework Integration 需要长期投入
2. **硬件依赖**: GB200/GB300 优化需要实机测试
3. **兼容性问题**: All Parallelisms + Speculative Decoding + PD 的组合复杂度极高
4. **人力瓶颈**: 多个项目寻找 PoC，需要更多贡献者

### 推荐关注项目

对于新贡献者，推荐从以下项目入手：
1. Model Coverage (Day-0 support)
2. Server Reliability (grammar crash tolerance)
3. Model Gateway (API support)
4. Tracing and Profiling

对于核心开发者，重点关注：
1. Base Engine Core/Plugin 重构
2. Parallelism 兼容性与优化
3. PD Disaggregation 稳定性
4. RL Framework 集成
