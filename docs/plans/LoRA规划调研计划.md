# LoRA规划调研

## 参考
+ sglang roadmap: [https://github.com/sgl-project/sglang/issues/2929](https://github.com/sgl-project/sglang/issues/2929)
+ 论文: s-lora, punica

## 调研和确认的具体方向
1. 内存池与Lazy加载机制
    1. 预取、淘汰优化
        1. [https://github.com/sgl-project/sglang/issues/8712#issuecomment-3535979430](https://github.com/sgl-project/sglang/issues/8712#issuecomment-3535979430)
        2.
2. kernel
    1. sglang当前可选：triton和csgmv, 未来cutlass/cuda： https://docs.sglang.ai/advanced_features/lora.html
    2. [https://github.com/sgl-project/sglang/issues/7910](https://github.com/sgl-project/sglang/issues/7910)
    3. [https://github.com/sgl-project/sglang/issues/9040](https://github.com/sgl-project/sglang/issues/9040)
3. 改进多LoRA调度机制
    1. lora-aware
4. 交叉方向
    1. support lora for speculative decoding
5. 差异化lora共享
    1. 不同层不同秩的场景深入
        1. attn, fnn, embedding...
    2. 文献（待筛选）
        1. [https://arxiv.org/abs/2501.15103](https://arxiv.org/abs/2501.15103)
        2. [https://arxiv.org/abs/2505.22694](https://arxiv.org/abs/2505.22694)
        3. [https://arxiv.org/abs/2408.03290](https://arxiv.org/abs/2408.03290)
        4. [https://arxiv.org/abs/2403.03432](https://arxiv.org/abs/2403.03432)
        5. [https://arxiv.org/abs/2505.14468](https://arxiv.org/abs/2505.14468)
6. 模型适配
    1. moe
    2. vl

## 任务

根据上面给出的这些信息，给出一封在sglang上在lora方向上的技术规划、roadmap。
可以有多个提案，但是每个提案要包括现状分析、预计收益、投入产出比预估或者其他你认为必要的信息。
输出内容最后全部写到 TODO/lora_plan.md 中




