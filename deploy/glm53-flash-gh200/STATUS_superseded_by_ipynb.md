# GLM-5.3-Flash 这条线做到哪了

任务清单在 `glm53_flash/results/REPRO_PLAN.md`：用 GLM-5.3-Flash 当提案模型，
复现 LDM v0.1 论文（[arXiv:2608.15669](https://arxiv.org/abs/2608.15669)）的主要结论。

~~划掉的是做完的~~；没做完的加粗，每一项配一个 notebook。

---

## 一句话

服务部署完了，C1 也真跑完了 —— **但没复现出论文的结论**。
论文说 LDM 比 LLM-only 好 2.4 倍，我们量出来是 **0.97 倍**，三个种子里只有一个
LDM 领先。C2 和 C3 从一开始就跑不了，卡在依赖和数据上。

## 总览

| 项 | 状态 |
|---|---|
| GLM-5.3-Flash 部署（TP4，OpenAI 兼容） | ~~做完~~（就是本 PR） |
| C1 nanoGPT 复现：跑起来 | ~~做完~~ 6 条轨迹，每条 40 次真评测 |
| C1 复现：**跟论文对上** | **没有** · [01](status_notebooks/01_c1_ratio_gap.ipynb) |
| C2 抗体 CDRH3 | **跑不了** · [02](status_notebooks/02_c2_antibody_blocked.ipynb) |
| C3 小分子多目标 | **跑不了** · [03](status_notebooks/03_c3_small_molecule_blocked.ipynb) |

---

## ~~部署~~

~~官方 arm64/cu129 镜像 + apptainer，TP4，端口 8383，OpenAI 兼容。
FP8 权重 306 GiB，每卡约 77 GiB，热缓存下一分钟内加载完。~~

~~pip 和源码构建都试过，都不行。pip 装得上但没有 `Glm5Next` 架构，
vLLM 不吭声地退回 transformers 后端，很久之后死在 KDA 的 `k_conv1d`。
源码构建六轮没成，第 5、6 轮 CUTLASS 报的错看着像 CUDA 版本问题，
其实是 `-ccbin` 没传、宿主编译器落到了 gcc-7。~~

~~容器里四个坑，全部写进了 `serve_vllm.sh`，`ci/check_glm53_deploy.py` 盯着它们
不被人顺手删掉。细节见 [README.md](README.md)。~~

## C1 nanoGPT：~~跑完了~~，但**结论对不上**

~~设计是成对的：A 组开 GP 引导（`acquisition-feedback=brief`），B 组关掉
（`none`），其余全部一样 —— 同一个提案模型、同一份起始代码、
`breadth=1 depth=1`、`warmup=5`、`iterations=40`、每次评测真训练 300 秒。
3 个种子 × 2 组 = 6 条轨迹，每条独占一张卡。~~

~~三轮起跑暴露的四个缺陷都修了：服务和实验分作业导致空跑、
`breadth×depth` 被误当成打分候选数（害 A 组多跑 7.7 倍评测）、
自设 `CUDA_VISIBLE_DEVICES` 指向不存在的卡、起服漏了 tool calling 标志。~~

**没做完的是解释这个差距**：

```
seed 1:  Δ_A/Δ_B = 0.92×
seed 2:  Δ_A/Δ_B = 1.00×
seed 3:  Δ_A/Δ_B = 0.98×
均值 0.97×          论文 2.4×
```

→ [01_c1_ratio_gap.ipynb](status_notebooks/01_c1_ratio_gap.ipynb)

三个种子挤在 0.92–1.00 之间，看不出 GP 引导有什么用。差距是真的，
但**差在哪还不知道**：可能是 GLM-5.3-Flash 跟论文用的模型不一样，
可能是 40 轮不够（论文跑 100 轮），也可能是 `acquisition-feedback` 这个开关
根本不是论文说的那件事。这三条都没验。

## C2 抗体：**跑不了**

→ [02_c2_antibody_blocked.ipynb](status_notebooks/02_c2_antibody_blocked.ipynb)

两道坎。Absolut! 那个结构模拟器要单独装，是外部二进制。
更麻烦的是 `ldm-tts-antibody` 钉死 transformers 4.13，它拉的 tokenizers 0.10.3
在 aarch64 上没有轮子，得现编 Rust。这条已经实测失败过（TEST_REPORT E3）。

## C3 小分子：**跑不了**

→ [03_c3_small_molecule_blocked.ipynb](status_notebooks/03_c3_small_molecule_blocked.ipynb)

活性模型 `best_g12d_model.joblib` 不公开发，得找论文作者要。
Vina 二进制这边已经有了（小分子 RL 那条线在用），但没有那个活性模型，
C3 的评测口径就和论文对不上，复现无从谈起。

---

## 东西在哪

| | |
|---|---|
| 复现计划与设计 | `glm53_flash/results/REPRO_PLAN.md` |
| 执行流水账 | `glm53_flash/results/RUNLOG.md` |
| C1 六条轨迹的数 | `glm53_flash/results/c1_repro.json` |
| 分析脚本 | `glm53_flash/code/analyze_repro.py` |
| 服务脚本与四个坑 | 本目录 |
