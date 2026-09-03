# 交接的活干到哪了

对着 [`HANDOFF.md`](HANDOFF.md) 一条一条说：哪些做完了，哪些没做完，没做完的卡在哪。
机器是 Isambard-AI（GH200，aarch64，驱动 565 / CUDA 12.7，分区 `workq`）。

~~划掉的做完了~~；**没做完的加粗，点进去是一个 notebook**，里面有图、有数、
还有一句「怎么算过了」——那句话是在测之前写的，不是看完结果再补的。

---

## 一眼看完

| HANDOFF | 要做的事 | 现在 |
|---|---|---|
| §0 | 机器够不够 | ~~够~~ |
| §1 | torch 只能用 cu128/cu129 | ~~照做了~~ |
| §2A | 训练那套（torch / TE / Megatron / Slime / sglang） | ~~都装上了~~ |
| §2B | 打分那套（docking + GP + 活性模型） | ~~都装上了~~ |
| §2C | 代码本体 | ~~拉下来了~~ |
| §3 | 模型、vina、活性模型、受体 | ~~六样都有~~ |
| §4 | 转格式、生成 episodes、暖 GP | ~~都跑过了~~ |
| §5 | 四个 run，每个 ≥3–5 个种子 | **没做** · [03](handoff_notebooks/03_p2_training_matrix.ipynb)；而且 `--seed-offset` 这个参数压根不存在 · [06](handoff_notebooks/06_seed_offset_missing.ipynb) |
| §6 P0 | 1.5B 跑通，验 backward + docking + **reward** | 前两样 ~~过了~~，**reward 这样没过** · [01](handoff_notebooks/01_reward_always_zero.ipynb) |
| §6 P1 | 9B 小规模冒烟 | ~~冒烟成过一次~~，但 **29 个 9B run 没一个活下来** · [02](handoff_notebooks/02_9b_run_stability.ipynb) |
| §6 P2 | R1–R4 × 种子，正式训练 | **一个都没起** · [03](handoff_notebooks/03_p2_training_matrix.ipynb) |
| §6 第 4 步 | 在 C 和 D 上离线打分 | **没开始**；~~它要的 QSAR 模型倒是训好了~~ · [04](handoff_notebooks/04_offline_evaluation.ipynb) |
| §7 | 六个已知的坑 | ~~都撞过一遍，确认属实~~，另外又踩出三个（见最后） |
| — | GP 用哪个 kernel、EHVI 会衰减 | HANDOFF 没写，但 P2 之前必须定 · [05](handoff_notebooks/05_gp_kernel_and_ehvi_decay.ipynb) |

**一句话**：装机器、备数据这些全做完了，1.5B 能从头跑到尾，9B 也冒烟成功过一次。
**GRPO 拿不到梯度的原因已经查清并可以修**——不是 reward 坏了（reward 好得很，
1097 步里 89% 都在 1e-6 以上，中位数 1e-2），是**一组里几条轨迹拿到的 reward
一模一样**，除以组内标准差就等于除以 0。

成因定了：**组里只有两条轨迹**。把 `n_samples_per_prompt` 从 2 提到 8，
零方差组从每步 0.87 降到 0.00（四档实测，见 §6 P0）。**P2 起跑前把这个值
设成 8 即可**，不需要 16。

---

## §0 机器

- ~~`uname -m` 是 aarch64，驱动 565 / CUDA 12.7，4 张 GH200 每张 96 GB~~
- ~~盘够：模型、torch_dist、中间产物都放得下~~
- ~~能连 HuggingFace~~

## §1 CUDA 那条红线

- ~~torch 装的是 `+cu128`，`torch.cuda.is_available()` 是 True~~

这条我们在别的项目上撞过：cu130 的轮子跨大版本一定挂，`is_available()` 直接
返回 False，和 HANDOFF 写的一样。

## §2 装环境

### 2A 训练那套
- ~~torch（aarch64，cu128）~~
- ~~Transformer Engine：装上了，**而且反向传播不 SIGSEGV**（P0 已经验过）~~
- ~~Megatron-LM 切到 `1dcf0daf`~~
- ~~Slime + ray~~
- ~~sglang（aarch64 能用的版本）~~
- ~~APEX 没装，脚本走 `--no-gradient-accumulation-fusion`，HANDOFF 说可以省~~

TE 是 HANDOFF 特意点出来「ABI 最容易出问题」的一环。实测在 aarch64 上装得上，
反向也正常——**HANDOFF 担心的那件事没发生**。

### 2B 打分那套
- ~~gpytorch / gauche / lightgbm / scikit-learn / joblib / rdkit / meeko / gemmi 这些~~
- ~~vina 的 aarch64 二进制，路径填进了 `config_real.json:vina_bin`~~
- ~~`vina_max_workers` 按 288 核调过了~~

实测一次 `env.step` 4.09 秒，和 HANDOFF 说的差不多。

### 2C 代码本体
- ~~clone、submodule、PYTHONPATH 都弄好了~~

## §3 要准备的东西

| 东西 | 现在 |
|---|---|
| base Qwen3.5-9B | ~~下好了~~ |
| SFT 模型（LDM-CoT-SFT） | ~~下好了~~ |
| 冒烟用的 Qwen2.5-1.5B-Instruct | ~~下好了~~ |
| vina 二进制（aarch64） | ~~装好了~~ |
| G12D 活性模型 | ~~repo 自带~~ |
| 8UN5 受体 | ~~有~~ |

## §4 一次性的准备工作

- ~~`convert_9b.sh` 转 Megatron torch_dist，base 和 SFT 各转一次，两个目录都在~~
- ~~`gen_episodes_runs.sh` 生成四个 run 的 episodes~~
- ~~`run_warmup_real_slime.sh` 暖 GP，得到 63 行、41 个不重样的分子~~

后来各个 run 又跑出 **1751 次真实评测**，去重之后分子池现在有 **1493 个**。

## §5 怎么跑

- ~~单节点 4 卡的 TP、rollout-gpus、actor-gpus 都按 4 卡改过了~~
- ~~sbatch 能提交~~（实际是 attach 到已有的分配上跑，原因见最后第 2 条）
- **每个 run ≥3–5 个种子**：没做，见 [03](handoff_notebooks/03_p2_training_matrix.ipynb)
- **`--seed-offset` 这个参数不存在**：见 [06](handoff_notebooks/06_seed_offset_missing.ipynb)

## §6 训练计划

### P0：1.5B 跑通 —— 三样过了两样

- ~~ARM 上 backward 不 SIGSEGV~~
- ~~docking 从头到尾能跑~~
- **reward 这样没过** → [01](handoff_notebooks/01_reward_always_zero.ipynb)

  **从头跑到尾不报错，不等于模型在学东西。**

  先说我自己看错的一个地方：我一直盯着 `rollout/rewards`，看它总在 1e-8 上下，
  就断定 reward 是零。**那个字段不是环境给的 reward**，是 GRPO 归一化之后的
  advantage，`(r − mean) / std`。环境 reward 在同一个字典里，叫 `raw_reward`。

  看对地方之后：**43 个 run、771 步，环境 reward 有 86% 在 1e-6 以上，
  中位数 1e-2**，正是 EHVI 该有的大小。reward 一点毛病没有。

  出问题的是**一组里几条轨迹的 reward 全一样**。GRPO 要拿组内的标准差去除，
  几条一样就是除以 0，这一组对参数更新一点贡献都没有——reward 再大也白搭。
  771 步里这样的组累计出现 **565 次**。

  **本来以为是解析失败导致的，测下来不是。** 之前发现有八成的轮次解析不出
  候选（环境给的观测直接接在模型上一轮回答后面，把 chat 格式弄坏了，
  从第 1 轮起模型就不再输出 JSON），一组里全失败就全拿 0。把观测包成
  一个独立的 user turn 之后：

  | | 旧代码 | 新代码 |
  |---|---:|---:|
  | 解析失败率 | 78.3% | **0.4%** |
  | 每步零方差组 | 0.68 | **0.82** |

  **解析失败基本清零了，零方差反而略升。** 按事先写好的那句标准，
  这说明解析失败不是零方差的原因，该去查 `n_samples_per_prompt`。

  **那个测量做完了，成因定了**——四档除组内轨迹数外一切相同，同期起跑：

  | n_samples | run 数 | 步数 | 零方差组/步 |
  |---:|---:|---:|---:|
  | 2 | 3 | 52 | **0.87** |
  | 4 | 2 | 32 | **0.28** |
  | 8 | 8 | 24 | **0.00** |
  | 16 | 6 | 15 | **0.00** |

  **0.87 → 0.28 → 0.00 → 0.00，到 n=8 就完全归零。** 道理很直白：
  GRPO 的梯度来自组内差异，一组只有两条轨迹时，两条撞出同一个分子
  就足以让标准差为 0。

  **对 P2**：`n_samples` 直接乘在每步的 rollout 成本上，4 已压掉 68%，
  8 完全消除，**16 没有额外收益、只是更贵**。

  （中间我报过一次「零方差降了 43%」，那是新代码只跑了 36 步时的读数，
  跑到 180 步就翻过来了。统计时已经改成只算跑够 15 步的 run——
  只跑几步的 run，这个数噪声大到没意义，0.00 和 1.00 都出现过。）

### P1：9B 冒烟 —— 冒烟成了，但 run 活不下来

- ~~hybrid 的 backward 和显存都验过~~：双节点分卡的摆法下 `needs_offload=False`，
  绕开了 `torch_memory_saver` 那个断言；`R2-nonanguard` 一个 run 就跑出 500 次真实评测
- **29 个 9B run 到现在一个都没活下来** → [02](handoff_notebooks/02_9b_run_stability.ipynb)

  45% 是显存被邻居占满，21% 是**主机内存被 OOM killer 杀掉**（上限在 job 这一级的
  cgroup 上，同一个分配里所有 step 一起分 449 GB）。后面这种起跑前能查出来，
  前面那种在 `nodelock` 被禁之后没法预留，只能认。

### P2：R1–R4 × 种子 —— **一个都没起**

→ [03](handoff_notebooks/03_p2_training_matrix.ipynb)

技术上该弄的弄好了七项（kernel 选定、分卡摆法、跨分配起 run、每个 run 独立的
GP/种子/输出目录、编排能活过会话、排掉自己在用的节点、起跑前查主机内存）。
**挡着的是三件**：GRPO 拿不到梯度、9B 活不下来、以及凑不出 16 个空节点。

### 第 4 步：在 C 和 D 上离线打分 —— **没开始**

→ [04](handoff_notebooks/04_offline_evaluation.ipynb)

- ~~HANDOFF 说可以并行做的那件前置（`train_g12c_qsar.py` 训 QSAR）已经做完~~
  （`g12c_qsar_20260901T010923Z/best_model.joblib`，还有配套的 G12D 那个）
- 缺的只是**要打分的东西**——P2 还没训出检查点

## §7 HANDOFF 列的六个坑，都撞过

- ~~cu130 确实不能用~~
- ~~TE 在 aarch64 上装得上，1.5B 冒烟通过~~
- ~~9B 是 hybrid，转换和训练都用 `qwen3.5-9B.sh` 的 spec，走通了~~
- ~~显存宽裕：分卡之后每个 rank 37.06 GiB，和算出来的一分不差~~
- ~~docking 吞吐：`vina_max_workers=32` 生效，aarch64 的 vina 正常~~
- ~~代码本身与架构无关：`ldm_rl` 的 61 个测试在这台机器上全过~~

### 我们又踩出三个

1. **`--rollout-num-gpus 4` 会起四个各自独立的 sglang 引擎**，每个都要把整个 9B
   读进主机内存——主机内存被杀就是这么来的。改成两个能减一半，代价是推理吞吐减半。
2. **`setsid nohup` 保不住 Slurm step。** 它只让编排脚本脱离会话，可 srun 是这个
   脚本的子进程，而 step 的命跟着 srun 走。会话一换，这样起的 run 全被
   `srun: forcing job termination` 收走。得放进 tmux。
3. **`ehvi_all.py` 里有一个 `except Exception: return _fallback(...)`**，
   GP 但凡出点问题，它就悄悄返回一整片 0 的 EHVI。这次的证据说明它没被触发过
   （`fallback_reason` 全是 None），但它是个会把建模问题装扮成「reward 就是 0」的地方。

---

## 六个 notebook 分别讲什么

| # | notebook | 讲什么 |
|---|---|---|
| 01 | [GRPO 拿不到梯度](handoff_notebooks/01_reward_always_zero.ipynb) | reward 是好的（86% 在 1e-6 以上），是一组里几条轨迹的 reward 全一样；565 次 |
| 02 | [9B run 活不下来](handoff_notebooks/02_9b_run_stability.ipynb) | 两种 OOM 得反着修；job 级 cgroup 那 449 GB 是怎么分的 |
| 03 | [P2 矩阵](handoff_notebooks/03_p2_training_matrix.ipynb) | 十项准备各是什么状态；这个矩阵要多少机器、跑多久 |
| 04 | [离线打分](handoff_notebooks/04_offline_evaluation.ipynb) | 前置已经做完；不等 P2 也能先做的两件事 |
| 05 | [GP kernel 和 EHVI 衰减](handoff_notebooks/05_gp_kernel_and_ehvi_decay.ipynb) | sk 涨得太快（2.68 次方），生产规模跑不完；EHVI 会随历史变 0 |
| 06 | [`--seed-offset` 不存在](handoff_notebooks/06_seed_offset_missing.ipynb) | 有三个 seed，管的是三件不同的事；HANDOFF 该怎么改 |

完整的过程记录（每条结论的证据，以及后来被自己推翻的那些）在
`/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/ldm_rl/results/FINDINGS.md`。

改掉的四个代码问题已经开了 PR：
<https://github.com/YihangChen9/Large-Discovery-Models/pull/1>
