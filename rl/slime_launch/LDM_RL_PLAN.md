# LDM-RL(小分子 acquisition-RL)在 Isambard-AI 上的落地计划


> ## 当前状态 2026-09-05
>
> | | |
> |---|---|
> | **主目标** | SFT+RL vs SFT-only,主指标 **budget=80 的 Pareto hypervolume**,G12D(同分布)+ G12C(迁移) |
> | **主目标状态** | **未做**。`results/` 里 budget=80 的评测产物为 **0** 个 |
> | **阻塞缺口** | **没有任何 9B run 完成过一个梯度有限的优化步** —— 七个有 checkpoint 的 9B run,238 条 grad_norm **全部 nan/inf** |
> | **1.5B** | 只是**流程 pilot**,不计入主目标 |
> | **旧 lr=0 对照** | 1.5B 上的**诊断**,结论有限见下;与 hypervolume 无关 |
> | **下一决定性步骤** | 让一个 9B run 记录到有限梯度。这是**训练问题**,不是评测问题 |
> | **CPU 侧** | 评测链依赖已修好并验证(`code/cpu_eval_prereqs.sh`);未解决:8UN5 受体文件在深度 4 内找不到 |
>
> 报告:`plan/MAIN_OBJECTIVE_AUDIT.md`、`plan/MINIMAL_RL_GAIN_TEST.md`、
> 机器可读 `results/main_objective_manifest.json`。
> **以下正文为历史记录,日期见各节;与上表冲突时以上表为准。**

> **2026-09-04 05:00 更新。** 下面第 0 节是当前状态与未来 12 小时的用卡安排；
> 第 1 节起是原始的落地计划，其中「§2 aarch64 逐项落地表」已全部完成，
> 「§5 已知会踩的坑」在实践中又长出十几条，全部记在
> `/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/ldm_rl/results/FINDINGS.md`。

---

## 0. 当前状态(2026-09-04)

### 0.1 里程碑:1.5B 完整训练跑通

`RUN2_20260904T050732Z` **跑满 30/30 轮,120/120 个优化步全部生效,0 个被 nan 跳过**,
存下 3 个检查点。完整路径:

```
/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/ldm_rl/runs/RUN2_20260904T050732Z
```

配置:`n_samples_per_prompt=4`、acquisition reward、`kernel=fp`、每 run 独立 GP(从同一份
63 行暖机结果拷贝)。`grad_norm` 中位 6.21。

~~**但要说清楚**:`raw_reward` 前半 0.0172 → 后半 0.0023,**是降的**。
训练完整跑完、每个梯度都真正应用了,但**没有证据说策略变好**,反而有证据说变差。~~

> **撤回 2026-09-05。** 「下降 ⇒ 策略变差」不成立。把学习率设为 0、并在字节层面确认
> 226/226 个参数张量跨 2,944.8 MiB 完全不变之后,**同样的下降照样出现**:冻结组
> Spearman ρ = +0.0946 (n=4),训练组 +0.0286 (n=7),差异 −0.0660,95% CI
> [−0.1475, +0.0155],p = 0.0961。**一个不可能改变的策略复现了这个下降**,所以下降的
> 来源在采集过程,不在权重。
>
> 原文的前一半仍然成立并保留:训练确实完整跑完、梯度确实被应用。错的是后一半的归因。
>
> 附带限定,一并记住:那个对照是 **1.5B**,是**两个被条件选择的观测组之间的描述性
> 差异**(两组各自按 ≥400 提议筛选,而筛选标准位于处理下游),**不是共同人群上的处理
> 效应**;等效性未确立(TOST 上侧 p=0.0011 拒绝「训练比对照高 0.10 以上」,下侧
> p=0.176 拒绝不了反向);在冻结界上功效不足(分辨力 0.108 对 0.10)。
> 完整记录见 `plan/PRESPEC_lr0_control.md`。
>
> **而且这整条都不是主目标。** 主目标是 SFT+RL 是否优于 SFT-only,主指标是
> budget=80 上的 Pareto hypervolume,在 G12D 与 G12C 上分别评测。
> `raw_reward` 的 Spearman 与 zero_std 都替代不了它。见
> `plan/MAIN_OBJECTIVE_AUDIT.md`。

### 0.2 四个阻塞的真实状态

| 阻塞 | 判据 | 状态 |
|---|---|---|
| inode 耗尽 | 单 run 887 个文件里 **877 个是 docking 暂存** | ✅ **真解决**:改写节点本地(`/local/user/$(id -u)/ldm_env_out/`),每 run 3 个 Lustre 文件。这本来就是 CLAUDE.md 的规定,只是这条线从没执行过 |
| 把良性字符串当死亡信号 | `AttributeError` 在 sglang 的 HTTP 指标中间件里,run 还在训练 | ✅ **真解决**:判据改成「致命信号 **且** 日志静默 5 分钟」。不依赖穷举失败字符串——而穷举这件事我已证明做不到(两次误报) |
| 落到有邻居的节点 | squeue 说无人认领,卡上有 87 GiB | ⚠️ **缓解,非解决**:起跑前逐卡验证 4/4 真空。但预检只能验证**起跑那一刻**,不能预留;启动窗口 15 分钟,邻居有一刻钟可以进来(RUN22 就是这样死的) |
| sglang 死锁 | `pause_generation`/`continue_generation` 计数 **2/0**(健康的是 62/62) | ⚠️ **此前归因错了**:挂死的 `FULL-acq-n4` 和跑成的 `DONE-acq-n4` **用的都是 `mem_fraction_static=0.7`**。真正的差别是节点上有没有邻居——它和上一条是同一个问题 |

### 0.3 唯一未解:9B 的梯度爆炸

| | 1.5B | 9B |
|---|---:|---:|
| `grad_norm` 中位 | **6.21** | **5.1e7**(唯一一个没被 nan 跳过的步) |
| 优化步生效率 | 4536/4536 = **100%** | 37/269 = **13.8%** |

loss 全程有限(1e-4 到 4e-2),所以故障在**反向**。诊断显示 `linear_attn.A_log` 与
`dt_bias`(GDN 的状态空间参数)**每次都 32/32 全 nan**,而第 27 层的 `linear_proj`
梯度是**有限的**(5.25e5)——**nan 不是从后面传回来的,是在特定层里生成的**。

**已排除的七个假说**(每一个都有实测,不是推理):

| 假说 | 怎么排除的 |
|---|---|
| fla 的 `chunk_gated_delta_rule` kernel 有 bug | 探针**按真实调用点复刻**(`use_qk_l2norm_in_kernel=True` + `cu_seqlens`)后,单序列/varlen/各长度/三种子全部正常 |
| GDN 模块本身有数值缺陷 | 用**真实模块** `Qwen3_5GatedDeltaNet` 隔离测,4 种长度 × 3 种输入尺度,**零 nan** |
| 24 层 GDN 复合放大 | 堆 1/2/4/8/16/24 层测第 0 层梯度:**1.0× 无复合**(预测的 1.7e7 倍不存在) |
| `l2norm` 在全零行(padding)上的反向奇点 | fla 的 l2norm 在全零/极小行上梯度有限(1e3 = 1/eps),**无 nan** |
| 词表与检查点不匹配 | 124160 × TP2 = 248320 = `padded_vocab_size`,一致 |
| advantage 过大 | 9B 与 1.5B 都是 ~1e-8 |
| 两个后端 logprob 分歧 | R3* 是 0.25 而其余 0.005,看似判别因素;**SN2–SN5 分歧全是 0.008 正常值,照样 0–1/20 生效** |
| 配置交叉(base HF 配置 + SFT 权重) | SN3–5 已修成 `hf_checkpoint=LDM-CoT-SFT`,照样失败 |

**探针给出的正面结论**:GDN 是个**高增益放大器**——输入放大 50 倍,梯度就到 2.3e5,
已是真实 run 里 5.25e5 的量级。所以问题从「哪个部件坏了」变成了「**输入为什么那么大**」。
而探针用的是**随机初始化的权重**,真实 run 用的是 SFT 训练过的权重——这是剩下的最大差别。

**下一步**:`G1`–`G4` 四个 9B run 带层定位诊断在跑
(`SLIME_REPORT_NONFINITE_GRADS=1`,两档 `--clip-grad` 1.0 / 0.01)。
它会报出**编号最大的坏层**——因为反向会把 nan 带到所有更早的层,那个编号才是起点。

### 0.4 未来 12 小时的用卡安排

此刻 80 张卡只跑 15 张(19%),60 张空转 6.07 kW。名下 20 个节点,walltime 分三档。

| 分配 | 节点 | 剩余 | 轨道 | 跑什么 |
|---|---:|---:|---|---|
| `6266773` / `6266774` | 8 | 3–4h | **1** | `G1`–`G4`:9B 层定位诊断。短 walltime 够用——诊断在第 0 步就出 |
| `6282177` / `6269978` | 8 | 9–11h | **2** | `M-*` ×8:2×2 矩阵每格 2 份,各跑满 30 轮 |
| `6282179` | 4 | 11h | **3** | 长预算 run + base/SFT 无 RL 参照 |
| 任何 ≥3 张真空卡的节点 | — | — | **4** | 自动补位,上限 12 个并行 |

合计约 **152 节点·小时 = 608 GPU·小时**。

**轨道 2 为什么值得占 8 个节点**:修正后的 2×2 里**三格只有 1 个 run**,
而且 `acquisition + n=4` 那格测出的「零方差 0.000」是 **22 步短跑**的乐观值——
`RUN2` 完整跑完实测是 **13/30 轮**。一段序列的开头不代表整体,这是今天反复踩的同一件事。

**轨道 3 为什么必须做**:`TRAINING_PLAN.md` 里的最终评测(G12C/G12D 各 5 seed)
**包括 base(无RL)和 SFT(无RL)两行参照,它们不需要任何训练**。
在拿到「5 个 eval seed 之间的 HV 散布」和「SFT 相对 base 差多少」之前,
「R2 > SFT(无RL)」是一句**不可证伪**的话,而 P2 是 20 次 4×80G 的训练。

**补位为什么用「≥3 张真空卡」而不是「整节点空」**:1.5B 启动器本来就用
`CUDA_VISIBLE_DEVICES=1,2,3`(跳过 GPU 0),所以 GPU 0 被占的节点照样能用。
判据是**逐卡显存**(1–9 MiB 才是真空),不是 gtop 头行的 idle 计数(它把 held 也算进去),
也不是 squeue 的认领状态(认领 ≠ 在算)。

---

## 1. 本机与作者机器的差别(决定了要改什么)

| | 作者机器 | Isambard-AI |
|---|---|---|
| 架构 | x86_64 | **aarch64 (GH200 Grace-Hopper)** |
| 运行方式 | 单机 docker,root | **Slurm 多用户共享,无 root,Lustre** |
| 路径 | `/mnt/data0/ys/LDM`、`/root/megatron-lm` | 全部要参数化 |
| 环境 | micromamba `slime` in container | conda env on `/home`(VAST,不吃 Lustre inode) |
| CUDA | 12.9 | 驱动 565 = **CUDA 12.7**,同 major 前向兼容 cu128/cu129 |

**Isambard 侧的额外硬约束**(与作者机器无关,但违反会被封号):
- Lustre inode 已用 **97.4%**(49.85M / 51.2M)。**环境一律建在 `/home`(VAST 文件系统)**,
  只有源码和权重放 Lustre。
- 登录节点禁止编译、禁止 GPU。**所有源码编译步骤必须在计算节点上跑。**
- 每次 `sbatch` 前先 `gtop`;名下有空卡就 attach,不排队。

---

## 2. aarch64 逐项落地表(本计划的核心)

交接文档 §7 把 **TE / sglang 在 ARM 上装不上**列为头号风险。逐项查证后,
**大部分件现在都有 aarch64 预编译轮子**,真正要源码编的只剩三个。

| build_conda.sh 里的件 | 钉的版本 | aarch64 现状 | 本机做法 |
|---|---|---|---|
| python | 3.12 | — | 同(小分子栈要求 `>=3.11,<3.13`) |
| torch / torchaudio | `2.11.0+cu129` | ✅ `manylinux_2_28_aarch64` | **原版照用** |
| sglang | `v0.5.15.post1` | ✅ cp310–cp313 aarch64 轮子 | **用 PyPI 轮子**,不走源码 `-e`(源码要 Rust 编 sglang-grpc) |
| sglang-kernel | `0.4.4+cu129` | ✅ `manylinux2014_aarch64` | **原版照用** |
| sgl-deep-gemm | `0.1.4+cu129` | ⚠️ 索引里最低 `0.1.5rc3` | 用 `0.1.5rc3+cu129` aarch64 |
| transformer_engine | `2.16.1` | ❌ 2.16.1 **只有 x86** | **用 `transformer_engine_cu12==2.16.0`**(有 aarch64,差一个补丁号) |
| flash-attn | `2.8.3` 社区轮子 | ❌ 该轮子是 `linux_x86_64` | **源码编**(GH200 有前例,~10–40 min) |
| flash-linear-attention | `0.4.2` | ✅ `py3-none-any` | 原版照用 |
| FlashQLA + tilelang | — | ❌ tilelang nightly **只有 x86** | **跳过**。FlashQLA 是 Qwen3.5 GDN 的*可选*后端,默认 FLA(triton)后端可用;slime 自己的 Dockerfile 在 CUDA13 分支也是这么绕的 |
| apex | git commit | 源码编 | **跳过**。启动脚本已带 `--no-gradient-accumulation-fusion` |
| torch_memory_saver | `8d30c59` | 源码(需 nvcc) | **源码编**。必须 `--no-build-isolation`,否则出的是 46KB 纯 python 轮子,sglang 会报 `Only hook_mode=preload supports pauseable CUDA Graph` |
| sglang_router(slime fork) | `v0.3.2-9daabcd` | ❌ release 只有 x86 | **源码编**(Rust,conda-forge 装 rust) |
| Megatron-LM | `1dcf0dafa884ad52ffb243625717a3471643e087` | 源码 `-e`,编 `helpers_cpp` | 同,必须 `--no-build-isolation` |
| int4_qat kernel | slime 内 | CUDA 编 | 编;失败则看是否为必需 |
| CUDA toolkit | conda `cuda=12.9.1` | conda-forge 有 aarch64 | 同(TE/flash-attn/TMS 编译要 `CUDA_HOME`) |

**结论:要源码编的只有 flash-attn、torch_memory_saver、sglang_router、Megatron 的
helpers_cpp、int4_qat 五处**,全部是 CPU 编译,GH200 每节点 288 核,可高并发。
**TE 这块最大的风险被预编译轮子消掉了。**

### 2.1 cu13 溢出问题
sglang 0.5.15.post1 的元数据要 `flashinfer_python[cu13]` + PyPI 默认 cu13 的 torch,
而驱动 565 只到 CUDA 12.7 —— 装上去 `torch.cuda.is_available()` 直接 False(交接文档 §1)。
作者的解法是**装完再强制回退**:`--force-reinstall --no-deps` 把 torch / sglang-kernel /
sgl-deep-gemm 换成 `+cu129`,再把 `nvidia-*-cu13` 卸掉换 `-cu12`。本机照做。
作者还用 `touch /root/cudart_block/libcudart.so.13` 加 `LD_LIBRARY_PATH` 首位来堵住
系统里的 cudart 13 —— **本机没有系统级 CUDA 13**,这一步先不做,若 TE 的 cudnn-frontend
检查报 cu13 再补。

---

## 3. 两个环境,不是一个

`rl/ldm_rl/remote_env.py` 里 `RemoteLDMEnv` 用 **`task_python` 起子进程**,走 JSON-lines
stdio,并显式剥掉 `LD_LIBRARY_PATH` / `CUDA_HOME` / `CUDA_VISIBLE_DEVICES`。
即:**评测栈本来就跑在另一个 Python 里**,这是代码写死的边界,不是我的选择。

| 环境 | 路径 | 装什么 | 风险 |
|---|---|---|---|
| **评测栈** | `/home/u6gb/kangli.u6gb/envs/ldm-rl` | rdkit / gpytorch / gauche / lightgbm / meeko / gemmi / vina + torch(CPU 用) | 低。全是现成 aarch64 轮子 |
| **训练栈** | `/home/u6gb/kangli.u6gb/envs/ldm-rl-train` | torch2.11+cu129 / TE / sglang / Megatron / slime / ray | 高。见上表 |

分开的另一个理由:训练栈的装法里有大量 `--force-reinstall --no-deps` 和
`pip uninstall nvidia-*`,是个脆弱环境;把定义 reward 的评测栈绑在它上面,
重建一次就连累另一边。

---

## 4. 执行顺序(先做不依赖未解问题的部分)

| 阶段 | 内容 | 依赖 | 状态 |
|---|---|---|---|
| **E0** | 评测栈环境 + vina(aarch64) | 无 | 进行中 |
| **E1** | `ldm_rl` 单元测试(53 个)跑通 | E0 | |
| **E2** | `small_molecule_real_smoke.py` —— **真 vina + 真活性模型 + 真 reward**,纯 CPU | E0 + 8UN5 受体 | |
| **E3** | 训练栈环境(§2 那张表) | 计算节点(要编译) | |
| **E4** | **P0 关卡**:Qwen2.5-1.5B GRPO 冒烟,验 backward 不 SIGSEGV | E2+E3 | |
| **E5** | 模型下载 + HF→Megatron torch_dist 转换(base + SFT) | E3 | |
| **E6** | **P1**:9B real 极小 count 冒烟(count=1, iterations=2),验 hybrid backward / 显存 | E4+E5 | |
| **E7** | 暖共享 GP(`run_warmup_real_slime.sh`,dock 60 个分子) | E2 | |
| **E8** | **P2**:R1–R4 × 3–5 seed 真训练 | 全部 | |
| **E9** | 离线评测 G12C + G12D,填对照表;并行训 `best_g12c_model.joblib` | E8 | |

**E2 是第一个真结果**:它不需要 GPU、不需要 slime,却验证了整条 reward 路径
(docking → 活性 → GP/SIR → acquisition/hypervolume)。这条路径是四个 run 共用的,
坏了什么都别谈。

---

## 5. 已知会踩的坑(记在前面)

1. **启动脚本全是写死路径**(`/mnt/data0/ys/LDM`、`/root/megatron-lm`、
   `/root/micromamba/envs/slime`)。必须参数化后才能在 Slurm 上跑。
2. **`ray start --head --node-ip-address 127.0.0.1`** 在多节点上不成立;单节点 4 卡先按
   127.0.0.1 跑通,再考虑跨节点。
3. **`config_real.json` 里三条路径要改**:`gp_history_file`、`vina_bin`、
   `nn_model_path`、`output_dir`。
4. **`vina_max_workers=32`**:vina 是纯 CPU,GH200 每节点 288 核,可继续加大;
   但 docking 是真瓶颈,**先量一个分子的真实耗时再定并发**。
5. **不要自设 `CUDA_VISIBLE_DEVICES`**——`srun --gres=gpu:N` 已经隔离好卡。
   但注意 `run_train_real_9b.sh` 里 actor 与 rollout 是按 `CUDA_VISIBLE_DEVICES=0,1,2,3`
   在**同一节点内**分卡的(actor 2 卡 + sglang 2 卡),这个是 slime 自己的分配,要保留。
6. **`--gres=gpu:4` 的作业要落在 4/4 全空的节点上**;部分空会在启动窗口之后 SIGABRT。
