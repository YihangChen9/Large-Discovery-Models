# 小分子 RL 交接指南（装什么 · 怎么跑 · 训练计划）

从零把 **小分子 acquisition-RL（GRPO on Slime）** 跑起来。运行矩阵见 `RUNS.md`,实验设计见 `TRAINING_PLAN.md`。
**方向：train KRAS G12D → eval G12C & G12D**（G12D 评测器现成;G12C 只在离线评测用,不进训练循环）。

目标机器:**NVIDIA GH200(aarch64/ARM）Slurm 集群**(驱动 565 / CUDA 12.7,分区 `workq`,单节点 4×GH200 每卡 96GB)。

---

## 0. 硬件前置
- `uname -m` → **aarch64**;`nvidia-smi` → 驱动 **565 / CUDA 12.7**,GPU 96GB。
- 1.5B 冒烟 1–2 卡够;9B 训练一节点 4 卡(TP2/TP4)。
- 磁盘 ~150G(模型 + Megatron torch_dist + 中间产物)。
- 能联网到 HuggingFace(下模型)、RCSB(受体,可选)。

## 1. ⚠️ CUDA 红线（最重要,先记住）
驱动 565 = CUDA 12.7 → **所有 torch/CUDA 轮子必须是 `+cu128` 或 `+cu129`**。
**`+cu130` 装上去 `torch.cuda.is_available()` 直接 False**(PyPI 默认 torch 已切 CUDA 13,装前先看 `+cuXXX` 后缀)。

## 2. 装环境（建一个 conda/venv,装两组依赖）

```bash
conda create -n ldm-rl python=3.11 -y && conda activate ldm-rl
```

### 2A. 训练栈（Slime RL）
```bash
# torch(aarch64, cu128) —— 别用 cu130
pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch;print(torch.__version__, torch.cuda.is_available())"   # 必须 True

# Transformer Engine(ABI 最敏感的一环;源码装,需 CUDA toolkit 头文件)
pip install transformer_engine[pytorch]           # 装完必须能 import,且训练 backward 不 SIGSEGV(见 §6 P0 先验)

# Megatron-LM: checkout slime 期望的 commit,放 PYTHONPATH
git submodule update --init rl/slime rl/megatron-lm
(cd rl/megatron-lm && git checkout 1dcf0dafa884ad52ffb243625717a3471643e087)

# Slime(RL 编排)+ 依赖
pip install -e rl/slime --no-deps
pip install ray                                    # slime 用 ray 起 actor/rollout

# sglang(rollout 推理引擎;装 aarch64 可用版本)
pip install "sglang[all]"

# APEX(可选;脚本用 --no-gradient-accumulation-fusion 时可不装)
```
> 若 TE / sglang 在 ARM 上源码装困难:可用 **NGC 的 aarch64 PyTorch 容器**(apptainer)当底座,里面 torch+TE 已配平——但你要求纯装环境,这里给 pip 路径,TE 那步失败就退容器。

### 2B. 小分子评测栈（docking + GP + activity，real reward 必需）
来自 `tasks/small_molecule/pyproject.toml`:
```bash
pip install gpytorch gauche lightgbm scikit-learn joblib \
            rdkit meeko gemmi numpy scipy pandas omegaconf pydantic python-dotenv pyyaml openai
# vina 二进制(aarch64):
conda install -c conda-forge vina        # 或源码编 AutoDock Vina;记住它的路径,填进 config_real.json 的 vina_bin
```

### 2C. LDM 仓库本体
```bash
git clone --recurse-submodules -b rl https://github.com/YihangChen9/Large-Discovery-Models.git LDM
cd LDM
export PYTHONPATH=$PWD/rl/megatron-lm:$PWD/rl:$PWD:$PYTHONPATH
```

## 3. 资产（放到脚本里写死的路径,或改脚本）

| 资产 | 路径 | 来源 |
|---|---|---|
| base Qwen3.5-9B | `.../hf_models/models/Qwen3.5-9B` | HF `Qwen/Qwen3.5-9B` |
| SFT no-GP 模型 | `.../hf_models/models/LDM-CoT-SFT` | HF `Yangtze-ailab/LDM-CoT-SFT-Qwen3.5-9B-MixedScience` |
| 冒烟 Qwen2.5-1.5B | `.../hf_models/models/Qwen2.5-1.5B-Instruct` | HF |
| vina 二进制(aarch64) | 见 §2B,填进 `config_real.json:vina_bin` | conda-forge / 源码 |
| G12D 活性模型 | `tasks/small_molecule/resources/models/best_g12d_model.joblib` | **已在 repo** |
| 8UN5 受体 | `docking_work/receptors/…` | 随包 / meeko 从 RCSB 8UN5 制备 |

```bash
pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen3.5-9B --local-dir .../hf_models/models/Qwen3.5-9B
hf download Yangtze-ailab/LDM-CoT-SFT-Qwen3.5-9B-MixedScience --local-dir .../hf_models/models/LDM-CoT-SFT
```

## 4. 准备（一次）
```bash
cd rl/slime_launch
# 转 Megatron torch_dist(base + SFT 各一次)
MODEL_HF=.../Qwen3.5-9B  SAVE=.../rl/qwen3.5-9B_torch_dist      bash convert_9b.sh
MODEL_HF=.../LDM-CoT-SFT SAVE=.../rl/qwen3.5-9B-sft_torch_dist  bash convert_9b.sh
# 生成 4 个 run 的 episodes(reward 已烤进数据)
bash gen_episodes_runs.sh
# 暖共享 GP(rollout-only,dock warmup.num_samples 个分子)
bash run_warmup_real_slime.sh
```

## 5. 怎么跑（4 个 run;命令全在 `RUNS.md`）
一节点 4×GH200 → TP2/TP4。**注意脚本里的 `CUDA_VISIBLE_DEVICES` / `--rollout-num-gpus` / `--actor-num-gpus-per-node` 按节点 4 卡调**。Slurm 提交:
```bash
#!/bin/bash
#SBATCH -p workq -N 1 --gres=gpu:4 -t 24:00:00 -J ldm-rl-R2
conda activate ldm-rl
export PYTHONPATH=$LDM/rl/megatron-lm:$LDM/rl:$LDM
srun bash $LDM/rl/slime_launch/run_train_real_9b.sh
```
每个 run 建议 ≥3–5 seed:**换环境 seed 要用 `python -m ldm_rl.episodes --seed-offset N` 重新生成 episodes**(`--seed-offset` 是 `episodes.py` 的参数,**不是** slime/训练启动器的;slime 的 `--seed`/`--rollout-seed` 只管框架侧随机性,改了不换环境轨迹)。开 wandb: `export WANDB_KEY=<key>`。

## 6. 训练计划（分阶段,带验证闸;实验组见 TRAINING_PLAN.md）
1. **P0**:1.5B 跑通 GRPO,验 **backward(ARM/TE 是否 SIGSEGV)+ docking + reward** 全链路。**这是最先做的闸——TE 装对没对,这一步见分晓。**
2. **P1**:9B **real 极小 count**(如 count=1、iterations=2,只 dock 极少分子)冒烟,验 hybrid backward / 显存。**训练全程 real,不用 mock。**
3. **P2**:R1–R4 × seed 真训练(train G12D)。
4. **评测**:离线在 C & D 上评,填 TRAINING_PLAN 里的对照表。
   - 前置(可并行):`tasks/small_molecule/core/activity_modeling/train_g12c_qsar.py` + `g12c_docking_benchmark.csv` 训出 `best_g12c_model.joblib`(受体沿用 8UN5)。

## 7. 已知坑
- **CUDA 红线**:torch 只用 `+cu128/+cu129`,cu130 → `cuda.is_available()=False`(§1)。
- **TE / backward**:aarch64 上 TE 是 ABI 最敏感的一环;装完先按 §6 P0 用 1.5B 冒烟确认 backward 不 SIGSEGV,再上 9B。
- **9B 是 hybrid**(线性注意力+MTP):转换/训练用 `qwen3.5-9B.sh` 的 spec(脚本已 source);先用 real 极小 count 冒烟。
- **显存**:9B + TP2/4 + sglang 于 96GB/卡,宽裕;OOM 就升 TP 或降 `max_tokens_per_gpu`(recompute-full 已开)。
- **docking 吞吐**:`config_real.json` 已把 `vina_max_workers=32`(vina 纯 CPU;GH200 每节点 288 ARM 核,可并行大批 docking,按核数继续调大);配合 canonical-SMILES 缓存进一步降开销。vina 二进制记得用 **aarch64** 版(§2B)。
- **代码架构无关**:`ldm_rl` + reward 是纯 Python,已验证(zsgpu 上 53 测试通过);移植只在底层运行时。
