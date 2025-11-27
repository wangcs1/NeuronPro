## 一、到底要做什么？

**项目标题：**

> Poisson Spike Train + Tuning Curve 的视觉方向群体编码模型，及基于 SNN 的解码性能研究

**核心目标：**

1. 构建一个**视觉方向编码模型**：

   * 若干个“V1/MT样的”方向选择性神经元
   * 每个神经元用 **方向 tuning curve + Poisson spike train** 描述
   * 可以显式控制：tuning 宽度、最大 firing rate、baseline、噪声类型（Poisson / 额外噪声 / 相关噪声）

2. 在这个编码模型之上，研究不同解码方法对方向信息的恢复能力：

   * 线性解码（LogReg / Linear readout）
   * Rate-based MLP 解码
   * **SNN 解码**（Spiking decoder，利用时间信息）

3. 系统性地回答几个**明确的研究问题**（这个就是项目的“灵魂”）：

   * RQ1：在 Poisson + tuning curve 的方向编码下，**单纯 firing rate（rate code）是否已经足够解码方向？**
   * RQ2：在有时间结构的刺激（比如方向突变、短时刺激）下，**SNN 对 spike timing 的利用是否带来明显优势？**
   * RQ3：**调节 tuning 宽度、神经元数、时间窗口和噪声强度** 时，解码性能如何变化？（这就是你可以画很多漂亮曲线/图的地方）

---

## 二、为了变成一个**像样的研究项目**，我们做几件事：

### 1）明确模型结构：编码层 + 解码层

* 编码层：

  * 一群神经元 ({i=1,...,N})，每个有 preferred direction ($\theta_{\text{pref},i}$)
  * Tuning：
    [
    $$f_i(\theta) = r_0 + r_{\max,i}\exp\left(-\frac{\Delta\theta_i^2}{2\sigma^2}\right)$$
    ]
  * Spike：给定 ($f_i(\theta$))，用 Poisson 生成 spike train ($s_i(t)$)

* 解码层：

  * 输入：($\mathbf{S}(t) \in {0,1}^{T \times N}$) （spike train）
  * 输出：方向类别 ($\hat{\theta}$)
  * 模型包括：线性、MLP、SNN（LIF）

### 2）把“分析”写进设计里

除了 accuracy，还要**显式分析**：

* PSTH / tuning curve（验证编码合理）
* Fano factor（验证 Poisson 及额外噪声特性）
* Accuracy vs 时间窗口长度 ($T_{\text{obs}}$)
* Accuracy vs neuron 数 (N)
* Accuracy vs tuning 宽度 ($\sigma$)
* （可选）和理想 Bayesian decoder 对比（Poisson likelihood 最大化）

这样写报告的时候，结构可以是：

> 2.x Encoding properties（tuning, variability）
> 3.x Decoding results（不同 decoder、不同参数）
> 4.x 讨论：spike timing 的作用、population coding 的效率等

---

## 三、项目工程结构


```text
project_root/
  ├─ config.py                # 存放统一参数与 experiment config
  ├─ encoding/
  │    ├─ tuning.py           # 生成 tuning curve
  │    ├─ poisson_spike.py    # 生成 Poisson spike train
  │    └─ dataset_builder.py  # 调用上面两个，生成 (spikes, labels)
  ├─ decoding/
  │    ├─ rate_decoder_mlp.py # rate-based baseline
  │    ├─ snn_decoder.py      # SNN 解码器
  │    └─ linear_decoder.py   # Logistic / Linear
  ├─ experiments/
  │    ├─ exp_time_window.py  # T_window sweep
  │    ├─ exp_population.py   # N_neurons sweep
  │    └─ exp_tuning_sigma.py # tuning 宽度 sweep
  ├─ analysis/
  │    ├─ psth_fano.py        # PSTH + Fano factor 分析
  │    └─ visualization.py    # 画混淆矩阵、曲线等
  └─ main.ipynb               # 你自己的探索 Notebook
```


---

## 四、具体实验设计

### 实验 1：**静态方向 + 长时间窗：rate vs SNN**

* 设置：

  * 方向数：8
  * (T = 800,$\text{ms}$)
  * 中等 neuron 数：(N = 40)
  * 中等噪声（gain + shared + indep）

* 解码器：

  * Linear / MLP：输入 spike count
  * SNN：输入全 spike train

* 预期：

  * 静态方向 + 长时间窗本身是典型 rate code 友好场景 → MLP ≈ SNN
  * 这说明在这种简单情形下“时间信息不会给太多额外收益”，是一个对 SNN 的 sanity check。

---

### 实验 2：**短时间窗 / 方向快速变化：SNN 的优势场景**

设计一个“时间相关”的刺激，例如：

* 前 (T_1 = 100,$\text{ms}$) 是方向 ($\theta_1$)，后 (T_2 = 100,$\text{ms}$) 是方向 ($\theta_2$)
* 标签可以是「后半段方向」或「方向是否发生变化」之类

然后：

* rate-baseline：只能看到两个方向的混合 spike count → 信息被“平均”
* SNN：可以按时间解析，**知道前后两段 spike pattern 不同**

理论上来讲，这里就可以展示：

> SNN 对 spike timing 具有优势，能在时间结构存在的时序任务上做得更好。

---

### 实验 3：**population size / tuning width / 噪声对解码性能的影响**

做 sweep：

* ($N \in {10, 20, 40, 80}$)
* ($\sigma \in {30^\circ, 45^\circ, 60^\circ}$)
* gain_sigma / shared_std 取 2–3 个档位

maybe可以画：

* Accuracy vs N（population coding 效率曲线）
* Accuracy vs (\sigma)（tuning 过窄 / 过宽 会导致类间 pattern 太相似 / 太局部）
* Accuracy vs 噪声强度（噪声越大，性能下降）


