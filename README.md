# 狼人杀多智能体系统（Werewolf Multi-Agent Arena）

这是一个基于 `ChatArena` 扩展而来的狼人杀多智能体项目。与最初的通用多智能体聊天环境不同，本项目当前重点已经转向：

- 狼人杀规则下的角色推理
- 基于共情建模的对手分析
- 以 LLM 为主决策、MCTS 为辅助参考的混合推理
- 基于“证据 → 含义 → 策略 → 发言”的思维链生成
- 可追踪、可分析、可回放的结构化日志系统

本项目的目标不是让模型“随便说话”，而是让不同角色在狼人杀博弈中表现出更强的认知、推理和策略能力。

---

## 1. 项目目标

狼人杀不是普通问答任务，而是一个包含：

- 隐藏身份
- 信息不对称
- 博弈欺骗
- 发言伪装
- 投票施压
- 动态更新信念

的社会推理游戏。

因此，本项目试图让 LLM Agent 不只是根据 prompt 输出一句话，而是能够：

1. 先理解局势
2. 再提炼证据
3. 再形成因果判断
4. 再决定策略
5. 最后生成自然语言发言或夜晚行动

---

## 2. 项目结构概览

### 根目录中的关键文件

- `run_werewolf.py`
  - 项目主入口。
  - 负责加载配置、创建竞技场、初始化玩家、启动游戏循环、写入调试日志。

- `README.md`
  - 当前中文说明文档。

- `PROJECT_OVERVIEW.md`
  - 更偏开发者视角的架构说明文档。

- `examples/werewolf.json`
  - 游戏配置示例。
  - 定义全局 prompt 和玩家 backend 配置。

- `config/1.json`
  - 角色数和角色分配相关配置。

- `requirements.txt` / `pyproject.toml` / `setup.py`
  - 依赖与打包配置。

- `model_reply*.log`、`debug_output.txt`
  - 运行日志与调试日志。

- `experiment_results/`
  - 历史实验结果与分析产物。

### 核心源码目录 `chatarena/`

- `chatarena/environments/werewolf.py`
  - 狼人杀环境与规则引擎。
- `chatarena/backends/openai.py`
  - Agent 行为生成主链路。
- `chatarena/MCTS.py`
  - MCTS 搜索与辅助决策。
- `chatarena/empathy_field.py`
  - 共情场与关系建模。
- `chatarena/empathy_module.py`
  - 共情信息抽取与转换。
- `chatarena/belief_state.py`
  - 信念状态跟踪。
- `chatarena/message.py`
  - 消息结构定义。
- `chatarena/arena.py`
  - 游戏调度器。
- `chatarena/agent.py`
  - Agent 抽象。
- `chatarena/utils.py`
  - 通用工具函数。
- `chatarena/config.py`
  - Arena 配置结构。
- `chatarena/database.py`
  - 经验/记忆/问答类数据辅助。
- `chatarena/backends/`
  - 各类模型后端实现，如 `openai`、`gemini`、`qwen`、`hf_transformers` 等。

---

## 3. 各模块职责说明

### 3.1 `run_werewolf.py`

这是整个项目的启动脚本，主要负责：

1. 解析命令行参数
2. 加载环境配置与角色配置
3. 设置日志输出
4. 构建玩家配置
5. 创建 `Arena`
6. 启动游戏循环
7. 保存日志与实验信息

它是狼人杀游戏的总入口。

---

### 3.2 `chatarena/environments/werewolf.py`

这是狼人杀规则执行的核心文件，主要负责：

- 夜晚阶段顺序控制
- 白天讨论阶段顺序控制
- 投票阶段控制
- 狼人、守卫、女巫、预言家的夜晚动作处理
- 发言/投票内容解析
- 游戏结束判定
- moderator 提示语输出

一句话概括：

> 它负责把狼人杀规则真正“跑起来”。

---

### 3.3 `chatarena/backends/openai.py`

这是当前最重要的后端文件，负责将“任务”变成“模型回复”。

它承担的功能包括：

- 调用 LLM API
- 组装 prompt
- 加载角色信息和局势信息
- 引入共情建模结果
- 引入 MCTS 辅助参考
- 构造证据链
- 生成夜晚行动或白天发言
- 清洗、校正和规范化输出
- 写入结构化日志

如果你想理解这个项目的智能逻辑，应该重点看这个文件。

---

### 3.4 `chatarena/MCTS.py`

这是 MCTS 搜索模块，用于在局势不确定时提供辅助搜索结果。

它主要做：

- 构建搜索树
- 模拟候选动作
- 对动作进行评分
- 结合局势信息给出参考建议

需要特别强调的是：

> 在当前设计中，MCTS 不是最终决策者，只是辅助参考。

最终决策仍然以 LLM 为主。

---

### 3.5 `chatarena/empathy_field.py`

这是共情场建模模块，负责把游戏中的关系、倾向和对抗结构转成更适合推理的表示。

它关注的问题包括：

- 谁支持谁
- 谁在压谁
- 谁在试探谁
- 谁在伪装
- 谁的发言可能是策略性误导
- 哪些关系链值得重点关注

它的作用不是“情绪表达”，而是**把社交博弈结构显式化**。

---

### 3.6 `chatarena/empathy_module.py`

共情抽取与转换模块，负责从对话、发言、投票和夜晚动作中提取结构化的共情信号。

---

### 3.7 `chatarena/belief_state.py`

信念状态模块，负责追踪 agent 对场上各玩家的信任/怀疑状态。

它用于支持：

- 后验判断更新
- 玩家风险评估
- 随时间变化的立场维护
- 多轮推理的一致性

---

## 4. 狼人杀整体调用流程

下面按一次完整游戏说明调用流程。

### 第 1 步：程序启动

入口是 `run_werewolf.py`。

它会：

- 读取参数
- 读取环境配置
- 读取角色配置
- 设置日志
- 构建 Arena

---

### 第 2 步：初始化竞技场

`Arena.from_config(...)` 会创建：

- 游戏环境
- moderator
- 玩家 backend
- 玩家列表
- 角色信息

---

### 第 3 步：夜晚阶段

环境按角色顺序发起夜晚任务：

1. 狼人选择击杀目标
2. 守卫选择保护目标
3. 女巫决定是否救人
4. 女巫决定是否毒人
5. 预言家选择验人目标

在这一阶段，`openai.py` 会：

- 收集当前局势
- 读取共情建模
- 必要时进行 MCTS 辅助搜索
- 构建角色行为 guidance
- 调用 LLM 生成动作
- 校验输出合法性
- 返回给环境执行

---

### 第 4 步：白天讨论阶段

白天会轮流发言。

每个玩家在发言前，后端会整合：

- 公共历史发言
- 夜晚结果
- 角色私有信息
- 共情数据
- 反思上下文
- 证据链

然后生成自然语言发言。

这里的重点不是“说得像”，而是：

- 能否真正分析局势
- 能否识别欺骗和矛盾
- 能否推动投票和阵营博弈

---

### 第 5 步：投票阶段

白天发言后，环境进入投票。

后端生成投票形式的回复，环境解析目标并汇总票数，最终决定淘汰玩家。

---

### 第 6 步：日志与回放

本项目非常重视日志。

每次回答都会记录：

- 原始回复
- 结构化日志
- 共情摘要
- 共情详情
- 反思上下文
- 因果链日志

这些日志用于检查：

- 共情是否有效
- 思维链是否成立
- 发言是否真的建立在证据之上
- MCTS 是否只是辅助而不是主导

---

## 5. 项目的核心创新点

本项目的核心创新可以概括为三层：

### 5.1 共情建模

共情建模不是单纯的“情绪识别”，而是对狼人杀社会博弈中的关系进行结构化建模。

它会帮助 agent 理解：

- 谁在支持谁
- 谁在攻击谁
- 谁在试探谁
- 谁在故意模糊信息
- 谁在制造伪线索
- 哪些关系更像是战略行为而不是自然反应

共情输出通常包括：

- `hard_wolf_prob`
- `soft_wolf_prob`
- `public_trust`
- `information_gain`
- `vote_pressure`
- `speech_vote_consistency`
- `semantic_memory`
- `uncertainty_notes`
- `signal`
- `supports / supported_by`
- `clears / cleared_by`
- `opposes / opposed_by`

---

### 5.2 MCTS 辅助搜索

MCTS 用于在不确定局势中提供搜索参考。

但它不是主决策者。

当前原则是：

- LLM 负责主要推理和决策
- MCTS 提供候选方向和弱参考
- 不能让 MCTS 把最终发言“带偏”

---

### 5.3 证据链驱动的发言

当前系统非常强调：

> evidence → meaning → strategy → speech

也就是说，发言不是直接从模板生成，而是先在内部完成推理流程：

1. **Evidence**：从共情、历史和局势中提取证据
2. **Meaning**：解释这些证据意味着什么
3. **Strategy**：决定这句话要达成什么博弈目标
4. **Speech**：最终转化为自然语言输出

这是本项目最重要的认知增强方向。

---

## 6. 当前后端的思维链结构

在 `chatarena/backends/openai.py` 中，智能链路大体是：

### 输入
- 游戏历史
- 当前任务
- 角色信息
- 存活玩家
- 共情数据
- 反思结果
- MCTS 参考

### 中间推理
- 提取证据
- 构建因果链
- 组织角色策略
- 生成反思摘要

### 输出
- 夜晚动作文本
- 白天讨论发言
- 投票文本

### 保障
- 格式清洗
- 合法性修正
- 身份修正
- 日志记录

---

## 7. 日志系统说明

项目会记录多层日志，方便排查模型思维是否有效。

### 主要日志文件

- `debug_output.txt`
- `model_reply.log`

### 结构化日志中常见字段

- `EMPATHY_SUMMARY`
- `EMPATHY_DETAIL`
- `REFLECTION_CONTEXT`
- `CAUSAL_CHAIN`
- `STAGE_DEBUG`
- `CONTENT`

### 这些日志能帮助你检查什么

- 共情数据是否真的被用上了
- 反思是否产生了可执行判断
- 因果链是否合理
- 最终发言是否和思维链一致
- MCTS 是否只是辅助而非主导

---

## 8. 代码文件职责速查表

| 文件 | 作用 |
|---|---|
| `run_werewolf.py` | 主入口，启动游戏 |
| `chatarena/environments/werewolf.py` | 狼人杀环境与规则 |
| `chatarena/backends/openai.py` | LLM 调用、共情、MCTS、证据链、发言生成 |
| `chatarena/MCTS.py` | 搜索与辅助决策 |
| `chatarena/empathy_field.py` | 共情场与关系建模 |
| `chatarena/empathy_module.py` | 共情抽取与转换 |
| `chatarena/belief_state.py` | 信念状态管理 |
| `chatarena/message.py` | 消息结构 |
| `chatarena/arena.py` | 游戏调度 |
| `experiment_results/` | 实验结果归档 |
| `model_reply.log` | 模型回复与结构化日志 |
| `debug_output.txt` | 全量调试输出 |

---

## 9. 与原始 ChatArena 版本的区别

原始仓库中的 README 更像是通用 ChatArena 说明，强调的是基础框架和论文复现。

而当前版本已经变成了一个更具体的狼人杀智能体系统，重点在于：

- 狼人杀规则执行
- 多角色行为建模
- 共情关系分析
- LLM 主导推理
- MCTS 弱辅助
- 证据链发言
- 可追踪日志分析

所以现在的 README 应该反映“增强后的狼人杀系统”，而不是只介绍原始 ChatArena。

---

## 10. 建议的阅读顺序

如果你想快速理解代码，建议按下面顺序阅读：

1. `run_werewolf.py`
2. `chatarena/environments/werewolf.py`
3. `chatarena/backends/openai.py`
4. `chatarena/empathy_field.py`
5. `chatarena/MCTS.py`
6. `chatarena/belief_state.py`
7. `chatarena/message.py`

---

## 11. 后续优化方向

如果继续改进系统，最值得做的方向是：

- 提高共情数据的可解释性
- 进一步强化证据链日志
- 继续降低 MCTS 的主导性
- 增强预言家、守卫、女巫、狼人四类角色的推理差异
- 让白天发言更具博弈压强，而不只是解释局势

---

## 12. 总结

当前项目可以概括为：

> 一个以狼人杀为任务载体、以共情建模为中间层、以 LLM 为主决策、以 MCTS 为辅助搜索、以证据链驱动发言的多智能体推理系统。

它的关键不只是“能玩狼人杀”，而是能让 agent 的思考过程更可解释、更可追踪，也更接近真实的社交博弈推理。

---

## 13. 许可

本项目遵循原仓库的许可证。详情请查看 `LICENSE`。
