提供的一些方法：

### Reasoning Module：
按顺序处理subtask，对每个stages提供solution
1. ReasoningIO（直接输入输出）
- 最简单的推理方式
- 直接将任务发送给LLM并返回结果
2. ReasoningCOT（思维链）
- Chain of Thought推理
- 要求模型逐步解决问题
3. ReasoningCOTSC（自洽思维链）
- 生成多个(5个)答案
- 使用Counter统计最常见的答案
- 选择出现频率最高的作为最终结果
4. ReasoningTOT（思维树）
- Tree of Thoughts推理
- 生成多个思路(3个)
- 使用投票机制选择最佳路径
5. ReasoningDILU
- 模拟真实用户行为
- 使用系统提示来设定角色
6. ReasoningSelfRefine（自我改进）
- 先生成初始推理
- 然后对结果进行反思和改进
7. ReasoningStepBack（后退策略）
- 先理解任务相关的常识
- 基于更宏观的理解来解决问题

### Memory Module:
1. MemoryDILU
- 最简单的记忆检索
- 找到最相似的一条记忆并返回
- 直接存储当前情况
2. MemoryGenerative（生成式记忆）
- 检索最相关的3条记忆
- 使用 LLM 为每条记忆评分(1-10)
- 返回评分最高的记忆
- 考虑记忆与当前任务的相关性
3. MemoryTP（任务规划记忆）
- 基于相似经验生成新的行动计划
- 不直接使用旧记忆，而是生成新的策略
- 特别关注具体行动步骤
4. MemoryVoyager（航行者记忆）
- 存储记忆时会生成简短摘要
- 限制摘要在6句话以内
- 返回最相似记忆的完整轨迹

### Planning Module:
1. PlanningIO（基础输入输出）
- 最基本的任务分解
- 关注子任务的推理和工具调用指令
2. PlanningDEPS（依赖关系规划）
- 强调多跳问题的子目标序列
- 关注动作序列的生成
3. PlanningTD（时序依赖规划）
- 特别强调时序依赖关系
- 确保子任务的逻辑顺序
- 明确指定依赖关系
4. PlanningVoyager（航行者规划）
- 强调完整的子目标分解
- 要求按顺序完成子目标
- 提供详细的推理和工具调用指令
5. PlanningOPENAGI（简洁规划）
- 强调任务列表简洁性
- 每个任务用单句描述
- 确保任务相关性和有效性
6. PlanningHUGGINGGPT（依赖顺序规划）
- 逐步思考所需任务
- 最小化任务数量
- 强调任务间的依赖和顺序