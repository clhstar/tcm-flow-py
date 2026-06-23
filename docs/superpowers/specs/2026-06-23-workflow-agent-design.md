# TCM-Flow Workflow Agent 设计

> 状态：设计已确认，等待书面复核
> 日期：2026-06-23
> 方案：B，新增 `workflow_agent`，保留现有 `lead_agent`
> 定位：固定中医问答多 Agent 主流程，动态子 Agent 作为辅助手段

## 1. 目标

新增一个独立的 `workflow_agent`，把当前主要依赖 `lead_agent` 大 prompt 自主决策的中医问答流程，拆成固定、可追踪、可测试的业务 Agent 流水线：

```text
用户问题
  -> InquiryAgent 问诊信息整理
  -> EvidenceAgent 证据检索
  -> SyndromeAgent 候选辨证方向分析
  -> AnswerAgent 回答生成
  -> SafetyAgent 安全审查
  -> 最终回答
```

本阶段要让论文和工程实现都能清楚说明：

```text
1. 问诊信息是否足够由 InquiryAgent 先判断；
2. 证据检索只归 EvidenceAgent 所有；
3. 候选辨证只输出“可能相关因素”，不输出诊断；
4. 最终回答之后必须经过独立 SafetyAgent 审查；
5. 动态子 Agent 只作为辅助手段，不替代固定主流程。
```

## 2. 非目标

- 不删除或替换现有 `lead_agent`；
- 不改变默认请求体中的 `assistant_id="lead_agent"` 行为；
- 不改前端 SSE 协议、thread/run 状态机和 `final` payload 的基本结构；
- 不新增方剂、药物、剂量、煎服法推荐能力；
- 不让除 EvidenceAgent 外的 Agent 直接调用检索工具；
- 不把动态 `task` 子 Agent 作为主流程的一部分强制执行。

## 3. 总体架构

新增目录：

```text
app/agents/workflow_agent/
  __init__.py
  agent.py
  models.py
  prompts.py
  workflow.py
```

职责划分：

| 文件 | 职责 |
| --- | --- |
| `models.py` | 定义固定 Agent 间传递的 Pydantic 结构 |
| `prompts.py` | 固定 Agent 的 system/user prompt 构造 |
| `workflow.py` | 编排 Inquiry/Evidence/Syndrome/Answer/Safety 的执行流程 |
| `agent.py` | 暴露与 `run_agent` 兼容的 agent factory |

注册新增 assistant：

```text
assistant_id="workflow_agent" -> make_workflow_agent
assistant_id="lead_agent"     -> make_lead_agent
```

旧 `lead_agent` 继续用于对照、回归和线上回滚。新路径通过调用方显式传入 `workflow_agent` 启用。

## 4. 运行时兼容策略

现有 `run_agent` 负责：

```text
1. 创建 run/thread 状态；
2. 发布 SSE metadata/messages/values/updates/final；
3. 检测澄清中断；
4. 应用现有 guardrail middleware；
5. 保存 conversation、last_validation、last_agent_trace。
```

`workflow_agent` 应尽量实现与 LangGraph agent 相同的最小运行时接口：

```text
astream({"messages": messages}, config=config, stream_mode=modes)
aget_state(config)
aupdate_state(config, values)
```

首版可以只保证当前 `run_agent` 实际依赖的行为：

```text
1. `astream` 产出 `values` 快照，包含 `messages`；
2. 如需要兼容前端 token 流，可额外产出 `messages` 事件；
3. `aget_state` 返回包含 `values["messages"]` 的快照；
4. `aupdate_state` 支持 guardrail 重写最终 AIMessage。
```

这样可以保留 SSE、conversation 和现有后处理链路，避免把 workflow 变成第二套运行时。

## 5. 固定 Agent 合同

### 5.1 InquiryAgent

InquiryAgent 只负责整理输入，不生成最终回答。

输入：

```text
用户本轮问题
当前 thread 可见历史
可选的上一轮澄清问题和用户补充
```

输出：

```json
{
  "chief_complaint": "胃胀",
  "known_facts": {
    "duration": "",
    "triggers": ["油腻后加重"],
    "associated_symptoms": ["嗳气"],
    "risk_flags": []
  },
  "missing_info": ["大便情况", "食欲", "是否反酸烧心"],
  "information_sufficiency": "insufficient",
  "clarification_questions": [
    "胃胀持续多久了？",
    "大便情况如何？",
    "是否伴有反酸、烧心或腹痛？"
  ],
  "should_pause_for_clarification": true
}
```

规则：

```text
1. 每次最多追问 3 个关键问题；
2. clarification_questions 只写问题正文，不带序号；
3. 信息严重不足时，暂停本轮，不进入检索；
4. 明显危险信号可记录到 risk_flags，但最终就医提醒由 SafetyAgent 统一把关。
```

### 5.2 EvidenceAgent

EvidenceAgent 是唯一能调用 `retrieve_tcm_knowledge` 的固定 Agent。

输入：

```text
用户问题
InquiryAgent 结构化问诊状态
```

输出：

```json
{
  "retrieval_status": "ok",
  "retrieval_mode": "hybrid_parent",
  "degraded": false,
  "evidence": [
    {
      "id": "E1",
      "citation_id": "E1",
      "role": "syndrome_pattern",
      "text": "检索证据摘要或原文",
      "source": "《景岳全书》 卷 / 章 / 节"
    }
  ],
  "allowed_terms": ["食滞", "脾胃运化不畅"],
  "raw_tool_content": "retrieve_tcm_knowledge 返回文本"
}
```

规则：

```text
1. 默认使用 mode="hybrid"；
2. 最多整理 E1-E5；
3. 保留 raw_tool_content，供 guardrail、trace 和调试使用；
4. 不输出诊断、不输出方药剂量；
5. 检索不足时明确标记 retrieval_status="insufficient_evidence"。
```

### 5.3 SyndromeAgent

SyndromeAgent 只做“可能相关因素”分析，不做诊断。

输入：

```text
InquiryAgent 问诊状态
EvidenceAgent E1-E5 证据
EvidenceAgent allowed_terms
```

输出：

```json
{
  "possible_patterns": [
    {
      "term": "食滞",
      "supporting_evidence": ["E1", "E2"],
      "confidence": "medium",
      "reason": "用户提到进食油腻后胃胀加重，并伴有嗳气。"
    }
  ],
  "not_enough_for_diagnosis": true,
  "need_more_info": ["舌象", "大便", "食欲", "疼痛性质"]
}
```

规则：

```text
1. 不输出“你这是某某证”；
2. 只能使用 EvidenceAgent allowed_terms 中的术语；
3. supporting_evidence 必须引用 E1-E5；
4. 依据不足时保留空 possible_patterns，并说明 need_more_info。
```

### 5.4 AnswerAgent

AnswerAgent 只负责组织最终语言，不做新推理。

输入：

```text
用户问题
InquiryAgent 结果
EvidenceAgent 结果
SyndromeAgent 结果
SafetyAgent 审查反馈，重写时提供
```

输出：

```json
{
  "draft_answer": "从你描述的胃胀、嗳气、油腻后加重来看..."
}
```

规则：

```text
1. 不新增术语；
2. 不新增证据；
3. 不新增诊断；
4. 不新增方药和剂量；
5. 涉及古籍证据时必须标注 E1-E5；
6. 信息不足时用谨慎表达，并提示可补充的信息。
```

### 5.5 SafetyAgent

SafetyAgent 必须独立执行，并放在 AnswerAgent 初稿之后。

输入：

```text
AnswerAgent 初稿
InquiryAgent risk_flags
EvidenceAgent 证据和 allowed_terms
SyndromeAgent 候选分析
```

输出：

```json
{
  "has_risk_flags": false,
  "risk_flags": [],
  "contains_diagnosis": false,
  "contains_prescription": false,
  "contains_dosage": false,
  "needs_offline_medical_advice": false,
  "final_safety_level": "low",
  "rewrite_required": false,
  "rewrite_instructions": []
}
```

规则：

```text
1. 检查危险症状是否需要线下就医提醒；
2. 检查是否出现直接诊断；
3. 检查是否出现处方、药物推荐、剂量、煎服法；
4. 检查是否声称替代医生面诊；
5. 如果 rewrite_required=true，AnswerAgent 必须按 rewrite_instructions 重写一次；
6. 重写后仍需再次 SafetyAgent 审查，最多重写 1 次；仍不安全则返回保守安全答复。
```

危险信号首版覆盖：

```text
胸痛、呼吸困难、意识异常、剧烈头痛、持续高热、肢体无力、
持续加重的腹痛、反复呕吐、黑便、明显出血。
```

## 6. 动态子 Agent 辅助边界

现有 `task` 动态子 Agent 保留，但不作为首版固定 workflow 的必经节点。

允许用途：

```text
1. 用户明确要求“从多个角度整理”时，辅助生成非最终的结构化整理；
2. 复杂输入需要拆出摘要时，作为 InquiryAgent 的辅助摘要来源；
3. EvidenceAgent 已完成检索后，可让动态子 Agent 基于已检索证据做局部整理。
```

禁止用途：

```text
1. 动态子 Agent 不得直接调用 retrieve_tcm_knowledge；
2. 动态子 Agent 不得绕过 SafetyAgent 生成最终回答；
3. 动态子 Agent 不得输出诊断、处方、剂量；
4. 动态子 Agent 结果只能进入 agent_trace 或固定 Agent 输入，不能直接展示给用户。
```

## 7. 消息与 Trace

`workflow_agent` 需要在 messages 中保留足够的工具样式记录，便于复用现有 trace 和 guardrail：

```text
1. InquiryAgent 需要澄清时，生成等价 ask_clarification 的 tool message；
2. EvidenceAgent 检索后，生成 name="retrieve_tcm_knowledge" 的 tool message；
3. 最终 AnswerAgent 输出一条无 tool_calls 的 AIMessage；
4. workflow 内部结构化结果写入 agent_trace。
```

`agent_trace` 建议结构：

```json
[
  {
    "agent": "InquiryAgent",
    "action": "assess_information",
    "status": "needs_clarification",
    "summary": "缺少持续时间、大便、反酸烧心等信息。"
  },
  {
    "agent": "EvidenceAgent",
    "action": "retrieve",
    "tool": "retrieve_tcm_knowledge",
    "summary": "完成中医知识检索。"
  },
  {
    "agent": "SafetyAgent",
    "action": "review",
    "status": "passed",
    "summary": "未发现诊断、处方或剂量表达。"
  }
]
```

首版仍可只把 `last_agent_trace` 存到 thread values；前端是否展示另行处理。

## 8. 错误与降级

| 场景 | 行为 |
| --- | --- |
| InquiryAgent 输出结构无效 | 返回安全错误并记录 trace |
| 信息严重不足 | 生成澄清问题，run 状态进入 `waiting_clarification` |
| 检索无证据 | 继续生成保守回答，说明“目前检索依据有限” |
| 检索工具异常 | 返回降级说明，不伪造 E1-E5 |
| SyndromeAgent 引入未授权术语 | 丢弃该候选项并记录 validation |
| AnswerAgent 初稿不安全 | SafetyAgent 要求重写 |
| 重写后仍不安全 | 返回通用安全答复和线下就医提醒 |

## 9. 测试计划

新增 focused tests：

```text
tests/test_workflow_agent_models.py
tests/test_workflow_agent_flow.py
tests/test_workflow_agent_registry.py
```

覆盖场景：

```text
1. registry 可以解析 assistant_id="workflow_agent"；
2. InquiryAgent 信息不足时最多返回 3 个澄清问题；
3. 信息不足路径不会调用 EvidenceAgent；
4. EvidenceAgent 是唯一调用 retrieve_tcm_knowledge 的固定 Agent；
5. EvidenceAgent 将检索结果整理为 E1-E5 和 allowed_terms；
6. SyndromeAgent 只能输出 possible_patterns，不输出诊断结论；
7. SyndromeAgent 不接受 allowed_terms 之外的术语；
8. AnswerAgent 不新增 evidence 和专业术语；
9. SafetyAgent 能检出处方、剂量、直接诊断和危险信号；
10. SafetyAgent 要求重写时，AnswerAgent 只重写一次；
11. `workflow_agent` 在 `run_agent` smoke test 中能发布 final；
12. `lead_agent` 注册路径保持不变。
```

验证命令优先使用 targeted unittest：

```powershell
python -m unittest tests.test_workflow_agent_models tests.test_workflow_agent_flow tests.test_workflow_agent_registry
python -m unittest tests.test_async_agent_factory tests.test_clarification_flow tests.test_lead_agent_factory
```

## 10. 完成标准

满足以下条件视为完成：

```text
1. `assistant_id="workflow_agent"` 可被 registry 解析；
2. 用户信息严重不足时能暂停并返回澄清问题；
3. 信息足够时固定执行 Evidence/Syndrome/Answer/Safety；
4. 只有 EvidenceAgent 触达 retrieve_tcm_knowledge；
5. 最终回答不包含直接诊断、处方、剂量；
6. SafetyAgent 审查结果进入 trace 或 validation；
7. 旧 `lead_agent` 行为未被替换；
8. focused tests 通过。
```

## 11. 一句话总结

方案 B 的核心是新增一条可对照、可回滚的固定 workflow 主线：

```text
lead_agent 保留旧 prompt/tools 路径，
workflow_agent 承担论文中的固定多 Agent 流程，
EvidenceAgent 管证据，
SafetyAgent 管最终安全闸门。
```
