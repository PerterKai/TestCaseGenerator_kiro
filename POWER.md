---
name: "test-case-generator"
description: "从需求文档和概要设计文档自动生成、迭代审查并导出测试用例的智能Power"
keywords:
  - test case
  - 测试用例
  - test generation
  - xmind
  - 需求文档
  - 概要设计
  - docx
  - QA
  - 测试
  - 用例生成
---

# Test Case Generator Power

从 `.docx` 需求文档和概要设计文档自动生成结构化测试用例，AI 自动审查优化，最终导出 XMind 和测试报告。

## 架构说明

- docx 文档先转换为 Markdown 文件，存放在 `.tmp/doc_mk/` 目录
- 文档中的图片提取到 `.tmp/picture/` 目录，每张图片有唯一标识
- Markdown 文件中用 `{{IMG:唯一标识}}` 占位符标记图片原始位置
- 图片逐一通过 agent 多模态视觉能力分析
- 分析结果直接写回 Markdown 文件，替换对应占位符
- 所有工作流状态持久化到 `.tmp/cache/` 目录，支持跨 session 恢复
- 不依赖 OCR，完全基于 agent 多模态视觉能力

## 工作流程（分阶段，支持跨 session）

### 步骤 0: 恢复进度（新 session 时）
调用 `get_workflow_state` 检查是否有未完成的工作流。
如果有，从上次中断的阶段继续，无需重新解析文档。

### 步骤 1: 环境检查
调用 `setup_environment` 确保依赖就绪。

### 步骤 2: 解析文档 → Markdown + 图片
调用 `parse_documents` 将 `.docx` 文件转换为：
- `.tmp/doc_mk/*.md` — Markdown 格式文档
- `.tmp/picture/*` — 提取的图片文件
- `.tmp/cache/` — 工作流状态缓存
- 返回待处理图片列表

### 步骤 3: 逐一处理图片（支持跨 session）
循环执行：
1. 调用 `get_pending_image` 获取下一张待处理图片
2. 用视觉能力分析图片内容
3. 调用 `submit_image_result(image_id, analysis)` 提交分析结果
4. 重复直到所有图片处理完毕

当系统触发新 session 时：
- 所有图片处理进度已自动持久化到 `.tmp/cache/`
- 新会话中调用 `get_workflow_state` 即可自动恢复进度，继续处理剩余图片
- 不再强制按固定数量切换 session，由系统自行判断

关键原则：
- 每次只获取和处理一张图片
- 提取具体数据，禁止笼统概括

### 步骤 4: 分段获取文档，按模块生成用例
1. 调用 `get_doc_summary` 获取文档结构概览（标题树 + 字数统计）
2. 按模块调用 `get_doc_section(doc_name, section_heading)` 分段读取
3. 每读取一个模块的内容，就生成该模块的测试用例
4. 调用 `save_testcases(append_module=该模块的完整JSON对象)` 增量保存每个模块的用例
   - **`save_testcases` 必须提供 `modules`（全量数组）或 `append_module`（单个模块对象）之一，不能都不传**
   - 生成阶段用 `append_module`（单个模块对象，非数组）；Review 阶段用 `modules`（全量替换）
5. 重复直到所有模块处理完毕
6. 调用 `get_testcases` 确认用例完整性

覆盖维度：正向功能、边界条件、异常处理、安全性、性能、兼容性

用例结构：模块 > 子模块 > 用例标题 > 前置条件 / 执行步骤（合并） / 预期结果

重要：避免使用 `get_parsed_markdown` 一次性加载全部文档，优先使用分段读取。

### 步骤 4.5: 模块结构审查
用例初步生成完毕后，调用 `review_module_structure` 审查模块划分是否合理：
- 检查模块大小是否均衡（避免某模块过大或过小）
- 检查是否有空模块/子模块
- 检查是否有重复命名
- 检查子模块粒度是否合适
- 检查用例质量（前置条件、步骤、预期结果是否完整）

根据审查结果调整模块结构，调整后调用 `save_testcases(modules=调整后的全部用例数组)` 保存。

### 步骤 5: 自动 Review 与迭代
1. 调用 `get_testcases` 获取当前所有用例
2. 审查功能覆盖、边界条件、异常场景、步骤可执行性
3. 修改后调用 `save_testcases(modules=修改后的全部用例数组)` 保存（Review 阶段用 `modules` 全量替换）
4. 重复 1-3，迭代 2-3 轮
5. 在 Review 过程中，记录发现的需求疑问点和确认项

### 步骤 6: 导出 XMind + 报告
1. 调用 `export_xmind` 导出 `.xmind` 文件（自动命名为 `需求名_testCase.xmind`）
2. 调用 `export_report(questions=[...])` 导出测试报告（自动命名为 `需求名_testCaseReport.md`）
   - 传入 `questions` 参数，包含 Review 过程中发现的需求疑问点
   - 报告包含：模块概览、覆盖维度统计、需求疑问点

## 跨 session 恢复机制

所有状态自动持久化到 `.tmp/cache/` 目录：
- `phase_state.json` — 工作流阶段进度
- `image_progress.json` — 图片处理进度
- `testcases.json` — 已生成的测试用例
- `doc_summary.json` — 文档结构摘要

新 session 中调用 `get_workflow_state` 即可恢复到上次中断的位置。

## 与用户交互

对文档内容有疑问时，主动向用户确认，不要自行猜测。

## 注意事项

- MCP Server 通过 `python -c` 启动器自动定位，从 `~/.kiro/powers/repos/` 中查找 `server/main.py`
- MCP Server 是长驻进程，修改 `main.py` 后需要在 Kiro 的 MCP Server 面板中重连服务器才能生效
- 如果遇到缓存状态不一致，可以调用 `parse_documents(force=true)` 重新开始
- `append_module` 支持同名模块替换，不会产生重复
- `save_testcases` 的 `append_module` 参数必须是单个模块对象（非数组），`modules` 参数必须是数组
