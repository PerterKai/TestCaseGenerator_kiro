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
- 图片通过外部多模态 LLM API 批量分析（支持多线程并发）
- 分析结果直接写回 Markdown 文件，替换对应占位符
- 所有工作流状态持久化到 `.tmp/cache/` 目录，支持跨 session 恢复
- 外部LLM配置通过GUI窗口完成，配置持久化到 `.tmp/cache/llm_api_config.json`

## 工作流程（分阶段，支持跨 session）

### 步骤 0: 环境检查与任务决策（每次启动必做）

调用 `setup_environment`，该工具会自动完成以下检查：
1. **依赖检查**: 检查并安装 Python 依赖（python-docx, Pillow）
2. **目录检查**: 检查工作目录是否存在，不存在则自动创建：
   - `doc/` — 存放 .docx 源文档
   - `.tmp/doc_mk/` — 转换后的 Markdown 文件
   - `.tmp/picture/` — 提取的图片文件
   - `.tmp/cache/` — 工作流状态缓存
3. **缓存任务检测**: 检查是否存在未完成的缓存任务

#### 缓存任务处理逻辑

如果 `setup_environment` 返回 `has_cache=true`，**必须询问用户**：

> 检测到上次未完成的任务（当前阶段: xxx，已处理图片: x/x，已生成用例: x模块x用例）。
> 请选择：
> 1. **继续上次任务** — 从上次中断的位置继续
> 2. **开始新任务** — 清除缓存，重新开始

- 用户选择"继续" → 调用 `get_workflow_state` 恢复进度，按返回的 resume 指令继续
- 用户选择"新任务" → 调用 `clear_cache` 清除缓存，然后进入步骤 1

如果 `has_cache=false`，直接进入步骤 1。

### 步骤 1: 确认文档与处理模式（新任务时）

开始新任务前，需逐步与用户确认：

#### 1.1 确认文档就位
向用户确认：
> 请确认以下文档已放入 `doc/` 目录：
> - 需求文档（.docx）
> - 概要设计文档（.docx）
>
> 确认后我将开始解析。

等待用户确认后再继续。

#### 1.2 选择处理模式
向用户说明两种模式并让用户选择：

> 请选择用例生成模式：
> 1. **文档+图片模式**（需调用外部多模态LLM API，高token消耗）— 解析文档文字和图片，图片通过外部多模态LLM API分析（如GPT-4o、Claude等），支持多线程并发
> 2. **纯文档模式**（低token消耗）— 仅解析文档文字内容，跳过图片分析

- 用户选择"文档+图片模式" → 执行步骤 2，然后调用 `configure_llm_api` 打开GUI配置外部API，配置完成后调用 `process_images_with_llm` 批量处理
- 用户选择"纯文档模式" → 执行步骤 2 后跳过步骤 3，直接进入步骤 4

### 步骤 2: 解析文档 → Markdown + 图片
调用 `parse_documents` 将 `.docx` 文件转换为：
- `.tmp/doc_mk/*.md` — Markdown 格式文档
- `.tmp/picture/*` — 提取的图片文件
- `.tmp/cache/` — 工作流状态缓存
- 返回待处理图片列表

### 步骤 3: 图片处理（仅"文档+图片模式"）

1. 调用 `configure_llm_api` 打开GUI配置窗口
   - 输入API地址（如 `http://localhost:4141/`，支持OpenAI兼容API）
   - 输入API Key（非必填，本地服务可留空）
   - 点击"测试连接"验证API连通性
   - 点击"获取模型列表"获取可用模型
   - 选择支持视觉/多模态的模型（如 gpt-4o、claude-3.5-sonnet 等）
   - 可选：启用多线程并发处理，设置并发线程数（2-10）
   - 点击"确认并保存"（配置自动持久化，下次打开恢复上次输入）
2. 调用 `process_images_with_llm` 批量处理所有待处理图片
   - 如启用多线程，将并发调用外部LLM处理多张图片
   - 分析结果自动回填到Markdown文档对应的 `{{IMG:id}}` 占位符位置
   - 处理进度自动持久化，支持断点续传
   - 如部分图片失败，可重新调用此工具重试

### 步骤 4: 分段获取文档，按模块生成用例
1. 调用 `get_doc_summary` 获取文档结构概览（标题树 + 字数统计）
2. 按模块调用 `get_doc_section(doc_name, section_heading)` 分段读取
3. 每读取一个模块的内容，就生成该模块的测试用例
4. 调用 `save_testcases(append_module=该模块的完整JSON对象)` 增量保存每个模块的用例
   - **`save_testcases` 必须提供 `modules`（全量数组）或 `append_module`（单个模块对象）之一，不能都不传**
   - **始终优先使用 `append_module`**（单个模块对象，非数组），按模块名自动替换已有模块
   - `modules`（全量替换）仅在模块数量极少（≤3个）时使用，大量用例时禁止使用以避免参数截断
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

根据审查结果调整模块结构，对需要调整的模块逐个调用 `save_testcases(append_module=调整后的单个模块对象)` 保存。

### 步骤 5: 自动 Review 与迭代
1. 调用 `get_testcases` 获取当前所有用例
2. 审查功能覆盖、边界条件、异常场景、步骤可执行性
3. 对需要修改的模块，逐个调用 `save_testcases(append_module=修改后的单个模块对象)` 保存
   - **Review 阶段也使用 `append_module` 逐模块更新**，按模块名自动替换，避免全量提交导致参数截断
   - 不需要修改的模块无需重新提交
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

新 session 中调用 `setup_environment` 即可检测缓存任务，询问用户后决定恢复或重新开始。

## 与用户交互

- 启动时如有缓存任务，必须询问用户是继续还是重新开始
- 新任务开始前，必须确认文档已就位
- 新任务开始前，必须让用户选择处理模式（文档+图片 / 纯文档）
- 对文档内容有疑问时，主动向用户确认，不要自行猜测

## 注意事项

- MCP Server 通过 `python -c` 启动器自动定位，从 `~/.kiro/powers/repos/` 中查找 `server/main.py`
- MCP Server 是长驻进程，修改 `main.py` 后需要在 Kiro 的 MCP Server 面板中重连服务器才能生效
- 如果遇到缓存状态不一致，可以调用 `clear_cache` 清除缓存重新开始
- `append_module` 支持同名模块替换，不会产生重复
- `save_testcases` 的 `append_module` 参数必须是单个模块对象（非数组），`modules` 参数必须是数组
- **始终优先使用 `append_module` 逐模块保存**，`modules` 全量替换仅在模块极少（≤3个）时使用，大量用例时会因参数过大导致截断失败
