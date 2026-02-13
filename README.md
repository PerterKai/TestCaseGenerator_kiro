# Test Case Generator Power

从 `.docx` 需求文档和概要设计文档自动生成结构化测试用例，AI 自动审查优化，最终导出 XMind 和测试报告。

## 安装

在 Kiro 中通过 GitHub 仓库地址直接导入：

```
https://github.com/<owner>/TestCaseGenerator
```

Power 安装后会自动配置 MCP Server。

### 依赖

需要 Python 3.8+ 且安装以下依赖：

```bash
pip install python-docx Pillow openpyxl
```

> `mcp.json` 使用 `python -c` 启动器自动定位 Power 仓库中的服务端脚本，无需手动配置路径。
> 请确保系统 PATH 中的 `python` 已安装上述依赖。

## 使用

1. 在 Kiro 中打开你的项目工作区
2. 将 `.docx` 格式的需求文档放入工作区的 `doc/` 目录
3. 在 Kiro 聊天中输入"生成测试用例"即可启动工作流

## 仓库结构

```
/
├── POWER.md                 # Power 定义（Kiro 识别入口）
├── mcp.json                 # MCP Server 配置
├── steering/                # 工作流引导文件
│   └── testcase-generation-workflow.md
├── server/                  # MCP Server 实现
│   ├── main.py              # MCP Server 主程序（全部工具实现）
│   ├── launcher.py           # MCP Server 启动器（自动定位 main.py）
│   └── gui_llm_config.py    # 外部 LLM API 配置 GUI（tkinter）
├── requirements.txt         # Python 依赖
└── README.md
```

## 工作流程

1. **文档解析** — 将 .docx 转为 Markdown，提取图片
2. **图片分析** — 通过外部多模态 LLM API 逐张分析图片内容（表格、流程图、接口定义等）
3. **用例生成** — 按模块分段读取文档，生成测试用例
4. **模块审查** — 审查模块划分合理性，优化结构
5. **自动 Review** — 多轮迭代审查用例质量
6. **首次导出 + 用例概述** — 输出 XMind + 测试报告，向用户展示生成概述和需求疑问点
7. **用户验收循环** — 用户查看 XMind 确认用例，反馈修改意见后迭代更新，直到用户确认完善

## 跨 Session 支持

所有工作流状态持久化到用户工作区的 `.tmp/cache/`，支持中断后恢复。
