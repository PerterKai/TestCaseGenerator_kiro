# Test Case Generator Power

从 `.docx` 需求文档和概要设计文档自动生成结构化测试用例，AI 自动审查优化，最终导出 XMind 和测试报告。

## 快速开始

### 1. 克隆仓库

```bash
git clone <repo-url>
cd TestCaseGenerator
```

### 2. 安装 Python 依赖

需要 Python 3.10+。

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. 导入 Power 到 Kiro

在 Kiro 中打开本项目文件夹作为工作区，Power 会自动识别 `testcase-power/` 目录。

如果 MCP Server 未自动启动，在 Kiro 的 MCP Server 面板中手动连接 `testcase-generator`。

> **注意**: `mcp.json` 中使用 `"command": "python"`，请确保激活了虚拟环境或系统 PATH 中的 python 已安装依赖。
> 如果使用虚拟环境，可以修改 `testcase-power/mcp.json` 中的 `command` 为虚拟环境的 python 路径。

### 4. 放入需求文档

将 `.docx` 格式的需求文档和概要设计文档放入 `doc/` 目录：

```
doc/
  ├── 需求文档.docx
  └── 概要设计文档.docx
```

### 5. 开始生成

在 Kiro 聊天中输入"生成测试用例"即可启动工作流。

## 项目结构

```
TestCaseGenerator/
├── testcase-power/              # Kiro Power 定义
│   ├── POWER.md                 # Power 说明文档
│   ├── mcp.json                 # MCP Server 配置
│   └── steering/                # 工作流引导文件
│       └── testcase-generation-workflow.md
├── testcase-mcp-server/         # MCP Server 实现
│   └── main.py                  # 服务端主程序
├── doc/                         # 需求文档目录（用户自行放入，不纳入版本控制）
├── .tmp/                        # 运行时临时文件（自动生成，不纳入版本控制）
├── requirements.txt             # Python 依赖
├── .gitignore
└── README.md
```

## 工作流程

1. **文档解析** — 将 .docx 转为 Markdown，提取图片
2. **图片分析** — AI 逐张分析图片内容（表格、流程图、接口定义等）
3. **用例生成** — 按模块分段读取文档，生成测试用例
4. **模块审查** — 审查模块划分合理性，优化结构
5. **自动 Review** — 多轮迭代审查用例质量
6. **导出** — 输出 `需求名_testCase.xmind` + `需求名_testCaseReport.md`

## 输出文件

| 文件 | 说明 |
|------|------|
| `需求名_testCase.xmind` | XMind 格式测试用例 |
| `需求名_testCaseReport.md` | 测试报告（覆盖概览 + 需求疑问点） |

## 跨 Session 支持

所有工作流状态持久化到 `.tmp/cache/`，支持中断后恢复。新 session 中会自动检测并继续上次进度。
