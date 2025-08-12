# 智能数据库查询 Agent

## 项目简介
本项目实现了一个基于大语言模型（LLM）和 LangChain 框架的智能数据库查询 Agent，支持自然语言到 SQL 的转换、任务规划、数据库查询执行及结果格式化。

## 主要功能
1. **自然语言转 SQL**：输入中文问题，自动生成 MySQL 查询语句。
2. **任务规划**：将复杂任务拆解为 JSON 步骤列表。
3. **数据库查询执行**：自动连接 MySQL 数据库并执行 SQL 查询。
4. **结果格式化**：将原始查询结果转为结构化中文报告，便于人类阅读。

## 技术栈
- Python 3.10
- vLLM（本地大模型推理）
- HuggingFace Transformers
- LangChain（含 langchain-community, langchain-experimental, langchain-huggingface）
- SQLAlchemy & PyMySQL

## 环境依赖
建议使用 Anaconda 环境，参考 `lc_agent_anaconda.yml` 文件进行安装：
```bash
conda env create -f lc_agent_anaconda.yml
conda activate lc_agent
```

## 主要文件说明
- `agent_4.py`：主程序，包含 Agent 的所有核心逻辑。
- `lc_agent_anaconda.yml`：环境依赖配置文件。

## 使用方法
1. **准备模型**：将 Qwen2.5 系列模型权重放在指定路径（见代码中的 `MODEL_PATH` 和 `PLANNER_MODEL_PATH`）。
2. **配置数据库**：确保 MySQL 数据库可访问，账号密码等信息在代码中明文配置。
3. **运行 Agent**：
   ```bash
   python agent_4.py
   ```
   按提示输入查询任务，Agent 会自动完成任务规划、SQL 生成、查询执行和结果格式化。

## 交互示例
```
请输入查询任务：查询2024年8月12日各楼层的用电量
====== Agent Intermediate Steps ======
[...中间步骤...]
====== Agent 最终输出 ======
[结构化中文报告]
```

## 注意事项
- 需具备 GPU 环境以加速大模型推理。
- `format_output` 工具只能调用一次，并且只能在最后一步。
- 数据库连接失败时，相关功能不可用。

## 致谢
本项目参考了 LangChain 官方文档及 Qwen2.5 模型使用指南。
