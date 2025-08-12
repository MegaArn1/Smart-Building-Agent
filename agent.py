'''
04/23
agent 搭建
'''
# agent.py
import os, json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, pipeline
# from langchain.agents import Tool, LLMChain, PromptTemplate
from langchain.agents import Tool
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
# from langchain.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.agents import initialize_agent, AgentType
# from langchain.sql_database import SQLDatabase
from langchain_community.utilities import SQLDatabase
# from langchain.chains import SQLDatabaseChain
from langchain_experimental.sql import SQLDatabaseChain


# —— 1. Text2SQL 用 vLLM 加载本地模型 —— #

# MODEL_PATH   = "/home/ubuntu/data/models/merge/Qwen/Qwen2.5-7B-Instruct_lora_sft"
MODEL_PATH   = "/home/ubuntu/data/models/merge/Qwen/Qwen2.5-1.5B-Instruct_lora_sft"

EOS_TOKEN    = "<|im_end|>"

# 1.1 tokenizer + stop token
print("Loading Text2SQL Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if EOS_TOKEN not in tokenizer.get_vocab():
    tokenizer.add_special_tokens({"eos_token": EOS_TOKEN})
print("Text2SQL Tokenizer loaded.")

# 1.2 vLLM 实例
print("Loading Text2SQL vLLM Model (targeting default GPU, likely 0)...")
llm_sql = LLM(
    model=MODEL_PATH,
    enable_lora=True,
    trust_remote_code=True,
    # device="cuda", # vLLM typically defaults to CUDA if available
    tensor_parallel_size=1, # Explicitly use 1 GPU for this instance
    gpu_memory_utilization=0.90, # Adjust as needed
    # device="cuda",
    dtype="float16",
    swap_space=100.0,
)
print("Text2SQL vLLM Model loaded.")

# 1.3 采样参数
sampling_params = SamplingParams(
    temperature=0.0, top_p=0.9,
    max_tokens=256,
    stop=[EOS_TOKEN],
)

# 1.4 prompt formatter
def format_qwen_prompt(system_prompt: str, question: str, human_template: str) -> str:
    p = ""
    p += "<|im_start|>system\n"  + system_prompt + "\n<|im_end|>\n"
    p += "<|im_start|>user\n"    + f"问题: {question}\n" + human_template + "\n<|im_end|>\n"
    p += "<|im_start|>assistant\n"
    return p

# 1.5 Text2SQL 调用函数
SYSTEM_PROMPT   = "You are a helpful assistant."
HUMAN_TEMPLATE  = "## 任务 \n\n你是⼀个 MySQL 数据库资深专家。能结合下⾯给定的输⼊信息，按照下⾯的MySQL 的 SQL 语句⽣成规则，⽣成⽤于回答⽤户问题的 SQL 查询语句: \n\n## 数据库的基础信息\n\n1. 表1表名: data_list\n* 描述: 此表用于存储传感器数据的记录。\n* 字段:\n\t* \"sensor_data_number\" (int, 主键, 自动递增): 传感器数据记录的唯一识别号。\n\t* \"id\" (varchar(8)): 设备的标识符，与 device_list 表中的 id 字段相关。\n\t* \"timestamp\" (datetime): 记录数据的时间戳。\n\t* \"co2\" (smallint): 记录的 CO2 浓度。\n\t* \"pm10\" (smallint): 记录的 PM10 浓度。\n\t* \"pm25\" (smallint): 记录的 PM2.5 浓度。\n\t* \"relative_humidity\" (smallint): 相对湿度。\n\t* \"temperature\" (float): 温度。\n\t* \"tvoc\" (smallint): 总挥发性有机化合物浓度。\n\n* 说明: 该表通过 id 字段与 device_list 表关联，记录的传感器数据与设备的唯一标识符相关。\n\n2. 表2表名: device_list\n* 描述: 此表用于存储设备的基本信息。\n* 字段:\n\t* \"device_number\" (int, 主键, 自动递增): 设备的唯一识别号。\n\t* \"id\" (varchar(8)): 设备的标识符，与 data_list 表中的 id 字段相关。\n\t* \"floor (char(3)): 设备所在的楼层编号。可选值包括: F02, F03, F05, F06, F07, F08, F09, F10, F11, F12, F13, F15, F16, F17, F18.\n\t* \"region\" (varchar(10)): 设备所在的区域。可选值包括: 开敞办公区, 茶歇区, 预留办公室, 会议室, 总经理办公室, 预留副总办公室, 贵宾接待室, 副总办公室, 董事长办公室, 财务总监办公室, 董秘办公室, 总经办办公室, 法务、董办办公室, 中会议室, 大会议室, 阶梯教室, 休息区, 羽毛球场, 办公室, 餐厅, 前台, 预留开敞办公区, BIM中心, 绿建中心, 董事会议室, 多功能厅.\n\t* \"device_name\" (varchar(20)): 设备名称。\n\n* 说明: 此表用于管理设备的基础信息，设备通过 id 字段与 data_list 表中的设备数据相关联。\n\n3. 表3表名: electric_meter\n* 描述: 此表用于存储电表的用电记录。\n* 字段:\n\t* \"electric_number\" (int, 主键, 自动递增): 电表记录的唯一识别号。\n\t* \"date\" (date): 电表数据的记录日期。\n\t* \"electricity_usage\" (decimal(9,2)): 电表记录的用电量。\n\t* \"category\" (varchar(20)): 电表的类别。可选值包括: 照明和插座, 空调, 消防电梯, \"消防（主）应急照明a\", 消防风机, 面包房, 机房（单独接线）, \"全大楼楼道、地下室照明\", 水泵房, 双电源自动切换柜, \"商业（总表）\", 楼照明和插座, \"电梯（三台客梯）\", 电梯井道照明, \"消防（备）应急照明b\", 循环泵,\" 消防风机（事故风机）\".\n\t* \"floor\" (char(3)): 电表所在的楼层编号。可选值包括: F01, L01, F02, F03, F05, F06, F07, F08, F09, F10, F11, F12, F13, F15, F16, F17, F18, F19, TOP.\n\n* 说明: 该表记录电表的每日用电量，支持通过楼层信息进行查询。\n\n4. 表4表名: water meter\n* 描述: 此表用于存储水表的用水记录。\n* 字段:\n\t* \"water_number\" (int, 主键, 自动递增): 水表记录的唯一识别号。\n\t* \"date\" (date): 水表数据的记录日期。\n\t* \"water_usage\" (decimal(9,2)): 水表记录的用水量。\n\t* \"floor\" (char(3)): 水表所在的楼层编号。可选值包括: F01, L01, F02, F03, F05, F06, F07, F08, F09, F10, F11, F12, F13, F15, F16, F17, F18, F19.\n\n* 说明: 该表记录水表的每日用水量，支持通过楼层信息进行查询。\n\n## SQL语句的生成规则 \n\n1. 直接输出SQL语句，⽆需任何解释说明\n2. 输出的SQL中\"or\"前后的SQL语句分别⽤括号括起来\n3. 如输⼊问题的查询类型是\"统计数量\", 请在输出的SQL中使⽤ COUNT()函数\n4. 如输⼊问题的查询类型是\"差评\"，评分小于3的是差评\n5. 如问题的查询类型是某月某 \"号\" 或某月某 \"日\"，输出的SQL请注意调用合适的函数根据表的字段说明设计判断逻辑\n6. 如问题的查询类型是 \"月份\"，请在SQL中使用MONTH()函数.\n\n## 问题"

def generate_sql(question: str) -> str:
    prompt = format_qwen_prompt(SYSTEM_PROMPT, question, HUMAN_TEMPLATE)
    resp = llm_sql.generate(prompt, sampling_params)
    out  = resp[0].outputs[0].text
    # 去掉 stop token 及其之后
    return out.split(EOS_TOKEN)[0].strip()

nl2sql_tool = Tool(
    name="nl2sql",
    func=generate_sql,
    description="将自然语言转换为合法的 SELECT SQL 语句"
)

# —— 2. 任务规划模型（示例用 HF pipeline） —— #
PLANNER_GPU_INDEX = 1
PLANNER_MODEL_PATH = "/home/ubuntu/data/models/Qwen/Qwen2.5-3B-Instruct"

try:
    planner_pipe = pipeline(
        "text-generation",
        model=PLANNER_MODEL_PATH,
        device=PLANNER_GPU_INDEX, # <--- Key change: Specify GPU index
        trust_remote_code=True,
        max_new_tokens=1024,
        # max_length=1024,
        # Add other pipeline arguments if needed, e.g., torch_dtype
        # torch_dtype=torch.bfloat16 # Example if using newer GPUs
    )
    planner_llm = HuggingFacePipeline(pipeline=planner_pipe)
    print(f"Planner/Formatter Pipeline loaded on GPU {PLANNER_GPU_INDEX}.")
except Exception as e:
    print(f"Error loading planner pipeline on GPU {PLANNER_GPU_INDEX}: {e}")
    print("Falling back to CPU for planner pipeline.")
    # Fallback to CPU if GPU loading fails
    planner_pipe = pipeline(
        "text-generation",
        model=PLANNER_MODEL_PATH,
        device=-1, # Use CPU
        trust_remote_code=True,
        max_new_tokens=1024,
        # max_length=1024,
    )
    planner_llm = HuggingFacePipeline(pipeline=planner_pipe)
    print("Planner/Formatter Pipeline loaded on CPU.")

plan_prompt = PromptTemplate(
    input_variables=["task"],
    template="""
你是一个任务规划专家。
将用户任务“{task}”拆分为严格的 JSON 步骤列表，每步包含 step_id、action、note：
[
  {{ "step_id": 1, "action": "...", "note": "..." }},
  ...
]
"""
)
plan_chain = LLMChain(llm=planner_llm, prompt=plan_prompt)
plan_tool  = Tool(
    name="plan_task",
    # func=lambda task: plan_chain.run(task),
    func=lambda task: plan_chain.invoke({"task": task})['text'], # Use invoke
    description="将用户任务拆解为 JSON 步骤列表"
)

# —— 3. 数据库执行工具 —— #
import urllib.parse  

# 3.1 账户信息（保持明文，后面统一做转义）
username = "intebuil_visitor"
password = "intebuilvisitor@njmu.edu.cn"   # 注意这里面带有 “@”
host     = "172.16.135.10"
port     = 3306
database = "sensor_production_2"

# 3.2 urlencode，避免 URI 语法冲突
encoded_username = urllib.parse.quote_plus(username)
encoded_password = urllib.parse.quote_plus(password)

DB_URI = (
    f"mysql+pymysql://{encoded_username}:{encoded_password}"
    f"@{host}:{port}/{database}"
)
# >>> 结果示例
# mysql+pymysql://intebuil_visitor:intebuilvisitor%40njmu.edu.cn@172.16.135.10:3306/sensor_production_2

# 3.3 创建 SQLDatabase（保持原来的用法）
print("Connecting to database...")
db = SQLDatabase.from_uri(DB_URI)

try:
    db = SQLDatabase.from_uri(DB_URI)
    print("Database connection successful.")
except Exception as e:
    print(f"FATAL: Database connection failed: {e}")
    # Decide how to handle this - exit? proceed without DB tools?
    # For now, let it raise the error if db is used later.
    db = None # Set db to None if connection fails

# 3.4 构造执行链 & tool

if db:
    # Note: SQLDatabaseChain is in langchain_experimental
    sql_exec_chain = SQLDatabaseChain.from_llm(
        llm=planner_llm, # Uses the LLM potentially on GPU 1 / CPU
        db=db,
        verbose=False,
        return_direct=True # Returns only the SQL result
    )
    exec_tool = Tool(
        name="exec_sql",
        # func=lambda sql: sql_exec_chain.run(sql), # .run is deprecated
        func=lambda sql: sql_exec_chain.invoke({"query": sql})['result'], # Use invoke
        description="执行 SQL 并返回原始查询结果 JSON"
    )
else:
    # Define a dummy tool or handle the absence of the tool if DB connection failed
    def failed_exec_sql(sql):
        return "Database connection failed. Cannot execute SQL."
    exec_tool = Tool(
        name="exec_sql",
        func=failed_exec_sql,
        description="执行 SQL (Currently unavailable due to DB connection issue)"
    )


# —— 4. 结果格式化 —— #

# 在最上面或合适位置定义一个 module‐level flag
_format_output_called = False

def format_once(results):
    global _format_output_called
    if _format_output_called:
        # 第二次调用就直接返回一个空结果或提示
        return "⚠️ format_output 已被调用过一次，请直接给出 Final Answer。"
    _format_output_called = True
    # 真正调用你的 format_chain
    return format_chain.invoke({"results": results})['text']

format_prompt = PromptTemplate(
    input_variables=["results"],
    template="""
以下是数据库查询结果（JSON array of records）：
{results}

请将它转换成条理清晰的结构化语言报告，方便人类阅读。
"""
)
format_chain = LLMChain(llm=planner_llm, prompt=format_prompt)
format_tool  = Tool(
    name="format_output",
    # func=lambda res: format_chain.run(res), # .run is deprecated
    # func=lambda res: format_chain.invoke({"results": res})['text'], # Use invoke
    func=format_once,
    description="将查询结果 JSON 转为结构化语言报告（只能调用一次）"
)

# —— 5. 初始化 Plan‐and‐Execute Agent —— #

tools = [plan_tool, nl2sql_tool, exec_tool, format_tool]

from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

react_prompt = ChatPromptTemplate.from_messages([
     ("system",
      "你是一个 ReAct agent.\n"
      "可用的工具，Action 请从中选取：TOOLS:\n{tool_names}\n\n{tools}\n\n"
      "规则：\n"
      "1. 只能写 Thought / Action / Action Input。\n"
      "2. Observation 由系统提供，永远不要自己写。\n"
      "3. 当你写出 Final Answer: 后，必须立即停止，绝对不能再输出任何 Thought、Action、Observation。\n"
+     "4. **format_output 工具只能被调用一次，并且只能在最后一步**。\n"
      "\n"
      "范例：\n"
      "Thought: 我已得出最终答案\n"
      "Final Answer: <这里放 Markdown 报告>\n"
      "注意，写完 Final Answer 后，必须立即停止，停止一切行为，不能再有任何输出。\n"
     ),
     ("human", "{input}"),
     ("system", "先前步骤：\n{agent_scratchpad}")
 ])

react_agent = create_react_agent(
    llm      = planner_llm,
    tools    = tools,
    prompt   = react_prompt,
)

agent = AgentExecutor(
    agent  = react_agent,
    tools  = tools,
    verbose=True,
    max_iterations=10,
    return_intermediate_steps=True,
    handle_parsing_errors=True # Recommended for robustness
)
# handle_parsing_errors=True # Recommended for robustness

# —— 6. 运行示例 —— #

from langchain_core.exceptions import OutputParserException

if __name__ == "__main__":
    print("Starting agent execution...")
    query = input("请输入查询任务：")
    try:
        result = agent.invoke({
            "input": query,
            "agent_scratchpad": ""
        })
        print("\n====== Agent Intermediate Steps ======\n")
        print(result.get("intermediate_steps", "No intermediate steps recorded."))
        print("\n====== Agent 最终输出 ======\n")
        print(result.get("output", "No final output found."))
    except OutputParserException as e:
        # 兜底：打印模型最后一次原始输出
        print("\n====== 输出解析失败，直接打印模型原始内容 ======\n")
        print(e.raw)    # 或者 e.last_raw_output，根据你用的版本
    except Exception as e:
        print(f"\n====== Agent 执行出错 ======\n")
        import traceback
        traceback.print_exc()
