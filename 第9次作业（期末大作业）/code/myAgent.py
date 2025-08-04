import os
import requests
import re
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain.tools import Tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from bs4 import BeautifulSoup
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# -------------------- 初始化设置 --------------------
llm = ChatOpenAI(
    model="qwen-turbo",
    api_key="sk-69609c4ebdda4d1c8e77165ca03cb942",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# -------------------- 工具定义 --------------------
class EnhancedSogouSearchTool:
    """增强版搜狗搜索工具（支持多结果提取）"""
    def __init__(self):
        self.search_url = "https://www.sogou.com/web"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
    
    def run(self, query: str) -> str:
        try:
            params = {"query": query}
            response = requests.get(self.search_url, params=params, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # 提取更多类型的结果
            for item in soup.select(".vr-title, .str_info, .text-layout, .rb"):
                text = item.get_text(strip=True)
                if text and len(text) > 10:  # 过滤短文本
                    results.append(text)
            
            return "\n\n".join(results[:8]) if results else "未找到相关旅行信息"
        except Exception as e:
            return f"搜索失败: {str(e)}"

# 初始化增强版搜狗搜索工具
enhanced_search = EnhancedSogouSearchTool()
search_tool = Tool(
    name="EnhancedSogouSearch",
    func=enhanced_search.run,
    description="使用搜狗搜索引擎获取旅行信息，支持更全面的结果提取"
)

# 预算计算器
def budget_calculator(input_str: str) -> str:
    try:
        days, people, daily_budget = map(int, input_str.split("-"))
        total = days * people * daily_budget
        return f"总预算：{total}元 (天数×人数×每日预算 = {days}×{people}×{daily_budget})"
    except:
        return "请输入正确格式：天数-人数-人均每日预算，例如：7-2-500"

calc_tool = Tool(
    name="BudgetCalculator",
    func=budget_calculator,
    description="计算旅行总预算，输入格式：天数-人数-人均每日预算"
)

# -------------------- 智能体定义 --------------------
class TravelPlanningAgent:
    """智能旅行规划代理（动态生成方案）"""
    def __init__(self):
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""你是一个专业的旅行规划师，根据用户需求生成3个不同的旅行方案。
请严格按照以下格式生成方案：
1. [目的地] [天数]天（[特色]）
2. [目的地] [天数]天（[特色]）
3. [目的地] [天数]天（[特色]）

示例：
1. 三亚 5天（亚龙湾、蜈支洲岛）
2. 青岛 4天（海滨风光、啤酒节）
3. 厦门 5天（鼓浪屿、美食之旅）

注意：必须严格按照上述格式，每行一个方案。"""),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        self.tools = [search_tool]
        self.agent = create_tool_calling_agent(llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools)
    
    def plan(self, state: Dict) -> Dict:
        user_request = state["user_request"]
    
        # 简化分析过程，直接生成方案
        search_query = f"{user_request} 旅行推荐 2024"
        search_result = search_tool.run(search_query)
        
        # 生成个性化方案
        response = self.agent_executor.invoke({
            "messages": [
                HumanMessage(content=f"""根据以下用户需求生成3个旅行方案：
用户需求：{user_request}
搜索结果：{search_result}

请严格按照格式生成：
1. [目的地] [天数]天（[特色]）
2. [目的地] [天数]天（[特色]）  
3. [目的地] [天数]天（[特色]）

要求：
- 每个方案包含明确的目的地和天数
- 天数在3-7天之间
- 特色简洁明了""")
            ],
        })
        
        print(f" AI生成的原始内容：\n{response.get('output', '')}")
        
        # 解析生成的方案
        plans = self._parse_plans_improved(response.get("output", ""))
        if not plans:
            print(" 使用默认方案")
            plans = self._generate_default_plans(user_request)
        
        return {
            "plans": plans[:3],
            "keywords": user_request,
            "default_days": "5"
        }
    
    def _parse_plans_improved(self, content: str) -> List[str]:
        """改进的方案解析逻辑"""
        plans = []
        print(f" 开始解析内容: {content[:200]}...")
        
        # 按行分割内容
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 尝试多种解析模式
            # 模式1: "1. 目的地 天数天（特色）"
            pattern1 = r"^\d+\.\s*([^0-9]+?)\s+(\d+)天\s*[（(]([^）)]+)[）)]"
            match1 = re.match(pattern1, line)
            if match1:
                destination = match1.group(1).strip()
                days = match1.group(2)
                features = match1.group(3).strip()
                plan = f"{destination} {days}天（{features}）"
                plans.append(plan)
                print(f"✅ 模式1匹配: {plan}")
                continue
            
            # 模式2: "目的地 天数天（特色）" (无序号)
            pattern2 = r"^([^0-9]+?)\s+(\d+)天\s*[（(]([^）)]+)[）)]"
            match2 = re.match(pattern2, line)
            if match2:
                destination = match2.group(1).strip()
                days = match2.group(2)
                features = match2.group(3).strip()
                plan = f"{destination} {days}天（{features}）"
                plans.append(plan)
                print(f"✅ 模式2匹配: {plan}")
                continue
            
            # 模式3: 包含天数的任何行
            if '天' in line and re.search(r'\d+天', line):
                # 提取天数
                days_match = re.search(r'(\d+)天', line)
                if days_match:
                    days = days_match.group(1)
                    # 提取目的地（天数前的内容）
                    destination_part = line[:days_match.start()].strip()
                    destination = re.sub(r'^\d+\.\s*', '', destination_part).strip()
                    # 提取特色（括号内容）
                    features_match = re.search(r'[（(]([^）)]+)[）)]', line)
                    features = features_match.group(1) if features_match else "精彩体验"
                    
                    if destination:
                        plan = f"{destination} {days}天（{features}）"
                        plans.append(plan)
                        print(f"✅ 模式3匹配: {plan}")
                        continue
            
            print(f"❌ 无法解析行: {line}")
        
        print(f" 解析结果: 共找到{len(plans)}个方案")
        return plans
    
    def _generate_default_plans(self, user_request: str) -> List[str]:
        """生成默认方案"""
        # 根据用户需求关键词生成默认方案
        if "海" in user_request or "沙滩" in user_request:
            return [
                "三亚 5天（海滨度假、潜水）",
                "青岛 4天（海滨风光、啤酒文化）",
                "厦门 5天（鼓浪屿、海鲜美食）"
            ]
        elif "山" in user_request or "爬山" in user_request:
            return [
                "张家界 4天（天门山、玻璃桥）",
                "黄山 3天（日出云海、温泉）",
                "泰山 3天（日出、古迹探访）"
            ]
        elif "古城" in user_request or "历史" in user_request:
            return [
                "西安 5天（兵马俑、古城墙）",
                "北京 6天（故宫、长城）",
                "南京 4天（明孝陵、夫子庙）"
            ]
        else:
            return [
                "三亚 5天（海滨度假、美食）",
                "桂林 4天（山水风光、漓江）",
                "丽江 5天（古城、雪山）"
            ]

class DestinationResearchAgent:
    """智能目的地研究代理（动态获取信息）"""
    def __init__(self):
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""你是一个旅行目的地专家，负责收集和整理目的地详细信息。
请确保信息与用户选择完全一致，包括：
1. 准确的目的地名称
2. 正确的旅行天数
3. 相关的旅行特色"""),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        self.tools = [search_tool]
        self.agent = create_tool_calling_agent(llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools)
    
    def research(self, state: Dict) -> Dict:
        selected_plan = state["selected_plan"]
        keywords = state.get("keywords", "")
        
        # 精确解析方案信息
        plan_info = self._parse_plan_precise(selected_plan)
        
        # 构建精准搜索查询
        search_query = f"{plan_info['destination']} {plan_info['days']}天旅行攻略"
        if plan_info['features']:
            search_query += f" {plan_info['features']}"
        
        print(f" 解析结果: 目的地={plan_info['destination']}, 天数={plan_info['days']}")
        
        response = self.agent_executor.invoke({
            "messages": [HumanMessage(content=f"""请收集以下旅行目的地的详细信息：
目的地：{plan_info['destination']}
天数：{plan_info['days']}天
特色：{plan_info['features']}

请严格按照以上信息进行搜索和整理，确保信息准确匹配。""")],
        })
        
        return {
            "destination_info": response.get("output", "未能获取该目的地详细信息"),
            "destination": plan_info['destination'],
            "days": plan_info['days']
        }
    
    def _parse_plan_precise(self, plan_str: str) -> Dict:
        """精确解析旅行方案"""
        print(f" 正在解析方案: {plan_str}")
        
        # 清理序号
        clean_plan = re.sub(r"^\d+\.\s*", "", plan_str.strip())
        
        # 标准格式解析: "目的地 天数天（特色）"
        pattern = r"^(.+?)\s+(\d+)天\s*[（(]([^）)]*)[）)]"
        match = re.match(pattern, clean_plan)
        
        if match:
            destination = match.group(1).strip()
            days = match.group(2)
            features = match.group(3).strip() if match.group(3) else ""
            
            print(f"✅ 解析成功: 目的地={destination}, 天数={days}, 特色={features}")
            return {
                "destination": destination,
                "days": days,
                "features": features
            }
        
        # 简化格式解析: "目的地 天数天"
        pattern_simple = r"^(.+?)\s+(\d+)天"
        match_simple = re.match(pattern_simple, clean_plan)
        
        if match_simple:
            destination = match_simple.group(1).strip()
            days = match_simple.group(2)
            
            print(f"✅ 简化解析成功: 目的地={destination}, 天数={days}")
            return {
                "destination": destination,
                "days": days,
                "features": ""
            }
        
        # 如果仍然解析失败，尝试提取关键信息
        days_match = re.search(r'(\d+)天', clean_plan)
        if days_match:
            days = days_match.group(1)
            destination = re.sub(r'\d+天.*', '', clean_plan).strip()
            if destination:
                print(f"⚠️ 兜底解析: 目的地={destination}, 天数={days}")
                return {
                    "destination": destination,
                    "days": days,
                    "features": ""
                }
        
        # 最后的兜底方案
        print(f"❌ 解析失败，使用默认值")
        return {
            "destination": "三亚",
            "days": "5",
            "features": "海滨度假"
        }

class BudgetPlanner:
    """预算规划师（动态计算）"""
    def calculate(self, state: Dict) -> Dict:
        days = state["days"]
        destination = state["destination"]
        people = state["people"]
        daily_budget = state["daily_budget"]
        
        print(f" 预算计算: {destination} {days}天 {people}人 每人每日{daily_budget}元")
        
        # 使用用户输入的参数计算预算
        budget_input = f"{days}-{people}-{daily_budget}"
        budget_result = calc_tool.run(budget_input)
        print(f" 预算结果: {budget_result}")
        
        return {"budget": budget_result}

class TravelAssistant:
    """旅行助手（智能整合）"""
    def summarize(self, state: Dict) -> Dict:
        destination = state["destination"]
        days = state["days"]
        people = state["people"]
        daily_budget = state["daily_budget"]
        
        print(f" 生成最终计划: {destination} {days}天 {people}人")
        
        prompt = f"""请为以下旅行信息生成详细的旅行计划书：

【确认信息】
目的地：{destination}
天数：{days}天
人数：{people}人
人均每日预算：{daily_budget}元
方案：{state['selected_plan']}

【目的地详情】
{state['destination_info']}

【预算信息】
{state['budget']}

请严格按照以上信息生成计划，特别要求：
1. 目的地必须是：{destination}
2. 天数必须严格按照：{days}天设计行程
3. 必须为每一天（第1天到第{days}天）安排具体活动
4. 预算控制在每人每日{daily_budget}元以内
5. 考虑{people}人的团队出行特点

格式要求：
【目的地】{destination}
【行程天数】{days}天
【出行人数】{people}人
【人均预算】{daily_budget}元/天

【详细行程安排】
第1天：[具体安排]
第2天：[具体安排]
...
第{days}天：[具体安排]

【预算分解】
【实用建议】"""
        
        response = llm.invoke(prompt)
        return {
            "final_plan": response.content,
            "destination": destination,
            "days": days,
            "people": people,
            "daily_budget": daily_budget
        }

# -------------------- 工作流定义 --------------------
class TravelState(TypedDict):
    user_request: str
    plans: List[str]
    selected_plan: str
    destination_info: str
    destination: str
    days: str
    people: str
    daily_budget: str
    budget: str
    final_plan: str
    keywords: str
    default_days: str

# 构建工作流 - 修改为只生成方案的简单流程
plan_workflow = StateGraph(TravelState)
plan_workflow.add_node("manager", TravelPlanningAgent().plan)
plan_workflow.set_entry_point("manager")
plan_workflow.add_edge("manager", END)

# 构建完整的处理流程
full_workflow = StateGraph(TravelState)
full_workflow.add_node("expert", DestinationResearchAgent().research)
full_workflow.add_node("planner", BudgetPlanner().calculate)
full_workflow.add_node("assistant", TravelAssistant().summarize)

full_workflow.set_entry_point("expert")
full_workflow.add_edge("expert", "planner")
full_workflow.add_edge("planner", "assistant")
full_workflow.add_edge("assistant", END)

plan_app = plan_workflow.compile()
full_app = full_workflow.compile()

# -------------------- 运行系统 --------------------
if __name__ == "__main__":
    print("=== 23级熊彦钧人工智能大作业：智能旅行规划系统===")
    print("示例：想去海边度假5天，喜欢潜水和水上运动")
    
    # 获取用户输入
    user_request = input("\n请输入您的旅行需求：")
    
    # 获取人数和预算
    while True:
        try:
            people = int(input("请输入出行人数："))
            if people > 0:
                break
            print("人数必须大于0")
        except ValueError:
            print("请输入有效数字")
    
    while True:
        try:
            daily_budget = int(input("请输入人均每日预算（元）："))
            if daily_budget > 0:
                break
            print("预算必须大于0")
        except ValueError:
            print("请输入有效数字")
    
    initial_state = {
        "user_request": user_request,
        "people": str(people),
        "daily_budget": str(daily_budget),
        "plans": [],
        "selected_plan": "",
        "destination_info": "",
        "destination": "",
        "days": "",
        "budget": "",
        "final_plan": "",
        "keywords": "",
        "default_days": "5"
    }

    # 优化的方案选择函数
    def select_plan_interactive(plans):
        if plans:
            print("\n 系统为您生成了以下旅行方案：")
            for i, plan in enumerate(plans, 1):
                print(f"{i}. {plan}")
            
            while True:
                try:
                    choice = int(input("\n请输入选择的方案编号(1-3): "))
                    if 1 <= choice <= len(plans):
                        selected = plans[choice-1]
                        print(f"\n✅ 您选择了: {selected}")
                        return selected
                    print("❌ 请输入有效的编号(1-3)")
                except ValueError:
                    print("❌ 请输入数字")
                except KeyboardInterrupt:
                    print("\n程序已退出")
                    exit()
        else:
            print("⚠️ 未生成有效方案，使用默认方案")
            return "三亚 5天（海滨度假）"

    # 运行流程
    print(f"\n 正在为{people}人分析旅行需求（人均每日预算{daily_budget}元）...")
    
    try:
        # 第一步：只生成方案
        print(" 正在生成旅行方案...")
        plan_result = plan_app.invoke(initial_state)
        
        if plan_result.get("plans"):
            # 交互式选择
            selected_plan = select_plan_interactive(plan_result["plans"])
            
            # 第二步：处理选择的方案
            print(f"\n 正在处理选择的方案: {selected_plan}")
            
            # 构建处理状态
            process_state = initial_state.copy()
            process_state.update(plan_result)
            process_state["selected_plan"] = selected_plan
            
            # 运行完整处理流程
            final_result = full_app.invoke(process_state)
            
            # 输出最终结果
            print(f"\n 最终旅行计划 ")
            print(f" 目的地: {final_result['destination']}")
            print(f" 天数: {final_result['days']}天")
            print(f" 人数: {final_result['people']}人")
            print(f" 预算: {final_result['budget']}")
            print("\n" + "="*60)
            print(final_result["final_plan"])
            print("="*60 + "\n")
        else:
            print("❌ 系统未能生成有效的旅行方案，请重试")
    
    except ValueError as e:
        print(f"❌ 解析错误: {e}")
        print("请重新运行程序并选择其他方案")
    except KeyboardInterrupt:
        print("\n程序已退出")
    except Exception as e:
        print(f"❌ 系统错误: {e}")
        print("请重新运行程序")