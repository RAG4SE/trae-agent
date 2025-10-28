import asyncio
from trae_agent.agent import Agent
from trae_agent.utils.config import Config

async def run_trae_agent():
    # 1. 加载配置
    config = Config.create(config_file="./trae_config.yaml")

    # 2. 创建Agent（不使用CLI界面，不允许编辑）
    # 使用轨迹录制功能记录中间过程
    trajectory_file = "trajectory.json"
    agent = Agent(
        agent_type="trae_agent",
        config=config,
        trajectory_file=trajectory_file,  # 记录轨迹到文件
        allow_edit=False  # 不允许编辑，只分析和回答
    )

    # 3. 运行任务
    task = """main.py has a `print(a)` statement, `a` is declared or defined in test1/guguga.py, test2/guguga.py, test3/guguga.py, help decide which `a` is used in `print(a)`, return json format {file_path: <file_path>, a_decl_or_a_def: <stmt>}. When you have found and returned them, call task_done to complete.
    """
    extra_args = {
        "project_path": "/Users/mac/repo/deepwiki-cli/bench/test_var_from_other_file_python",
        "issue": task,
        "must_patch": "false",
    }

    result = await agent.run(task, extra_args)
    return result

# 运行
if __name__ == "__main__":
    result = asyncio.run(run_trae_agent())
    print("=== Agent Answer ===")
    print(result)