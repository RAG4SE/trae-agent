import asyncio
from trae_agent.agent import Agent
from trae_agent.utils.config import Config

async def run_trae_agent():
    # 1. 加载配置
    config = Config.create(config_file="./trae_configasdasd.yaml")

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
    task = """List the full content of main.py. Return as JSON {"content": <the full content>}. When the main.py is read and the content is extracted, call task_done. 
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