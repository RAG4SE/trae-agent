import asyncio
import contextlib
from enum import Enum

from trae_agent.utils.cli.cli_console import CLIConsole
from trae_agent.utils.config import AgentConfig, Config
from trae_agent.utils.trajectory_recorder import TrajectoryRecorder


class AgentType(Enum):
    TraeAgent = "trae_agent"


class Agent:
    def __init__(
        self,
        agent_type: AgentType | str,
        config: Config,
        trajectory_file: str | None = None,
        cli_console: CLIConsole | None = None,
        allow_edit: bool = True,
    ):
        if isinstance(agent_type, str):
            agent_type = AgentType(agent_type)
        self.agent_type: AgentType = agent_type
        self.allow_edit: bool = allow_edit

        # Set up trajectory recording
        if trajectory_file is not None:
            self.trajectory_file: str = trajectory_file
            self.trajectory_recorder: TrajectoryRecorder = TrajectoryRecorder(trajectory_file)
        else:
            # Auto-generate trajectory file path
            self.trajectory_recorder = TrajectoryRecorder()
            self.trajectory_file = self.trajectory_recorder.get_trajectory_path()

        match self.agent_type:
            case AgentType.TraeAgent:
                if config.trae_agent is None:
                    raise ValueError("trae_agent_config is required for TraeAgent")
                from .trae_agent import TraeAgent

                self.agent_config: AgentConfig = config.trae_agent

                self.agent: TraeAgent = TraeAgent(
                    self.agent_config, allow_edit=allow_edit
                )

                self.agent.set_cli_console(cli_console)

        if cli_console:
            if config.trae_agent.enable_lakeview:
                cli_console.set_lakeview(config.lakeview)
            else:
                cli_console.set_lakeview(None)

        self.agent.set_trajectory_recorder(self.trajectory_recorder)

    async def run(
        self,
        task: str,
        extra_args: dict[str, str] | None = None,
        tool_names: list[str] | None = None,
    ):
        self.agent.new_task(task, extra_args, tool_names)

        if self.agent.allow_mcp_servers:
            if self.agent.cli_console:
                self.agent.cli_console.print("Initialising MCP tools...")
            await self.agent.initialise_mcp()

        if self.agent.cli_console:
            task_details = {
                "Task": task,
                "Model Provider": self.agent_config.model.model_provider.provider,
                "Model": self.agent_config.model.model,
                "Max Steps": str(self.agent_config.max_steps),
                "Trajectory File": self.trajectory_file,
                "Tools": ", ".join([tool.name for tool in self.agent.tools]),
            }
            if extra_args:
                for key, value in extra_args.items():
                    task_details[key.capitalize()] = value
            self.agent.cli_console.print_task_details(task_details)

        cli_console_task = (
            asyncio.create_task(self.agent.cli_console.start()) if self.agent.cli_console else None
        )

        try:
            execution = await self.agent.execute_task()
        finally:
            # Ensure MCP cleanup happens even if execution fails
            with contextlib.suppress(Exception):
                await self.agent.cleanup_mcp_clients()

        if cli_console_task:
            await cli_console_task

        # When allow_edit=False or execution is successful, prioritize json_formatter result
        if execution.success:
            # First check if json_formatter was used and return its result
            json_result = self._extract_json_formatter_result()
            if json_result:
                return json_result

            # If allow_edit is False, extract the final answer from trajectory
            if not self.allow_edit:
                final_answer = self._extract_final_answer()
                if final_answer:
                    return final_answer

        return execution

    def _extract_final_answer(self):
        """Extract the final answer from trajectory when allow_edit is False."""
        import json
        try:
            with open(self.trajectory_file, 'r', encoding='utf-8') as f:
                trajectory_data = json.load(f)

            # Look for the last meaningful response
            for interaction in reversed(trajectory_data.get('llm_interactions', [])):
                if 'response' in interaction and 'content' in interaction['response']:
                    content = interaction['response']['content'].strip()
                    if content and len(content) > 20:
                        return content
            return None
        except Exception:
            return None

    def _extract_json_formatter_result(self):
        """Extract JSON formatter result from trajectory."""
        import json
        try:
            with open(self.trajectory_file, 'r', encoding='utf-8') as f:
                trajectory_data = json.load(f)

            # Look for json_formatter results in agent_steps (primary location)
            for step in reversed(trajectory_data.get('agent_steps', [])):
                if 'tool_calls' in step and step['tool_calls']:
                    # Check if this step contains a json_formatter call
                    has_json_formatter = any(
                        tool_call.get('name') == 'json_formatter'
                        for tool_call in step['tool_calls'] if tool_call
                    )

                    if has_json_formatter and 'tool_results' in step:
                        # Find the corresponding tool result
                        for result in step['tool_results']:
                            if result.get('success') and result.get('result'):
                                result_content = result.get('result', '')
                                # Validate it's JSON-like
                                if result_content.startswith('{') and result_content.endswith('}'):
                                    try:
                                        # Validate it's valid JSON
                                        json.loads(result_content)
                                        return result_content
                                    except json.JSONDecodeError:
                                        continue

            return None
        except Exception:
            return None
