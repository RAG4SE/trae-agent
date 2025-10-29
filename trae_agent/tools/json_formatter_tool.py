# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""JSON formatter tool for formatting final answers in JSON format using LLM."""

import json
import os
from pathlib import Path

from json_repair import repair_json

from trae_agent.tools.base import Tool, ToolCallArguments, ToolExecResult, ToolParameter
from trae_agent.utils.config import ModelConfig
from trae_agent.utils.llm_clients.llm_basics import LLMMessage
from trae_agent.utils.llm_clients.llm_client import LLMClient


class JSONFormatterTool(Tool):
    """Tool for formatting answers in JSON format using a dedicated formatter LLM."""

    def __init__(self, model_provider: str, json_formatter_model: ModelConfig):
        """Initialize the JSON formatter tool."""
        super().__init__(model_provider)
        self._json_formatter_model = json_formatter_model
        self._formatter_llm_client: LLMClient | None = None

    def get_name(self) -> str:
        """Get the tool name."""
        return "json_formatter"

    def get_description(self) -> str:
        """Get the tool description."""
        return (
            "Format the final answer in strict JSON by delegating to a specialist formatter model. "
            "The tool automatically extracts JSON format requirements from the original task description "
            "(from trajectory file) and uses the `json_formatter_model` from configuration to format "
            "the answer accordingly. The LLM will parse format patterns like 'return json format {key: value}' "
            "and format the answer to match. Provides automatic JSON repair for minor formatting issues."
        )

    def get_parameters(self) -> list[ToolParameter]:
        """Get the tool parameters."""
        return [
            ToolParameter(
                name="answer",
                type="string",
                description="The answer text to be formatted as JSON",
                required=True,
            ),
        ]

    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        """Format the answer in JSON format."""
        try:
            answer = str(arguments.get("answer", "") or "")

            if not answer.strip():
                return ToolExecResult(
                    error="Error: No answer provided to format",
                    error_code=1,
                )

            # Use LLM to extract JSON format from task and format the answer
            json_result = await self._extract_and_format_with_llm(answer)
            return ToolExecResult(output=json_result)
        except Exception as exc:
            return ToolExecResult(
                error=f"Error formatting JSON: {exc}",
                error_code=1,
            )

    async def _extract_and_format_with_llm(self, answer: str) -> str:
        """Use LLM to extract JSON format from task and format the answer in two steps."""

        # Step 1: Get the task description from trajectory file
        task_description = self._get_task_description_from_trajectory()

        # Step 2: Extract JSON format requirement from task
        json_format = await self._extract_json_format_from_task(task_description)

        # Step 3: Format the answer according to the extracted JSON format
        messages = self._build_format_answer_prompt(json_format, answer)
        response = self._call_formatter_llm(messages)
        return self._normalize_and_validate_json(response.content)

    async def _extract_json_format_from_task(self, task_description: str) -> str:
        """Extract JSON format requirement from task description using LLM."""

        if not task_description.strip():
            # If no task description, return empty format
            return ""

        messages = [
            LLMMessage(
                role="system",
                content="Extract the JSON format requirement from the task description. "
                "Look for patterns like 'return json format {key: value}' or similar format specifications. "
                "Extract only the JSON structure (the part inside braces), nothing else. "
                "If no JSON format is found, return an empty string."
            ),
            LLMMessage(
                role="user",
                content=f"Extract JSON format from this task:\n\n{task_description}\n\n"
                "Return only the JSON format structure, no explanations."
            )
        ]

        response = self._call_formatter_llm(messages)
        extracted_format = response.content.strip()

        if not extracted_format:
            return ""

        # Validate and repair the extracted JSON format
        try:
            # Try to parse as JSON
            parsed = json.loads(extracted_format)
            return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            try:
                # Try to repair the JSON
                repaired = repair_json(extracted_format)
                parsed = json.loads(repaired)
                return json.dumps(parsed, ensure_ascii=False)
            except Exception:
                # If repair fails, return empty string
                return ""

    def _build_format_answer_prompt(self, json_format: str, answer: str) -> list[LLMMessage]:
        """Build prompt to format answer according to the extracted JSON format."""

        if not json_format:
            # No specific format found, create a simple structure
            json_format = '{"result": "answer"}'

        system_prompt = """You are a JSON formatting specialist. Format the given answer according to the provided JSON structure.

Instructions:
- Use the exact same keys and structure as provided in the JSON format
- Extract relevant information from the answer to fill the JSON fields
- Set fields to null if the information is not available in the answer
- Return ONLY valid JSON, no explanations or formatting"""

        user_prompt = f"""Please format the answer according to this JSON structure:

<json_format>
{json_format}
</json_format>

<answer_to_format>
{answer}
</answer_to_format>

Please return valid JSON only, with no additional text or formatting."""

        return [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]

    def _get_task_description_from_trajectory(self) -> str:
        """Get the task description from trajectory file."""
        # Try to find trajectory file in common locations
        trajectory_paths = []

        # Check if we can get trajectory file from the agent config (if available)
        # Note: This is a limitation of the current architecture - tools don't have direct access
        # to the agent's trajectory file path. We'll keep the existing fallbacks.

        trajectory_paths.extend([
            Path.cwd() / "trajectory.json",
            Path(__file__).resolve().parents[2] / "trajectory.json",
        ])

        # Also check environment variable
        trajectory_env = os.getenv("TRAJECTORY_FILE")
        if trajectory_env:
            trajectory_paths.insert(0, Path(trajectory_env))

        for trajectory_path in trajectory_paths:
            if trajectory_path.exists():
                try:
                    with open(trajectory_path, 'r', encoding='utf-8') as f:
                        trajectory_data = json.load(f)
                    return trajectory_data.get("task", "")
                except (json.JSONDecodeError, FileNotFoundError, KeyError):
                    continue

        # If no trajectory file found, return empty string
        return ""

    def _build_format_extraction_prompt(self, task_description: str, answer: str) -> list[LLMMessage]:
        """Build prompt for LLM to extract JSON format and format the answer."""

        system_prompt = """You are a JSON formatting specialist. Your task is to:
1. Extract the JSON format requirement from the task description
2. Format the given answer according to that format requirement

Instructions:
- Carefully read the task description to understand the required JSON structure
- Extract the answer content and organize it according to the detected format
- If you cannot find a specific JSON format requirement, create a reasonable JSON structure
- Set missing or unknown values to null
- Return ONLY valid JSON, no explanations or formatting"""

        user_prompt = f"""Please format the answer as JSON according to the requirements in the task description.

<task_description>
{task_description}
</task_description>

<answer_to_format>
{answer}
</answer_to_format>

Please return valid JSON only, with no additional text or formatting."""

        return [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]

    
    def _call_formatter_llm(self, messages: list[LLMMessage]):
        """Call the formatter LLM and return the response."""
        client = self._ensure_formatter_llm_client()
        response = client.chat(messages, self._json_formatter_model, tools=None, reuse_history=False)
        if not response.content or not response.content.strip():
            raise ValueError("Formatter LLM returned empty content.")
        return response

    def _normalize_and_validate_json(self, raw_content: str) -> str:
        """Ensure the formatter response is valid JSON, repairing if necessary."""
        cleaned = self._strip_code_fences(raw_content.strip())
        if not cleaned:
            raise ValueError("Formatter LLM returned empty JSON content.")

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                repaired = repair_json(cleaned)
            except Exception as repair_error:  # pragma: no cover - defensive guard
                raise ValueError("Unable to repair formatter JSON output.") from repair_error
            try:
                parsed = json.loads(repaired)
            except json.JSONDecodeError as parse_error:
                raise ValueError("Formatter JSON output is invalid even after repair.") from parse_error

        return json.dumps(parsed, ensure_ascii=False, indent=2)

    def _strip_code_fences(self, text: str) -> str:
        """Remove markdown code fences if present."""
        trimmed = text.strip()
        if not trimmed.startswith("```"):
            return trimmed

        lines = trimmed.splitlines()
        if not lines:
            return trimmed

        # Drop the opening fence (with optional language identifier)
        if lines[0].startswith("```"):
            lines = lines[1:]

        # Drop the closing fence if present
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]

        return "\n".join(lines).strip()

    def _ensure_formatter_llm_client(self) -> LLMClient:
        """Create the formatter LLM client on demand."""
        if self._formatter_llm_client is None:
            self._formatter_llm_client = LLMClient(self._json_formatter_model)
        return self._formatter_llm_client

    