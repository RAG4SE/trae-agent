# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""JSON formatter tool for formatting final answers in JSON format."""

import json
import re
from typing import override

from trae_agent.tools.base import Tool, ToolCall, ToolCallArguments, ToolExecResult, ToolParameter


class JSONFormatterTool(Tool):
    """Tool for formatting answers in JSON format when required."""

    def __init__(self, model_provider: str):
        """Initialize the JSON formatter tool."""
        super().__init__(model_provider)

    @override
    def get_name(self) -> str:
        """Get the tool name."""
        return "json_formatter"

    @override
    def get_description(self) -> str:
        """Get the tool description."""
        return (
            "Format the final answer in JSON format when the user requests JSON output. "
            "Use this tool when you have completed your analysis and the user has requested "
            "a JSON format response. This tool will extract and format your answer as valid JSON."
        )

    @override
    def get_parameters(self) -> list[ToolParameter]:
        """Get the tool parameters."""
        return [
            ToolParameter(
                name="answer",
                type="string",
                description="The answer text to be formatted as JSON",
                required=True,
            ),
            ToolParameter(
                name="json_template",
                type="string",
                description="Optional JSON template showing the expected format",
                required=False,
            ),
        ]

    @override
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        """Format the answer in JSON format."""
        try:
            answer = arguments.get("answer", "")
            json_template = arguments.get("json_template", "")

            if not answer:
                return ToolExecResult(
                    error="Error: No answer provided to format",
                    error_code=1,
                )

            # Try to extract JSON from the answer
            json_result = self._extract_and_format_json(answer, json_template)

            return ToolExecResult(
                output=json_result,
            )
        except Exception as e:
            return ToolExecResult(
                error=f"Error formatting JSON: {str(e)}",
                error_code=1,
            )

    def _extract_and_format_json(self, answer: str, json_template: str = "") -> str:
        """Extract and format JSON from the answer."""
        # First, try to find JSON in the answer
        json_patterns = [
            r'\{[^{}]*\}',  # Simple JSON objects
            r'\{(?:[^{}]|(?R))*\}',  # Nested JSON objects (recursive)
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, answer, re.DOTALL)
            for match in matches:
                try:
                    # Validate that it's valid JSON
                    parsed = json.loads(match)
                    # Return formatted JSON
                    return json.dumps(parsed, ensure_ascii=False, indent=2)
                except json.JSONDecodeError:
                    continue

        # If no JSON found, try to convert based on template
        if json_template:
            return self._convert_to_json_template(answer, json_template)

        # If no template and no JSON found, create a simple JSON response
        return self._create_simple_json_response(answer)

    def _convert_to_json_template(self, answer: str, template: str) -> str:
        """Convert answer to match the JSON template format."""
        try:
            # Parse the template to understand the expected structure
            template_obj = json.loads(template)

            # Simple heuristic: try to extract key-value pairs from the answer
            result = {}

            # Look for patterns like "key: value" or "key = value"
            patterns = [
                r'(\w+)\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
                r'(\w+(?:\s+\w+)*)\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
            ]

            for pattern in patterns:
                matches = re.findall(pattern, answer)
                for key, value in matches:
                    # Clean up the key and value
                    clean_key = key.strip().lower().replace(' ', '_')
                    clean_value = value.strip()

                    # Try to determine the type of value
                    if clean_value.isdigit():
                        clean_value = int(clean_value)
                    elif clean_value.lower() in ['true', 'false']:
                        clean_value = clean_value.lower() == 'true'

                    # Only add if the key seems relevant to the template
                    if any(clean_key in str(k).lower() for k in template_obj.keys()):
                        result[clean_key] = clean_value

            # If we couldn't extract structured data, put the whole answer in a generic field
            if not result:
                if 'answer' in template_obj:
                    result['answer'] = answer
                elif 'result' in template_obj:
                    result['result'] = answer
                else:
                    # Use the first key from template
                    first_key = list(template_obj.keys())[0]
                    result[first_key] = answer

            return json.dumps(result, ensure_ascii=False, indent=2)

        except json.JSONDecodeError:
            # If template is not valid JSON, create simple response
            return self._create_simple_json_response(answer)

    def _create_simple_json_response(self, answer: str) -> str:
        """Create a simple JSON response when no template is provided."""
        # Try to create a meaningful JSON object from the answer
        result = {"answer": answer}

        # Try to detect if this might be a file analysis result
        if 'file_path:' in answer.lower() or 'file:' in answer.lower():
            # Try to extract file path and declaration/definition
            file_match = re.search(r'file[:\s]+([^\s\n]+)', answer, re.IGNORECASE)
            decl_match = re.search(r'(?:declaration|definition|def)[:]?\s*([^\n]+)', answer, re.IGNORECASE)

            if file_match and decl_match:
                result = {
                    "file_path": file_match.group(1).strip(),
                    "a_decl_or_a_def": decl_match.group(1).strip()
                }

        return json.dumps(result, ensure_ascii=False, indent=2)