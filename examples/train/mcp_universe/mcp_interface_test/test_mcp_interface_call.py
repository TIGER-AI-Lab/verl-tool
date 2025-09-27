#!/usr/bin/env python3
import os
import sys
import json
from pathlib import Path


def ensure_project_on_path() -> None:
    # Add repo root to sys.path so `verl_tool` can be imported when running directly
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def main() -> None:
    ensure_project_on_path()

    # Require gateway address (e.g., http://localhost:8000) per MCP-Universe SSE client
    mcp_addr = os.environ.get("MCP_GATEWAY_ADDRESS", "").rstrip("/")
    if not mcp_addr:
        print("[ERROR] MCP_GATEWAY_ADDRESS not set. Start the gateway and export MCP_GATEWAY_ADDRESS first.")
        print("Example: export MCP_GATEWAY_ADDRESS=http://localhost:8000")
        sys.exit(2)

    # Action payload: read from argv or use a default example
    if len(sys.argv) > 1:
        raw_action = sys.argv[1]
    else:
        raw_action = (
            '<mcp_call>{"server":"google-search","name":"search","arguments":'
            '{"query":"paper accepted by CVPR 2025 last author works at Salesforce '
            'second to last author works at NUS second author studied at NTU '
            'paper uses ELO rating system full title"}}</mcp_call>'
        )

    from verl_tool.servers.tools.mcp_interface import MCPInterfaceTool

    tool = MCPInterfaceTool(num_workers=1)
    trajectory_id = "t1"
    extra_field = {}

    observation, done, valid = tool.conduct_action(
        trajectory_id=trajectory_id,
        action=raw_action,
        extra_field=extra_field,
    )

    # Pretty-print result
    print("valid:", valid)
    print("done:", done)
    if isinstance(observation, (dict, list)):
        print(json.dumps(observation, ensure_ascii=False, indent=2))
    else:
        print(str(observation))


if __name__ == "__main__":
    main()


