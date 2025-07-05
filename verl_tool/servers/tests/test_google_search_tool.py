#!/usr/bin/env python3

import argparse
import asyncio
import json
import aiohttp

async def test_google_search_tool(url: str):
    """Test the Google search tool with sample queries."""
    
    # Test data with different search query formats
    test_data = {
        "trajectory_ids": ["test_1", "test_2", "test_3"],
        "actions": [
            "<search>machine learning algorithms</search>",
            "```search\nPython programming tutorial\n```",
            "What is quantum computing? <search>quantum computing basics</search>"
        ],
        "extra_fields": [
            {"is_last_step": False},
            {"is_last_step": False},
            {"is_last_step": True}  # This will clean up the environment
        ]
    }

    async with aiohttp.ClientSession() as session:
        print(f"Testing Google Search Tool at {url}")
        print("=" * 50)
        
        # Send the request
        async with session.post(url, json=test_data) as response:
            if response.status == 200:
                result = await response.json()
                
                print("Request successful!")
                print(f"Status: {response.status}")
                print("\nResults:")
                
                observations = result.get("observations", [])
                dones = result.get("dones", [])
                valids = result.get("valids", [])
                
                for i, (obs, done, valid) in enumerate(zip(observations, dones, valids)):
                    print(f"\n--- Test {i+1} ---")
                    print(f"Query: {test_data['actions'][i]}")
                    print(f"Valid: {valid}")
                    print(f"Done: {done}")
                    print(f"Observation (first 500 chars): {obs[:500]}...")
                    if len(obs) > 500:
                        print(f"[Truncated - Total length: {len(obs)} characters]")
                    print("-" * 30)
                
            else:
                print(f"Request failed with status: {response.status}")
                error_text = await response.text()
                print(f"Error: {error_text}")

def main():
    parser = argparse.ArgumentParser(description="Test Google Search Tool")
    parser.add_argument("tool_name", help="Tool name (should be 'google_search')")
    parser.add_argument("--url", default="http://localhost:5500/get_observation", 
                       help="URL of the tool server endpoint")
    
    args = parser.parse_args()
    
    if args.tool_name != "google_search":
        print(f"Warning: Expected tool name 'google_search', got '{args.tool_name}'")
    
    print(f"Testing Google Search Tool")
    print(f"Server URL: {args.url}")
    print("Note: Make sure you have set GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables")
    print("or configured the tool server with these credentials.")
    print()
    
    # Run the async test
    asyncio.run(test_google_search_tool(args.url))

if __name__ == "__main__":
    main() 