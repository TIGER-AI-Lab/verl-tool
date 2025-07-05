import os
import re
import requests
from typing import List, Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor

from .base import BaseTool, register_tool

# --- Config ---
class OnlineSearchConfig:
    def __init__(
        self,
        search_url: str = "https://serpapi.com/search",
        topk: int = 3,
        serp_api_key: Optional[str] = None,
        serp_engine: str = "google",
    ):
        self.search_url = search_url
        self.topk = topk
        self.serp_api_key = serp_api_key
        self.serp_engine = serp_engine

# --- Online Search Wrapper ---
class OnlineSearchEngine:
    def __init__(self, config: OnlineSearchConfig):
        self.config = config

    def _search_query(self, query: str) -> Dict:
        """Execute a single search query using SerpAPI."""
        if not self.config.serp_api_key:
            raise ValueError("SerpAPI key is required")
        
        params = {
            "engine": self.config.serp_engine,
            "q": query,
            "api_key": self.config.serp_api_key,
        }
        
        try:
            response = requests.get(self.config.search_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": f"Search request failed: {str(e)}"}

    def batch_search(self, queries: List[str]) -> List[List[Dict]]:
        """Execute multiple search queries in parallel."""
        results = []
        with ThreadPoolExecutor() as executor:
            for result in executor.map(self._search_query, queries):
                results.append(self._process_result(result))
        return results

    def _process_result(self, search_result: Dict) -> List[Dict]:
        """Process SerpAPI search results into a standardized format."""
        results = []
        
        # Handle API errors
        if "error" in search_result:
            return [{'document': {'contents': search_result["error"]}, 'url': '', 'type': 'error'}]
        
        # Process answer box (direct answers)
        answer_box = search_result.get('answer_box', {})
        if answer_box:
            title = answer_box.get('title', 'No title.')
            snippet = answer_box.get('snippet', 'No snippet available.')
            link = answer_box.get('link', '')
            results.append({
                'document': {"contents": f'"{title}"\n{snippet}'},
                'url': link,
                'type': 'answer_box'
            })

        # Process organic search results
        organic_results = search_result.get('organic_results', [])
        for result in organic_results[:self.config.topk]:
            title = result.get('title', 'No title.')
            snippet = result.get('snippet', 'No snippet available.')
            link = result.get('link', '')
            results.append({
                'document': {"contents": f'"{title}"\n{snippet}'},
                'url': link,
                'type': 'organic'
            })

        # Process related questions
        related_results = search_result.get('related_questions', [])
        for result in related_results[:self.config.topk]:
            title = result.get('question', 'No title.')  # question is the title here
            snippet = result.get('snippet', 'No snippet available.')
            link = result.get('link', '')
            results.append({
                'document': {"contents": f'"{title}"\n{snippet}'},
                'url': link,
                'type': 'related_question'
            })

        return results[:self.config.topk * 3]  # Limit total results

@register_tool
class SerpSearchTool(BaseTool):
    tool_type = "serp_search"
    
    def __init__(self, num_workers=1, search_url="https://serpapi.com/search", 
                 serp_api_key=None, serp_engine="google", topk=3):
        super().__init__(num_workers)
        
        # Get API key from environment if not provided
        self.serp_api_key = serp_api_key or os.getenv('SERP_API_KEY')
        self.search_url = search_url
        self.serp_engine = serp_engine
        self.topk = topk
        
        if not self.serp_api_key:
            raise ValueError("SerpAPI key must be provided either as parameter or environment variable (SERP_API_KEY)")
        
        self.config = OnlineSearchConfig(
            search_url=self.search_url,
            topk=self.topk,
            serp_api_key=self.serp_api_key,
            serp_engine=self.serp_engine
        )
        self.engine = OnlineSearchEngine(self.config)
    
    def get_usage_inst(self):
        return f"You can search the web using SerpAPI ({self.serp_engine} engine). Provide search queries in <search>query</search> tags or ```search query``` code blocks."
    
    def parse_action(self, action: str) -> Tuple[str, bool]:
        """
        Parse the raw action string to extract search queries.
        
        Args:
            action: Raw action string containing search queries
            
        Returns:
            Tuple containing the extracted query and a validity flag
        """
        # Try to find search queries in various formats
        search_queries = re.findall(r"<search>(.*?)</search>", action, re.DOTALL)
        
        if not search_queries:
            search_queries = re.findall(r"```\n?search\n(.*?)```", action, re.DOTALL)
        
        if not search_queries:
            search_queries = re.findall(r"```search\n(.*?)\n```", action, re.DOTALL)
        
        if len(search_queries) == 0:
            return "", False
        
        # Use the first search query found
        parsed_query = search_queries[0].strip()
        
        return parsed_query, True
    
    def conduct_action(self, trajectory_id, action, extra_field):
        """
        Execute the search action using SerpAPI.
        
        Args:
            trajectory_id: ID for tracking the action
            action: Raw action string
            extra_field: Additional parameters
            
        Returns:
            Tuple containing observation, done flag, and validity flag
        """
        parsed_action, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)
        
        if not is_valid:
            observation = "No valid search query found. Please provide search queries in <search>...</search> tags or ```search...``` code blocks."
            done = False
            valid = False
        else:
            try:
                # Execute search using the batch_search method (which handles single queries too)
                search_results = self.engine.batch_search([parsed_action])[0]
                
                # Format the results for display
                if search_results:
                    formatted_results = []
                    for i, result in enumerate(search_results, 1):
                        content = result['document']['contents']
                        url = result.get('url', 'N/A')
                        result_type = result.get('type', 'unknown')
                        
                        # Add type indicator
                        type_indicator = {
                            'answer_box': '📋 Answer Box',
                            'organic': '🔍 Web Result',
                            'related_question': '❓ Related Question',
                            'error': '❌ Error'
                        }.get(result_type, '📄 Result')
                        
                        formatted_results.append(f"{type_indicator} {i}:\nURL: {url}\nContent: {content}")
                    
                    observation = "\n\n".join(formatted_results)
                else:
                    observation = "No search results found."
                
                # Format the observation based on the action type
                if action.endswith("</search>") or "</search>" in action:
                    observation = f"\n<search_results>\n{observation}\n</search_results>\n"
                elif action.strip().endswith("```") or "```search" in action:
                    observation = f"\n```search_results\n{observation}\n```\n"
                else:
                    observation = f"\nSearch Results:\n{observation}\n"
                
                done = False  # Search doesn't end the trajectory
                valid = True
                
            except Exception as e:
                observation = f"Search error: {str(e)}"
                done = False
                valid = False
        
        self.update_env(trajectory_id, env, parsed_action, is_valid, extra_field, observation)
        self.save_env(trajectory_id, env)
        
        return observation, done, valid 