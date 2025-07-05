import os
import re
import requests
import asyncio
import random
from typing import List, Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor

import chardet
import aiohttp
import bs4
from googleapiclient.discovery import build

from .base import BaseTool, register_tool

# --- Config ---
class OnlineSearchConfig:
    def __init__(self, topk: int = 3, api_key: Optional[str] = None, cse_id: Optional[str] = None, snippet_only: bool = False):
        self.topk = topk
        self.api_key = api_key
        self.cse_id = cse_id
        self.snippet_only = snippet_only

# --- Utilities ---
def parse_snippet(snippet: str) -> List[str]:
    segments = snippet.split("...")
    return [s.strip() for s in segments if len(s.strip().split()) > 5]

def sanitize_search_query(query: str) -> str:
    # Remove or replace special characters that might cause issues.
    sanitized_query = re.sub(r'[^\w\s]', ' ', query)  # Replace non-alphanumeric and non-whitespace with spaces.
    sanitized_query = re.sub(r'[\t\r\f\v\n]', ' ', sanitized_query) # replace tab, return, formfeed, vertical tab with spaces.
    sanitized_query = re.sub(r'\s+', ' ', sanitized_query).strip() #remove duplicate spaces, and trailing/leading spaces.
    return sanitized_query

def filter_links(search_results: List[Dict]) -> List[str]:
    links = []
    for result in search_results:
        for item in result.get("items", []):
            if "mime" in item:
                continue
            ext = os.path.splitext(item["link"])[1]
            if ext in ["", ".html", ".htm", ".shtml"]:
                links.append(item["link"])
    return links

async def fetch(session: aiohttp.ClientSession, url: str, semaphore: asyncio.Semaphore) -> str:
    user_agents = [
        "Mozilla/5.0 (Linux; Android 6.0.1; Nexus 5X Build/MMB29P) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2272.96 Mobile Safari/537.36 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (compatible; Googlebot/2.1; +https://www.google.com/bot.html)",
    ]
    headers = {"User-Agent": random.choice(user_agents)}

    async with semaphore:
        try:
            async with session.get(url, headers=headers) as response:
                raw = await response.read()
                detected = chardet.detect(raw)
                encoding = detected["encoding"] or "utf-8"
                return raw.decode(encoding, errors="ignore")
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return ""

async def fetch_all(urls: List[str], limit: int = 8) -> List[str]:
    semaphore = asyncio.Semaphore(limit)
    timeout = aiohttp.ClientTimeout(total=5)
    connector = aiohttp.TCPConnector(limit_per_host=limit, force_close=True)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [fetch(session, url, semaphore) for url in urls]
        return await asyncio.gather(*tasks)

# --- Search Engine ---
class OnlineSearchEngine:
    def __init__(self, config: OnlineSearchConfig):
        self.config = config

    def collect_context(self, snippet: str, doc: str) -> str:
        snippets = parse_snippet(snippet)
        ctx_paras = []

        for s in snippets:
            pos = doc.replace("\n", " ").find(s)
            if pos == -1:
                continue
            sta = pos
            while sta > 0 and doc[sta] != "\n":
                sta -= 1
            end = pos + len(s)
            while end < len(doc) and doc[end] != "\n":
                end += 1
            para = doc[sta:end].strip()
            if para not in ctx_paras:
                ctx_paras.append(para)

        return "\n".join(ctx_paras)

    def fetch_web_content(self, search_results: List[Dict]) -> Dict[str, str]:
        links = filter_links(search_results)
        contents = asyncio.run(fetch_all(links))
        content_dict = {}
        for html, link in zip(contents, links):
            soup = bs4.BeautifulSoup(html, "html.parser")
            text = "\n".join([p.get_text() for p in soup.find_all("p")])
            content_dict[link] = text
        return content_dict

    def search(self, search_term: str, num_iter: int = 1) -> List[Dict]:
        if not self.config.api_key or not self.config.cse_id:
            raise ValueError("Google API key and CSE ID must be provided")
        
        service = build('customsearch', 'v1', developerKey=self.config.api_key)
        results = []
        sanitize_search_term = sanitize_search_query(search_term)
        if search_term.isspace():
            return results
        res = service.cse().list(q=sanitize_search_term, cx=self.config.cse_id).execute()
        results.append(res)

        for _ in range(num_iter - 1):
            if 'nextPage' not in res.get('queries', {}):
                break
            start_idx = res['queries']['nextPage'][0]['startIndex']
            res = service.cse().list(q=search_term, cx=self.config.cse_id, start=start_idx).execute()
            results.append(res)

        return results

    def _retrieve_context(self, query: str) -> List[Dict]:
        try:
            search_results = self.search(query)
            contexts = []
            
            if self.config.snippet_only:
                for result in search_results:
                    for item in result.get("items", []):
                        title = item.get("title", "")
                        context = ' '.join(parse_snippet(item.get("snippet", "")))
                        if title != "" or context != "":
                            title = "No title." if not title else title
                            context = "No snippet available." if not context else context
                            contexts.append({
                                'document': {"contents": f'"{title}"\n{context}'},
                                'url': item.get("link", ""),
                            })
            else:
                content_dict = self.fetch_web_content(search_results)
                for result in search_results:
                    for item in result.get("items", []):
                        link = item["link"]
                        title = item.get("title", "")
                        snippet = item.get("snippet", "")
                        if link in content_dict:
                            context = self.collect_context(snippet, content_dict[link])
                            if title != "" or context != "":
                                title = "No title." if not title else title
                                context = "No snippet available." if not context else context
                                contexts.append({
                                    'document': {"contents": f'"{title}"\n{context}'},
                                    'url': link,
                                })
            
            return contexts[:self.config.topk]
        except Exception as e:
            return [{'document': {'contents': f'Search error: {str(e)}'}, 'url': ''}]

@register_tool
class GoogleSearchTool(BaseTool):
    tool_type = "google_search"
    
    def __init__(self, num_workers=1, api_key=None, cse_id=None, topk=3, snippet_only=False):
        super().__init__(num_workers)
        
        # Get credentials from environment if not provided
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        self.cse_id = cse_id or os.getenv('GOOGLE_CSE_ID')
        self.topk = topk
        self.snippet_only = snippet_only
        
        if not self.api_key or not self.cse_id:
            raise ValueError("Google API key and CSE ID must be provided either as parameters or environment variables (GOOGLE_API_KEY, GOOGLE_CSE_ID)")
        
        self.config = OnlineSearchConfig(
            topk=self.topk,
            api_key=self.api_key,
            cse_id=self.cse_id,
            snippet_only=self.snippet_only
        )
        self.engine = OnlineSearchEngine(self.config)
    
    def get_usage_inst(self):
        return "You can search the web using Google Custom Search. Provide search queries in <search>query</search> tags or ```search query``` code blocks."
    
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
        Execute the search action.
        
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
                search_results = self.engine._retrieve_context(parsed_action)
                
                # Format the results for display
                if search_results:
                    formatted_results = []
                    for i, result in enumerate(search_results, 1):
                        content = result['document']['contents']
                        url = result.get('url', 'N/A')
                        formatted_results.append(f"Result {i}:\nURL: {url}\nContent: {content}")
                    
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