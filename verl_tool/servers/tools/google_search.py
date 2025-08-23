import os
import json
import time
import pathlib
import threading
import aiohttp
import asyncio
from typing import Optional, Union, Dict, List, Any
import regex as re
import faulthandler
import langid
from .base import BaseTool, register_tool
from .utils.deepsearch_utils import extract_relevant_info_serper, extract_text_from_url, extract_snippet_with_context
from .utils.web_agent_utils import generate_webpage_to_reasonchain, get_prev_reasoning_chain
from tqdm import tqdm

faulthandler.enable() # Enable faulthandler

class GoogleSearchEngine:
    """
    Async Google search engine with basic caching.
    
    Uses aiohttp for non-blocking HTTP requests and simplified caching.
    """

    def __init__(
        self,
        api_key: str,
        max_results: int = 10,
        result_length: int = 1000,
        location: str = "us",
        language: str = "en",
        cache_file: Optional[str] = None,
        process_snippets: bool = False,
        summ_model_url: str = None,
        summ_model_path: str = None,
        max_doc_len: int = 3000
    ):
        """
        Initialize the Google search engine.
        
        Args:
            api_key: Serper API key
            max_results: Maximum number of search results to return
            result_length: Maximum length of each result snippet
            location: Country code for search localization
            language: Language code for search results
            cache_file: Path to cache file
            process_snippets: whether to further process snippets to extract relevant information through web scraping
            summ_model_url: the model used to summarize the processed snippets results
            summ_model_path: the path to the model used for summarization
        """
        # API configuration
        self._api_key = api_key
        self._max_results = max_results
        self._result_length = result_length
        self._location = location
        self._language = language
        
        # Simple cache with thread lock
        self._cache = {}
        self._cache_lock = threading.Lock()
        self._lang_id_lock = threading.Lock()
        self._search_count = 0
        self.process_snippets = process_snippets
        self.summ_model_url = summ_model_url
        self.summ_model_path = summ_model_path
        self._max_doc_len = max_doc_len
        
        # Setup cache file
        self._setup_cache_file(cache_file)
        
        # Load existing cache
        self._load_cache()
        
        # HTTP session for connection pooling
        self._session = None
    
    def _setup_cache_file(self, cache_file: Optional[str]) -> None:
        """Set up cache file path."""
        if cache_file is None:
            cache_dir = pathlib.Path.home() / ".verl_cache"
            cache_dir.mkdir(exist_ok=True)
            if not self.process_snippets:
                self._cache_file = cache_dir / "google_search_cache.jsonl"
            else:
                self._cache_file = cache_dir / "google_search_with_summ_cache.jsonl"
        else:
            self._cache_file = pathlib.Path(cache_file)
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _load_cache(self) -> None:
        """Load cache from JSON file."""
        if not self._cache_file.exists():
            return
            
        try:
            with open(self._cache_file, "r", encoding="utf-8") as f:
                cache_data = [json.loads(line) for line in f if line.strip()]
                cache_data = {item['query']: item['result'] for item in cache_data}
            
            self._cache = cache_data
            self._search_count = len(cache_data)
            print(f"Loaded {len(self._cache)} cache entries")
            
        except Exception as e:
            print(f"Failed to load cache: {e}")
            self._cache = {}
    
    async def _append_cache(self, cache_key: str, cache_value: Union[str, Dict]) -> None:
        """Async cache append without timeout decorator."""
        def _write_cache():
            with self._cache_lock:
                self._cache[cache_key] = cache_value
                self._search_count += 1
                try:
                    with open(self._cache_file, "a", encoding="utf-8") as f:
                        entry = {
                            "query": cache_key,
                            "result": cache_value
                        }
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"Failed to write cache: {e}")
        
        # Run cache write in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _write_cache)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with connection pooling."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=100,  # Total connection pool size
                limit_per_host=30,  # Max connections per host
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={'User-Agent': 'AsyncSearchEngine/1.0'}
            )
        return self._session

    async def _make_request(self, query: str, timeout: int) -> Dict:
        """
        Send async request to Serper API.

        Args:
            query: Search query
            timeout: Request timeout in seconds

        Returns:
            API response data as dict
        """
        # Determine language settings
        try:
            with self._lang_id_lock:
                lang_code, _ = langid.classify(query)
            if lang_code == 'zh':
                hl, gl = "zh-cn", "cn"
            else:
                hl, gl = self._language, self._location
        except:
            hl, gl = self._language, self._location
        
        # Prepare request
        payload = {
            "q": query,
            "hl": hl,
            "gl": gl,
            "num": min(self._max_results, 100)
        }

        headers = {
            'X-API-KEY': self._api_key,
            'Content-Type': 'application/json'
        }

        # Get session and make async request
        session = await self._get_session()
        
        async with session.post(
            "https://google.serper.dev/search",
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as response:
            if response.status != 200:
                text = await response.text()
                raise Exception(f"Search API error {response.status}: {text[:200]}")
            
            return await response.json()

    async def execute(self, query: str, timeout: int = 30, prev_steps: Union[List[str], str] = None) -> str:
        """
        Execute Google search query asynchronously.

        Args:
            query: Search query string
            timeout: API request timeout in seconds
            prev_steps: Previous reasoning steps

        Returns:
            Formatted search results as string
        """
        # Clean and validate query
        query = query.strip().replace('"', '')
        if not query:
            return "Empty search query provided."
        
        try:
            # Check cache first
            if query in self._cache:
                # print(f"Cache hit for: {query}") # debug
                if not self.process_snippets:
                    return self._cache[query]
                else:
                    data = json.loads(self._cache[query])
            else:
                # Make async API request
                data = await self._make_request(query, timeout)
            
            # Process results
            result = await self._extract_and_format_results(query, data, prev_steps)
            
            # Determine cache item
            if not self.process_snippets:
                cache_item = result
            else:
                cache_item = json.dumps(data, ensure_ascii=False)

        except asyncio.TimeoutError:
            error_msg = f"Search request timed out after {timeout} seconds"
            print(error_msg)
            return f"Search failed: {error_msg}"
            
        except Exception as e:
            error_msg = f"Search error: {str(e)}"
            print(error_msg)
            return f"Search failed: {error_msg}"

        try:
            # Save cache item asynchronously
            await self._append_cache(query, cache_item)
        except Exception as e:
            print(f"Failed to save cache: {str(e)}")
            # Don't raise - cache failure shouldn't break search
            
        return result
    
    async def _extract_and_format_results(self, query: str, data: Dict, prev_steps: Union[List[str], str] = None) -> str:
        """Extract and format search results asynchronously."""
        if 'organic' not in data or not data['organic']:
            return "No search results found."
        
        if not self.process_snippets:
            results = []
            seen_snippets = set()
            
            for idx, result in enumerate(data['organic'][:self._max_results], 1):
                title = result.get('title', 'No title').strip()
                link = result.get('link', '').strip()
                snippet = result.get('snippet', result.get('description', '')).strip()
                
                # Skip duplicates
                if snippet and snippet not in seen_snippets:
                    # Truncate if needed
                    if len(snippet) > self._result_length:
                        snippet = snippet[:self._result_length] + "..."
                    formatted = f"**Page {idx}**\n**Title:** {title}\n**Link:** {link}\n**Snippet:** {snippet}\n"
                    results.append(formatted)
                    seen_snippets.add(snippet)

            return "\n".join(results) if results else "No search results found."
        else:
            if self.summ_model_url is None or self.summ_model_path is None:
                max_doc_len = self._result_length
                do_summarization = False
            else:
                max_doc_len = self._max_doc_len
                do_summarization = True
            
            # Extract info and process in thread pool for CPU-bound work
            loop = asyncio.get_event_loop()
            
            def extract_and_process():
                extracted_info = extract_relevant_info_serper(data)
                for info in tqdm(extracted_info, desc="Processing Snippets", disable=True):
                    full_text = extract_text_from_url(info['url'], use_jina=False)
                    if full_text and not full_text.startswith("Error"):
                        success, context = extract_snippet_with_context(full_text, info['snippet'], context_chars=max_doc_len)
                        if success:
                            info['context'] = context
                        else:
                            info['context'] = f"Could not extract context. Returning first {max_doc_len} chars: {full_text[:max_doc_len]}"
                    else:
                        info['context'] = f"Failed to fetch full text: {full_text}"
                return extracted_info
            
            extracted_info = await loop.run_in_executor(None, extract_and_process)
            
            formatted_document = ""
            for i, doc_info in enumerate(extracted_info):
                formatted_document += f"**Web Page {i + 1}:**\n"
                formatted_document += json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"

            if do_summarization:
                # Run summarization in thread pool
                def run_summarization():
                    prev_reasoning_chain = get_prev_reasoning_chain(prev_steps, begin_search_tag="<search>", begin_search_result_tag="<result>")
                    return generate_webpage_to_reasonchain(
                        prev_reasoning_chain,
                        query,
                        formatted_document,
                        summ_model_url=self.summ_model_url,
                        summ_model_path=self.summ_model_path
                    )
                
                summary = await loop.run_in_executor(None, run_summarization)
                return summary
            else:
                return formatted_document if formatted_document else "No relevant information found."

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            
@register_tool
class GoogleSearchTool(BaseTool):
    """
    Async Google search tool with improved reliability.
    """
    
    tool_type = "google_search"
    
    def __init__(
        self,
        num_workers=1,
        api_key: str = None,
        max_results: int = 10,
        result_length: int = 1000,
        location: str = "us",
        language: str = "en",
        cache_file: Optional[str] = None,
        default_timeout: int = 10,
        process_snippets: bool = False,
        summ_model_url: str = None,
        summ_model_path: str = None  # Default model path
    ):
        """
        Initialize the Google search tool.
        """
        super().__init__(num_workers)
        
        # Get API key
        if api_key is None:
            api_key = os.getenv('SERPER_API_KEY')
            if api_key is None:
                raise ValueError("API key required: set SERPER_API_KEY environment variable or pass api_key parameter")
        
        # Initialize search engine
        self.search_engine = GoogleSearchEngine(
            api_key=api_key,
            max_results=max_results,
            result_length=result_length,
            location=location,
            language=language,
            cache_file=cache_file,
            process_snippets=process_snippets,
            summ_model_url=summ_model_url,
            summ_model_path=summ_model_path
        )
        
        self.default_timeout = default_timeout
        self.summ_model_url = summ_model_url
        self.summ_model_path = summ_model_path
        self.process_snippets = process_snippets
    
    def get_usage_inst(self):
        """Get usage instructions."""
        return "Search the web using Google. Use <search>your query</search> or ```search\nyour query\n``` format."
    
    def parse_action(self, action: str):
        """Parse action to extract search query."""
        # Try different patterns
        patterns = [
            r"<search>(.*?)</search>",
            r"```\s*search\s*\n(.*?)\n```",
            r"search:\s*(.*?)(?:\n|$)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, action, re.DOTALL | re.IGNORECASE)
            if matches:
                query = matches[0].strip()
                if query:
                    return query, True
        
        return "", False
    
    def get_action_priority(self, action: str, extra_field: dict) -> int:
        """Get action priority."""
        _, valid = self.parse_action(action)
        if not valid:
            return -1
        
        # Higher priority for explicit search tags
        search_indicators = ['<search>', 'search:', '```search']
        if any(indicator in action.lower() for indicator in search_indicators):
            return 2
        
        return 0
    
    async def aget_observations(self, trajectory_ids: List[str], actions: List[str], extra_fields: List[Dict[str, Any]]):
        """
        Async version of get_observations for better performance.
        """
        observations = []
        dones = []
        valids = []
        
        # Process all actions concurrently
        tasks = []
        for i, (trajectory_id, action, extra_field) in enumerate(zip(trajectory_ids, actions, extra_fields)):
            task = self._conduct_action_async(trajectory_id, action, extra_field)
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                observations.append(f"Search error: {str(result)}")
                dones.append(False)
                valids.append(False)
            else:
                obs, done, valid = result
                observations.append(obs)
                dones.append(done)
                valids.append(valid)
        
        return observations, dones, valids
    
    async def _conduct_action_async(self, trajectory_id: str, action: str, extra_field: Dict[str, Any]):
        """
        Conduct search action asynchronously.
        """
        parsed_query, is_valid = self.parse_action(action)
        
        # Load environment
        env = self.load_env(trajectory_id)
        
        if not is_valid:
            observation = "Invalid search query. Use <search>your query</search> format."
            done, valid = False, False
        elif len(parsed_query) > 500:
            observation = "Search query too long (max 500 characters)."
            done, valid = False, False
        else:
            # Get timeout
            timeout = self.default_timeout
            if extra_field and 'timeout' in extra_field:
                timeout = min(extra_field['timeout'], 60)  # Cap at 60 seconds
            
            prev_actions = [x['action'] for x in env['previous_obs']] if self.process_snippets else None
            
            try:
                # Execute async search
                search_results = await self.search_engine.execute(parsed_query, timeout, prev_actions)
                observation = f"Search results for '{parsed_query}':\n\n{search_results}"
                done, valid = False, True
                
            except Exception as e:
                observation = f"Search error: {str(e)}"
                done, valid = False, False
        
        # Wrap in result tags
        observation = f"<result>{observation}</result>"
        
        # Update environment
        self.update_env(trajectory_id, env, parsed_query, is_valid, extra_field, observation)
        self.save_env(trajectory_id, env)
        
        return observation, done, valid
    
    def conduct_action(self, trajectory_id, action, extra_field):
        """
        Synchronous wrapper for backward compatibility.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._conduct_action_async(trajectory_id, action, extra_field))
        finally:
            loop.close()
    
    def __del__(self):
        """Cleanup when tool is destroyed."""
        if hasattr(self, 'search_engine') and hasattr(self.search_engine, '_session'):
            if self.search_engine._session and not self.search_engine._session.closed:
                # Try to close session gracefully
                try:
                    loop = asyncio.get_event_loop()
                    if not loop.is_closed():
                        loop.create_task(self.search_engine.close())
                except:
                    pass  # Best effort cleanup