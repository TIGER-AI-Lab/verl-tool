import os
import json
import time
import queue
import atexit
import fcntl
import pathlib
import threading
import requests
from typing import Optional, Union, Dict, List, Any
from urllib.parse import urlencode
import regex as re

import langid
from .base import BaseTool, register_tool

class BingSearchEngine():
    """
    Original Bing search engine that provides web search capability with caching.
    
    This tool interfaces with the Brightdata API to perform Bing searches.
    It includes robust caching to minimize redundant API calls and supports
    both synchronous and asynchronous cache writing modes.
    
    Thread-safety is ensured via memory locks, and process-safety via file locks
    for the cache file.
    """

    def __init__(
        self,
        api_key: str,
        zone: str = "serp_api1",
        max_results: int = 10,
        result_length: int = 1000,
        location: str = "us",
        cache_file: Optional[str] = None,
        async_cache_write: bool = True,
        cache_refresh_interval: float = 15.0
    ):
        """
        Initialize the Bing search engine.
        
        Args:
            api_key: Brightdata API key
            zone: Brightdata zone name
            max_results: Maximum number of search results to return
            result_length: Maximum length of each result snippet
            location: Country code for search localization
            cache_file: Path to cache file (if None, uses ~/.verl_cache/bing_search_cache.json)
            async_cache_write: Whether to write cache updates asynchronously
            cache_refresh_interval: Minimum seconds between cache file checks
        """
        # API configuration
        self._api_key = api_key
        self._zone = zone
        self._max_results = max_results
        self._result_length = result_length
        self._location = location
        
        # Cache and synchronization
        self._cache = {}
        self._cache_lock = threading.Lock()
        self._async_cache_write = async_cache_write
        self._write_queue = queue.Queue() if async_cache_write else None
        self._cache_refresh_interval = cache_refresh_interval
        self._last_cache_check = 0.0
        self._cache_mod_time = 0.0
        
        # Setup cache file paths
        self._setup_cache_paths(cache_file)
        
        # Load existing cache
        self._load_cache()
        
        # Initialize async cache writer if enabled
        if self._async_cache_write:
            self._init_async_writer()
    
    def _setup_cache_paths(self, cache_file: Optional[str]) -> None:
        """
        Set up cache file and lock file paths.
        
        Args:
            cache_file: Path to cache file or None for default
        """
        if cache_file is None:
            cache_dir = pathlib.Path.home() / ".verl_cache"
            cache_dir.mkdir(exist_ok=True)
            self._cache_file = cache_dir / "bing_search_cache.json"
        else:
            self._cache_file = pathlib.Path(cache_file)
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create lock file path
        self._lock_file = str(self._cache_file) + ".lock"
    
    def _init_async_writer(self) -> None:
        """Initialize the asynchronous cache writer thread."""
        self._stop_writer = threading.Event()
        self._writer_thread = threading.Thread(
            target=self._cache_writer_thread, 
            daemon=True,
            name="BingSearchCacheWriter"
        )
        self._writer_thread.start()
        atexit.register(self._cleanup)
    
    def _cleanup(self) -> None:
        """Ensure cache is saved when program exits."""
        if self._async_cache_write and hasattr(self, '_stop_writer'):
            self._stop_writer.set()
            if self._writer_thread.is_alive():
                self._writer_thread.join(timeout=5.0)
            # Save any remaining cache updates
            self._save_cache_sync()
    
    def _cache_writer_thread(self) -> None:
        """Background thread for asynchronous cache writing."""
        while not self._stop_writer.is_set():
            try:
                # Wait for write requests, timeout after 1 second
                try:
                    _ = self._write_queue.get(timeout=1.0)
                    self._save_cache_sync()
                    self._write_queue.task_done()
                except queue.Empty:
                    continue
            except Exception as e:
                print(f"Cache writer thread error: {str(e)}")
    
    def _acquire_file_lock(self, timeout: int = 10) -> Optional[Any]:
        """
        Acquire a file lock to prevent concurrent cache file access.
        
        Args:
            timeout: Maximum seconds to wait for lock acquisition
            
        Returns:
            File descriptor if lock acquired, None if failed
        """
        start_time = time.time()
        lock_fd = None
        
        try:
            # Create or open lock file
            lock_fd = open(self._lock_file, 'w+')
            
            while True:
                try:
                    # Try to acquire exclusive lock
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return lock_fd  # Successfully acquired lock
                except IOError:
                    # Could not immediately acquire lock
                    if time.time() - start_time > timeout:
                        lock_fd.close()
                        raise TimeoutError(f"Failed to acquire file lock within {timeout} seconds")
                    # Retry after a short delay
                    time.sleep(0.1)
        except Exception as e:
            if lock_fd:
                lock_fd.close()
            print(f"Failed to acquire file lock: {str(e)}")
            return None
    
    def _release_file_lock(self, lock_fd: Any) -> bool:
        """
        Release file lock.
        
        Args:
            lock_fd: File descriptor to release
            
        Returns:
            True if successfully released, False otherwise
        """
        if lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
                return True
            except Exception as e:
                print(f"Failed to release file lock: {str(e)}")
                return False
        return False
    
    def _load_cache(self) -> None:
        """Load the cache from disk with file locking."""
        if not self._cache_file.exists():
            return
            
        lock_fd = None
        try:
            # Acquire file lock for reading
            lock_fd = self._acquire_file_lock()
            if not lock_fd:
                print("Unable to acquire file lock, using empty cache")
                return
            
            # Record file modification time
            self._cache_mod_time = os.path.getmtime(self._cache_file)
            
            with open(self._cache_file, "r", encoding="utf-8") as f:
                file_data = json.load(f)
            
            # Update in-memory cache
            with self._cache_lock:
                self._cache = file_data
            
            self._last_cache_check = time.time()
            print(f"Loaded {len(self._cache)} cache entries from {self._cache_file}")
        except json.JSONDecodeError:
            print(f"Cache file {self._cache_file} contains invalid JSON, using empty cache")
            self._cache = {}
        except Exception as e:
            print(f"Failed to load cache file: {str(e)}")
            self._cache = {}
        finally:
            if lock_fd:
                self._release_file_lock(lock_fd)
    
    def _check_cache_update(self) -> bool:
        """
        Check if cache file has been updated by another process.
        
        Returns:
            True if cache was reloaded, False otherwise
        """
        now = time.time()
        # Limit check frequency to avoid excessive I/O
        if now - self._last_cache_check < self._cache_refresh_interval:
            return False
        
        self._last_cache_check = now
        
        if not self._cache_file.exists():
            return False
        
        try:
            current_mod_time = os.path.getmtime(self._cache_file)
            if current_mod_time > self._cache_mod_time:
                print(f"Cache file update detected, reloading")
                self._load_cache()
                return True
        except Exception as e:
            print(f"Failed to check cache file updates: {str(e)}")
        
        return False

    def _save_cache(self) -> None:
        """Save cache to disk, either synchronously or asynchronously."""
        if self._async_cache_write:
            # Queue write request for background thread
            try:
                self._write_queue.put(True, block=False)
            except queue.Full:
                print("Cache write queue full, skipping this update")
        else:
            # Direct synchronous write
            self._save_cache_sync()
    
    def _save_cache_sync(self) -> None:
        """Save cache to disk synchronously with file locking."""
        lock_fd = None
        try:
            # Acquire exclusive file lock
            lock_fd = self._acquire_file_lock()
            if not lock_fd:
                print("Unable to acquire file lock, skipping cache save")
                return
            
            # Create temporary file for atomic write
            temp_file = self._cache_file.with_suffix('.tmp')
            
            # Copy cache data to minimize lock time
            with self._cache_lock:
                cache_copy = dict(self._cache)
            
            # If cache file exists, read and merge with current cache
            merged_cache = self._merge_with_existing_cache(cache_copy)
            
            # Write to temp file and replace original (atomic operation)
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(merged_cache, f, ensure_ascii=False, indent=2)
            
            temp_file.replace(self._cache_file)
            
            # Update modification time record
            self._cache_mod_time = os.path.getmtime(self._cache_file)
            print(f"Cache saved to {self._cache_file} with {len(merged_cache)} entries")
            
            # Update in-memory cache with merged data
            with self._cache_lock:
                self._cache = merged_cache
                
        except Exception as e:
            print(f"Failed to save cache file: {str(e)}")
        finally:
            if lock_fd:
                self._release_file_lock(lock_fd)
    
    def _merge_with_existing_cache(self, cache_copy: Dict) -> Dict:
        """
        Merge in-memory cache with existing cache file content.
        
        Args:
            cache_copy: Copy of the current in-memory cache
            
        Returns:
            Merged cache dictionary
        """
        merged_cache = cache_copy
        if self._cache_file.exists():
            try:
                with open(self._cache_file, "r", encoding="utf-8") as f:
                    existing_cache = json.load(f)
                # Update existing cache with new entries
                existing_cache.update(cache_copy)
                merged_cache = existing_cache
                print(f"Merged with existing cache, total entries: {len(merged_cache)}")
            except Exception as e:
                print(f"Failed to read existing cache file, using new cache: {str(e)}")
        return merged_cache

    @property
    def name(self) -> str:
        """Tool name identifier."""
        return "bing_search"

    @property
    def trigger_tag(self) -> str:
        """Tag used to trigger this tool."""
        return "search"

    def _make_request(self, query: str, timeout: int) -> requests.Response:
        """
        Send request to Brightdata API.

        Args:
            query: Search query
            timeout: Request timeout in seconds

        Returns:
            API response object
        """
        # Determine language settings based on query language
        lang_code, lang_confidence = langid.classify(query)
        if lang_code == 'zh':
            mkt, setLang = "zh-CN", "zh"
        else:
            mkt, setLang = "en-US", "en"
        
        # Prepare URL with query parameters
        encoded_query = urlencode({
            "q": query, 
            "mkt": mkt, 
            "setLang": setLang
        })
        target_url = f"https://www.bing.com/search?{encoded_query}&brd_json=1&cc={self._location}"

        # Prepare headers and payload
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "zone": self._zone,
            "url": target_url,
            "format": "raw"
        }

        # Send request
        return requests.post(
            "https://api.brightdata.com/request",
            headers=headers,
            json=payload,
            timeout=timeout
        )

    def execute(self, query: str, timeout: int = 60) -> str:
        """
        Execute Bing search query.

        Args:
            query: Search query string
            timeout: API request timeout in seconds

        Returns:
            Formatted search results as string
        """
        # Clean query
        query = query.replace('"', '')
        
        # Check if cache file has been updated
        self._check_cache_update()
        
        # Check cache for existing results
        with self._cache_lock:
            if query in self._cache:
                print(f"Cache hit for query: {query}")
                return self._cache[query]

        try:
            # Make API request
            response = self._make_request(query, timeout)

            if response.status_code != 200:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                print(error_msg)
                raise Exception(error_msg)

            # Parse response JSON
            data = json.loads(response.text)

            # Extract search results
            result = self._extract_and_format_results(data)
            
            # Update cache
            with self._cache_lock:
                self._cache[query] = result
            
            # Trigger cache save
            self._save_cache()
                
            return result

        except Exception as e:
            error_msg = f"Bing search failed: {str(e)}"
            print(error_msg)
            return ""
    
    def _extract_and_format_results(self, data: Dict) -> str:
        """
        Extract and format search results from API response.
        
        Args:
            data: API response data
            
        Returns:
            Formatted search results as string
        """
        # If no organic results, return empty response
        if 'organic' not in data:
            data['chunk_content'] = []
            return self._format_results(data)

        # Extract unique snippets
        chunk_content_list = []
        seen_snippets = set()
        for result in data['organic']:
            snippet = result.get('description', '').strip()
            if len(snippet) > 0 and snippet not in seen_snippets:
                chunk_content_list.append(snippet)
                seen_snippets.add(snippet)

        data['chunk_content'] = chunk_content_list
        return self._format_results(data)

    def _format_results(self, results: Dict) -> str:
        """
        Format search results into readable text.
        
        Args:
            results: Dictionary containing search results
            
        Returns:
            Formatted string of search results
        """
        if not results.get("chunk_content"):
            return "No search results found."

        formatted = []
        for idx, snippet in enumerate(results["chunk_content"][:self._max_results], 1):
            snippet = snippet[:self._result_length]
            formatted.append(f"Page {idx}: {snippet}")
        
        return "\n".join(formatted)


@register_tool
class BingSearchTool(BaseTool):
    """
    Bing search tool that follows the BaseTool interface.
    
    This tool wraps the BingSearchEngine to provide search functionality
    while adhering to the standard tool interface.
    """
    
    tool_type = "bing_search"
    
    def __init__(
        self,
        num_workers=1,
        api_key: str = None,
        zone: str = "serp_api1",
        max_results: int = 10,
        result_length: int = 1000,
        location: str = "cn",
        cache_file: Optional[str] = None,
        async_cache_write: bool = True,
        cache_refresh_interval: float = 15.0,
        timeout: int = 60
    ):
        """
        Initialize the Bing search tool.
        
        Args:
            num_workers: Number of workers (inherited from BaseTool)
            api_key: Brightdata API key
            zone: Brightdata zone name
            max_results: Maximum number of search results to return
            result_length: Maximum length of each result snippet
            location: Country code for search localization
            cache_file: Path to cache file
            async_cache_write: Whether to write cache updates asynchronously
            cache_refresh_interval: Minimum seconds between cache file checks
            timeout: Default timeout for search requests
        """
        super().__init__(num_workers)
        
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.getenv('BRIGHTDATA_API_KEY')
            if api_key is None:
                raise ValueError("API key must be provided either as parameter or BRIGHTDATA_API_KEY environment variable")
        
        # Initialize the search engine
        self.search_engine = BingSearchEngine(
            api_key=api_key,
            zone=zone,
            max_results=max_results,
            result_length=result_length,
            location=location,
            cache_file=cache_file,
            async_cache_write=async_cache_write,
            cache_refresh_interval=cache_refresh_interval
        )
        
        self.timeout = timeout
    
    def get_usage_inst(self):
        """
        Get the usage instructions for the tool
        """
        return "Use this tool to search the web using Bing. Provide search queries in <search>query</search> tags or ```search\\nquery\\n``` code blocks."
    
    def parse_action(self, action: str):
        """
        Parse the raw action string to extract the search query.
        
        Args:
            action: The raw action string
            
        Returns:
            tuple: (search_query, is_valid)
        """
        # Try to find search query in various formats
        search_queries = re.findall(r"<search>(.*?)</search>", action, re.DOTALL)
        
        if not search_queries:
            search_queries = re.findall(r"```\n?search\n(.*?)\n```", action, re.DOTALL)
        
        if not search_queries:
            # Try to find any search-like patterns
            search_queries = re.findall(r"search:\s*(.*?)(?:\n|$)", action, re.IGNORECASE)
        
        if len(search_queries) == 0:
            return "", False
        
        # Use the first search query found and clean it
        search_query = search_queries[0].strip()
        
        # Basic validation - ensure query is not empty and reasonable length
        if not search_query:
            return "", False
        
        return search_query, True
    
    def get_action_priority(self, action: str, extra_field: dict) -> int:
        """
        Get the priority for handling this action.
        
        Args:
            action: The raw action string
            extra_field: Extra fields associated with the action
            
        Returns:
            priority: Integer priority (-1 means cannot handle, 0 = default, positive = higher priority)
        """
        _, valid = self.parse_action(action)
        if not valid:
            return -1
        
        # Give higher priority if the action explicitly mentions search
        if any(keyword in action.lower() for keyword in ['<search>', 'search:', '```search']):
            return 2
        
        return 0
    
    def postprocess_observation(self, observation: Union[str, Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
        """
            add <result> tags to the observation
        """
        if isinstance(observation, str):
            # Wrap the observation in <result> tags
            return f"<result>{observation}</result>"
        elif isinstance(observation, dict):
            # If it's a dict, wrap the 'observation' field
            observation['obs'] = f"<result>{observation.get('observation', '')}</result>"
            return observation
        else:
            # If it's neither, return as is
            return observation

    def conduct_action(self, trajectory_id, action, extra_field):
        """
        Conduct the search action and return the observation.
        
        Args:
            trajectory_id: The trajectory ID
            action: The action to conduct (should contain search query)
            extra_field: Extra data to include in the request
            
        Returns:
            tuple: (observation, done, valid)
        """
        parsed_query, is_valid = self.parse_action(action)
        if len(parsed_query) > 500:
            observation = "Search query is too long. Please shorten your query."
            return observation, False, False
        
        env = self.load_env(trajectory_id)
        
        if not is_valid:
            observation = "No valid search query found. Please provide a search query in <search>query</search> tags or ```search\\nquery\\n``` code blocks."
            done = False
            valid = False
        else:
            try:
                # Get timeout from extra_field if provided
                timeout = extra_field.get('timeout', self.timeout) if extra_field else self.timeout
                
                # Execute the search
                search_results = self.search_engine.execute(parsed_query, timeout)
                
                if search_results:
                    observation = f"Search results for '{parsed_query}':\n\n{search_results}"
                    valid = True
                else:
                    observation = f"Search for '{parsed_query}' returned no results or failed."
                    valid = False
                
                # Search action is typically always done after one execution
                done = False
                
            except Exception as e:
                observation = f"Search failed with error: {str(e)}"
                done = False
                valid = False
        
        observation = self.postprocess_observation(observation)

        # Update environment
        self.update_env(trajectory_id, env, parsed_query, is_valid, extra_field, observation)
        self.save_env(trajectory_id, env)
        return observation, done, valid