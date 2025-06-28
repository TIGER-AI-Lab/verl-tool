#
# Re-structured and Rephrased Code
#

import os
import re
import random
import sqlite3
import time
import itertools
from collections import defaultdict, namedtuple
from contextlib import contextmanager
from typing import (
    Tuple, Any, List, Set, Literal, Iterator, Dict, Optional, Union
)

import sqlparse

# --- Constants and Type Definitions ---

WHERE_OPS = (
    'not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists'
)
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
VALUE_NUM_SYMBOL = 'VALUERARE'
DEFAULT_TIMEOUT_MS = 15000  # 15 seconds

# A parsed SQL structure, typically represented as a dictionary.
# The exact structure depends on the upstream SQL parser.
ParsedSQL = Dict[str, Any]

# A row from a database query result.
QueryResultRow = Tuple[Any, ...]


# --- Utility Functions ---

def extract_sql_from_markdown(text: str) -> str:
    """
    Extracts the last SQL code block from a markdown-formatted string.

    Args:
        text: The string containing the markdown.

    Returns:
        The extracted SQL query, or an empty string if not found.
    """
    program_pattern = r"```sql[ \t]*[\r\n]+(.*?)[\r\n]+[ \t]*```"
    matches = re.findall(program_pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        query = matches[-1].strip()
        # Clean up common formatting issues
        return query.replace('> =', '>=').replace('< =', '<=').replace('! =', '!=')
    return ""


def replace_current_year(query: str) -> str:
    """
    Replaces YEAR(CURDATE()) with a fixed year (2020) for consistent evaluation.
    """
    return re.sub(
        r"YEAR\s*\(\s*CURDATE\s*\(\s*\)\s*\)", "2020", query, flags=re.IGNORECASE
    )


def remove_distinct_from_query(query: str) -> str:
    """
    Removes the 'DISTINCT' keyword from a SQL query.
    """
    tokens = [t.value for t in list(sqlparse.parse(query)[0].flatten())]
    return ''.join([t for t in tokens if t.lower() != 'distinct'])


# --- Database Interaction Class ---

class DatabaseManager:
    """
    Manages SQLite database connections and query execution with timeouts.
    """
    def __init__(self):
        self._connection_pool: Dict[str, sqlite3.Connection] = {}

    @contextmanager
    def _connection(self, db_path: str) -> Iterator[sqlite3.Connection]:
        """Provides a database connection from the pool."""
        if db_path not in self._connection_pool:
            try:
                # Use immutable=1 for read-only access, which is safer and faster.
                uri_path = f"file:{db_path}?immutable=1"
                conn = sqlite3.connect(uri_path, uri=True, check_same_thread=False)
                # Performance pragmas
                conn.execute('PRAGMA journal_mode=OFF;')
                conn.execute('PRAGMA synchronous=OFF;')
                conn.execute('PRAGMA temp_store=MEMORY;')
                conn.text_factory = lambda b: b.decode(errors="ignore")
                self._connection_pool[db_path] = conn
            except sqlite3.Error as e:
                raise ConnectionError(f"Failed to connect to database at {db_path}: {e}")

        db_conn = self._connection_pool[db_path]
        yield db_conn

    @staticmethod
    @contextmanager
    def _query_timeout(conn: sqlite3.Connection, timeout_ms: int):
        """A context manager to enforce a timeout on a query."""
        deadline = time.perf_counter() + (timeout_ms / 1000)
        
        def handler():
            if time.perf_counter() >= deadline:
                return 1  # Returning 1 interrupts the query
            return 0

        # The progress handler is checked every N virtual machine instructions.
        # Set a low N for short timeouts.
        n_instructions = 100 if timeout_ms <= 100 else 1000
        conn.set_progress_handler(handler, n_instructions)
        try:
            yield
        finally:
            conn.set_progress_handler(None, n_instructions)
            
    def execute_query(
        self,
        db_path: str,
        query: str,
        timeout_ms: int = DEFAULT_TIMEOUT_MS
    ) -> Tuple[Optional[str], Optional[List[QueryResultRow]]]:
        """
        Executes a SQL query against the specified database.

        Args:
            db_path: Path to the SQLite database file.
            query: The SQL query string to execute.
            timeout_ms: Timeout in milliseconds.

        Returns:
            A tuple (error_message, results). If execution is successful,
            error_message is None. If it fails, results is None.
        """
        clean_query = replace_current_year(query)
        
        try:
            with self._connection(db_path) as conn:
                with self._query_timeout(conn, timeout_ms):
                    cursor = conn.cursor()
                    cursor.execute(clean_query)
                    results = cursor.fetchall()
                    cursor.close()
                    return None, results
        except sqlite3.OperationalError as e:
            if "interrupted" in str(e):
                return "Execution timed out.", None
            return f"Database operational error: {e}", None
        except Exception as e:
            return f"An unexpected error occurred: {e}", None

    def close_all_connections(self):
        """Closes all active connections in the pool."""
        for conn in self._connection_pool.values():
            conn.close()
        self._connection_pool = {}


# --- Execution-Based Evaluation Class ---

class ExecutionEvaluator:
    """
    Compares two lists of query results for equivalence.

    This class handles the complex task of determining if two denotations (query results)
    are the same, even if column and row orders differ.
    """
    
    @staticmethod
    def are_results_equivalent(
        result1: List[QueryResultRow],
        result2: List[QueryResultRow],
        order_matters: bool = False
    ) -> bool:
        """
        Checks if two query results are equivalent.

        Args:
            result1: The first list of result rows.
            result2: The second list of result rows.
            order_matters: If True, compares rows as ordered sequences. Otherwise,
                           treats them as multisets.

        Returns:
            True if the results are considered equivalent, False otherwise.
        """
        if len(result1) != len(result2):
            return False
        if not result1:  # Both are empty
            return True
        if len(result1[0]) != len(result2[0]):
            return False

        # --- Quick Rejection ---
        # Sort each row internally and compare the sets of sorted rows.
        # This is a fast way to check for bag-of-rows equivalence.
        s1 = {tuple(sorted(row, key=lambda x: str(x) + str(type(x)))) for row in result1}
        s2 = {tuple(sorted(row, key=lambda x: str(x) + str(type(x)))) for row in result2}
        if s1 != s2:
            return False
            
        # If order matters, a direct comparison is sufficient at this point.
        if order_matters:
            return result1 == result2
            
        # --- Permutation Check for Full Equivalence (order does not matter) ---
        # This handles cases where columns might be in a different order.
        # e.g., SELECT A, B FROM T is equivalent to SELECT B, A FROM T.
        num_cols = len(result1[0])
        
        # Get all possible column mappings (permutations).
        # We can optimize this by pre-filtering permutations based on column value types/sets.
        col_sets1 = [{row[i] for row in result1} for i in range(num_cols)]
        
        possible_perms = ExecutionEvaluator._get_constrained_permutations(col_sets1, result2)
        
        for perm in possible_perms:
            if len(perm) != len(set(perm)):  # Ensure the permutation is valid (no duplicate columns)
                continue
            
            result2_permuted = [
                tuple(element[i] for i in perm) for element in result2
            ]

            # For unordered comparison, we check if the multisets of rows are identical.
            if ExecutionEvaluator._are_multisets_equal(result1, result2_permuted):
                return True
                
        return False

    @staticmethod
    def _are_multisets_equal(list1: List, list2: List) -> bool:
        """Efficiently checks if two lists are equal as multisets."""
        if len(list1) != len(list2):
            return False
        counts = defaultdict(int)
        for item in list1:
            counts[item] += 1
        for item in list2:
            counts[item] -= 1
            if counts[item] < 0:
                return False
        return all(v == 0 for v in counts.values())
        
    @staticmethod
    def _get_constrained_permutations(
        col_sets1: List[Set],
        result2: List[QueryResultRow]
    ) -> Iterator[Tuple[int, ...]]:
        """
        Generates valid column permutations, pruning impossible ones.
        
        A permutation is impossible if a value in a column of result2 does not
        appear in the corresponding permuted column of result1.
        """
        num_cols = len(col_sets1)
        
        # Start with all columns being possible for each position.
        perm_constraints = [set(range(num_cols)) for _ in range(num_cols)]

        # If more than 3 columns, sample rows to constrain the search space.
        if num_cols > 3:
            sample_size = min(20, len(result2))
            for _ in range(sample_size):
                random_row2 = random.choice(result2)
                for i in range(num_cols):  # For each column in table 1
                    # Iterate over a copy as we modify the set
                    for j in list(perm_constraints[i]):  # For each possible mapping in table 2
                        if random_row2[j] not in col_sets1[i]:
                            perm_constraints[i].remove(j)
                            
        return itertools.product(*perm_constraints)


# --- Syntactic Evaluation Class ---

class SyntaxEvaluator:
    """
    Evaluates SQL queries based on their parsed syntactic components.
    
    This class contains methods for comparing specific clauses of two SQL queries,
    such as SELECT, WHERE, GROUP BY, etc., to calculate partial match scores.
    This type of evaluation does not execute the query.
    """

    @staticmethod
    def _get_scores(match_count: int, pred_total: int, gold_total: int) -> Tuple[float, float, float]:
        """Calculates precision, recall, and F1 score."""
        if pred_total != gold_total or gold_total == 0:
            return (0.0, 0.0, 0.0)
        if match_count == pred_total:
            return (1.0, 1.0, 1.0)
        return (0.0, 0.0, 0.0)

    def evaluate_partial_match(self, pred_sql: ParsedSQL, gold_sql: ParsedSQL) -> Dict[str, Dict]:
        """
        Calculates partial match scores for all SQL components.

        Args:
            pred_sql: The parsed predicted SQL query.
            gold_sql: The parsed ground-truth SQL query.

        Returns:
            A dictionary containing scores for each component.
        """
        scores = {}

        # SELECT clause
        gold_total, pred_total, cnt, cnt_wo_agg = self._eval_select(pred_sql, gold_sql)
        acc, rec, f1 = self._get_scores(cnt, pred_total, gold_total)
        scores['select'] = {'acc': acc, 'rec': rec, 'f1': f1}
        
        # WHERE clause
        gold_total, pred_total, cnt, cnt_wo_agg = self._eval_where(pred_sql, gold_sql)
        acc, rec, f1 = self._get_scores(cnt, pred_total, gold_total)
        scores['where'] = {'acc': acc, 'rec': rec, 'f1': f1}
        
        # Other clauses can be added here following the same pattern...
        # self._eval_group(...)
        # self._eval_order(...)

        return scores
        
    def _eval_select(self, pred: ParsedSQL, gold: ParsedSQL) -> Tuple[int, int, int, int]:
        """Evaluates the SELECT clause."""
        pred_select = pred['select'][1]
        gold_select = gold['select'][1]
        gold_select_no_agg = [unit[1] for unit in gold_select]
        
        pred_total = len(pred_select)
        gold_total = len(gold_select)
        
        match_count = 0
        match_count_no_agg = 0

        # Create copies to avoid modifying original lists during iteration
        gold_select_copy = gold_select[:]
        gold_select_no_agg_copy = gold_select_no_agg[:]

        for unit in pred_select:
            if unit in gold_select_copy:
                match_count += 1
                gold_select_copy.remove(unit)
            if unit[1] in gold_select_no_agg_copy:
                match_count_no_agg += 1
                gold_select_no_agg_copy.remove(unit[1])

        return gold_total, pred_total, match_count, match_count_no_agg

    def _eval_where(self, pred: ParsedSQL, gold: ParsedSQL) -> Tuple[int, int, int, int]:
        """Evaluates the WHERE clause."""
        pred_conds = [unit for unit in pred['where'][::2]]
        gold_conds = [unit for unit in gold['where'][::2]]
        gold_conds_no_agg = [unit[2] for unit in gold_conds]

        pred_total = len(pred_conds)
        gold_total = len(gold_conds)
        
        match_count = 0
        match_count_no_agg = 0
        
        gold_conds_copy = gold_conds[:]
        gold_conds_no_agg_copy = gold_conds_no_agg[:]
        
        for unit in pred_conds:
            if unit in gold_conds_copy:
                match_count += 1
                gold_conds_copy.remove(unit)
            if unit[2] in gold_conds_no_agg_copy:
                match_count_no_agg += 1
                gold_conds_no_agg_copy.remove(unit[2])

        return gold_total, pred_total, match_count, match_count_no_agg
    
    # ... Other evaluation functions (eval_group, eval_order, etc.) would go here ...


# --- Main Scoring Function ---

def score(
    predicted_query_str: str,
    ground_truth_info: Dict[str, Any]
) -> Tuple[float, str]:
    """
    Evaluates a predicted SQL query by executing it and comparing results.

    Args:
        predicted_query_str: The predicted SQL, potentially in a markdown block.
        ground_truth_info: A dictionary containing the gold SQL, db_id, etc.

    Returns:
        score: float, (1.0 for a match, 0.0 otherwise)
        pred_results: str, the results of the predicted query execution
        message: str, a message detailing the outcome (e.g., error details).
    """
    db_manager = DatabaseManager()
    evaluator = ExecutionEvaluator()
    
    # Extract info from ground truth
    gold_sql = ground_truth_info['gold_sql']
    # NOTE: Assuming a directory structure where db files are in '/cache/'.
    # This might need to be adjusted based on the actual environment.
    cache_dir = os.getenv('SQL_CACHE_DIR', 'data/nl2sql/cache')
    db_path = os.path.join(cache_dir, ground_truth_info['db_id'])
    comparison_method = ground_truth_info.get('cmp_method', 'bird')

    score = 0.0
    pred_results = ""
    message = ""

    # 1. Execute the ground truth query
    gold_error, gold_results = db_manager.execute_query(db_path, gold_sql)
    if gold_error:
        message = f"Ground truth query failed to execute: {db_path}\n{gold_error}"
        db_manager.close_all_connections()
        return score, pred_results, message

    # 2. Extract and execute the predicted query
    predicted_sql = extract_sql_from_markdown(predicted_query_str)
    if not predicted_sql:
        message = "Prediction is not a valid SQL code block."
        return score, pred_results, message

    pred_error, pred_results = db_manager.execute_query(db_path, predicted_sql)
    if pred_error:
        message = f"Execution Error: {pred_error}"
        db_manager.close_all_connections()
        return score, pred_results, message

    # 3. Compare the results
    try:
        if comparison_method == "spider":
            # Spider evaluation considers order if ORDER BY is present
            order_matters = 'order by' in gold_sql.lower()
            is_match = evaluator.are_results_equivalent(gold_results, pred_results, order_matters)
        else: # Default or 'bird' method
            # BIRD typically uses unordered set comparison
            is_match = evaluator.are_results_equivalent(gold_results, pred_results, order_matters=False)
        
        score = 1.0 if is_match else 0.0
        message = "Success: Results match." if is_match else "Mismatch: Results are not equivalent."
        return score, pred_results, message

    finally:
        # Ensure all database connections are closed
        db_manager.close_all_connections()

# --- Example Usage ---

if __name__ == "__main__":
    # Example 1: A query that should match the ground truth
    predicted_query = """
    Some text before the code...
    ```sql
    SELECT T1.Birthday, T1.SEX, T1.ID, T2.`T-BIL`
    FROM Patient AS T1
    INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID
    ORDER BY T2.`T-BIL` DESC
    LIMIT 1
    ```
    Some text after the code...
    """

    # Example 2: A ground truth that is syntactically different but semantically identical
    # Note: The original ground truth had `T-IL`, which might be a typo. Let's assume it should be `T-BIL`
    # for a successful match, demonstrating the robustness of execution evaluation.
    ground_truth = {
        "db_id": "thrombosis_prediction/thrombosis_prediction.sqlite",
        "gold_sql": "SELECT T1.Birthday, T1.SEX, T1.ID, T2.`T-BIL` FROM Patient AS T1 JOIN Laboratory AS T2 ON T1.ID = T2.ID ORDER BY T2.`T-BIL` DESC LIMIT 1",
        "cmp_method": "bird"  # Using 'bird' for unordered comparison
    }

    # NOTE: To run this example, you need a '/cache' directory with the specified database:
    # /cache/thrombosis_prediction/thrombosis_prediction.sqlite
    # Since we cannot create this environment, the following code is for demonstration.
    
    print("Demonstration of the scoring function.")
    print(f"Predicted Query:\n{extract_sql_from_markdown(predicted_query)}\n")
    print(f"Ground Truth Query:\n{ground_truth['gold_sql']}\n")
    
    # The following line would be executed in a real environment.
    # score, result_message = score_query_execution(predicted_query, ground_truth)
    
    # Mocking the output for demonstration purposes
    score, result_message = 1.0, "Success: Results match (mocked)."

    print(f"Final Score: {score}")
    print(f"Message: {result_message}")