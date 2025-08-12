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

def score(
    predicted_query_str: str,
    ground_truth_info: Dict[str, Any]
) -> Tuple[float, str, str]:
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
    
    db_path = ground_truth_info['db_path']
    sql = predicted_query_str
    gt_sql = ground_truth_info.get('gold_sql') or ground_truth_info.get('gt_sql')
    
    if gt_sql is None:
        return 0.0, "", "No ground truth SQL provided in ground_truth_info"
    
    # Check if database file exists
    if not os.path.exists(db_path):
        return 0.0, "", f"Database file {db_path} does not exist"
    
    # Execute predicted SQL
    pred_result = _execute_sql_for_score(db_path, sql)
    
    # Execute ground truth SQL
    gt_result = _execute_sql_for_score(db_path, gt_sql)
    
    # Extract results and success flags
    pred_success, pred_data, pred_error = pred_result
    gt_success, gt_data, gt_error = gt_result

    
    # Compare results
    if not pred_success:
        return 0.0, "", ""
    
    if not gt_success:
        return 0.0, "", ""
    
    # Compare the actual data
    if pred_data == gt_data:
        return 1.0, "", ""
    else:
        return 0.0, "", ""


def _execute_sql_for_score(db_file: str, sql: str) -> Tuple[bool, Optional[frozenset], Optional[str]]:
    """
    Execute SQL query for scoring purposes.
    
    Returns:
        success: bool, whether execution was successful
        results: frozenset or None, the query results
        error: str or None, error message if any
    """
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        conn.execute("BEGIN TRANSACTION;")
        cursor.execute(sql)
        execution_res = frozenset(cursor.fetchall())
        conn.rollback()
        conn.close()
        return True, execution_res, None
    except Exception as e:
        try:
            conn.rollback()
            conn.close()
        except:
            pass
        return False, None, str(e)


from func_timeout import func_timeout, FunctionTimedOut
import sys
import pandas as pd

def sql_observation(
    predicted_query_str: str,
    ground_truth_info: Dict[str, Any],
    timeout: int = 5
) -> str:
    """
    Generate an observation string for the SQL query.
    """
    
    db_path = ground_truth_info['db_path']
    sql = predicted_query_str
    
    
    if sql is None or sql == "":
        obs = "Your previous action is invalid. Follow the format of outputting thinking process and sql tool, and try again."
    # check if the sql file exists
    elif not os.path.exists(db_path):
        obs = f"The database file {db_path} does not exist."
        
    else:
        db_file = os.path.join(db_path)
        obs = _execute_sql_wrapper(db_file, sql, timeout)

    return obs
    
def _execute_sql(db_file, sql):
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            conn.execute("BEGIN TRANSACTION;")
            cursor.execute(sql)
            execution_res = frozenset(cursor.fetchall())
            conn.rollback()
            conn.close()
            return execution_res
        except Exception as e:
            conn.rollback()
            conn.close()
            return f"Error executing SQL: {str(e)}, db file: {db_file}"
        
def _execute_sql_wrapper(db_file, sql, timeout=5) -> str:
        try:
            res = func_timeout(timeout, _execute_sql, args=(db_file, sql))
            if isinstance(res, frozenset):
                df = pd.DataFrame(res)
                res = df.to_string(index=False)
                # NOTE: observation too long, just truncate
                if len(res) > 9000:
                    # just truncate
                    truncated_df = df.head(50)
                    res = "Truncated to 50 lines since returned response too long: " + truncated_df.to_string(
                        index=False
                    )  # or index=True if you want row numbers
            else:
                res = str(res)

        except KeyboardInterrupt:
            sys.exit(0)
        except FunctionTimedOut:
            res = f"SQL Timeout:\n{sql}"
        except Exception as e:
            res = str(e)

        return res

    