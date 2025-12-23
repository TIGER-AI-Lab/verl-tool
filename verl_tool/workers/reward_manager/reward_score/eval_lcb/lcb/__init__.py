from .general_utils import run_test
from typing import Optional
import traceback
import multiprocessing
import numpy as np
import regex as re
import json
import pickle
import zlib
import copy
import base64

def has_code(response):
    #pattern = r"```(?:[a-zA-Z]*)\n(.*?)```"
    pattern = r"```python(?:[a-zA-Z0-9]*)\n(.*?)```"
    # Use re.DOTALL to match multiline content inside backticks
    matches = re.findall(pattern, response, re.DOTALL)
    return matches

def check_correctness(problem_to_check: Optional[dict], timeout, debug=False):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""

    def _temp_run(problem_to_check, debug, result, metadata_list, timeout):
        try:
        #if True:
            res, metadata = run_test(problem_to_check, debug=debug, timeout=timeout)
            result.append(res)
            metadata_list.append(metadata)
        except Exception as e:
            traceback.print_exc(10)
            result.append([-1 for i in range(len(problem_to_check['input_output']))])
            metadata_list.append(e)

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()

    #result, metadata_list = [], []
    #_temp_run(problem_to_check, debug, result, metadata_list, timeout)
    #print(stop)

    total_timeout = (timeout + 1) * len(problem_to_check['input_output']) + 10
    p = multiprocessing.Process(target=_temp_run, args=(problem_to_check, debug, result, metadata_list, timeout))
    p.start()
    p.join(timeout=total_timeout + 1)
    if p.is_alive():
        p.kill()
    
    
    judge_value = bool(result and np.all(np.array(result[0]) > 0))

    ret = [
        [0], [0], [0]
    ]


    think_tokens = problem_to_check['generation'].split('</think>')

    summary_tokens = think_tokens[-1]
    think_tokens = think_tokens[0] if len(think_tokens) > 1 else ''

    generation = has_code(summary_tokens)

    if len(generation) >= 1 and generation[0] == 'Your code\n':
        generation = generation[1:]
  

    if judge_value == False:
        if result and result[0][0] <= -100: 
            if result[0][0] == -100:
                ret[1][-1] = 1
            elif result[0][0] == -200:
                ret[0][-1] = 1
                #print(summary_tokens)
            elif result[0][0] == -300:
                ret[2][-1] = 1
                
    if len(metadata_list) == 0:
        assert judge_value == False
        metadata_list.append({'error_message': 'Time Limit Exceeded'})
        #print(problem_to_check['generation'])
    return judge_value, ret, metadata_list[0]

def get_starter_code(header_str):
    if "def " in header_str:
        starter_code = header_str.split("def")[1].split("(")[0].strip()
    else:
        starter_code = header_str

    return starter_code


def load_test_cases(v):
    try:
        return json.loads(
            pickle.loads(
                zlib.decompress(
                    base64.b64decode(v.encode("utf-8"))
                )
            )
        )
    except:
        return json.loads(v)

def eval_lcb(problem_id:str, difficulty:str, generation:str, test_cases:list, starter_code:str="", timeout=10):
    starter_code = get_starter_code(starter_code)
    if isinstance(test_cases, str):
        test_cases = load_test_cases(test_cases)
    
    result = {
        "input_output": test_cases,
        "starter_code": starter_code,
        "question_id": problem_id,
        "generation": generation,
    }
    problem_to_check = copy.deepcopy(result)
    response_entry = {
        "split": difficulty,
        "output": generation,
        "qid": problem_id,
        "reason": None,
    }
    curr_res, ret, logs = check_correctness(problem_to_check,timeout=timeout)
    if curr_res:
        response_entry["correctness"] = True
        response_entry["reason"] = "AC"
    else:
        response_entry["correctness"] = False
        try:
            response_entry["reason"] = logs['error_message']
        except:
            response_entry["reason"] = "UK"
    response_entry['logs'] = json.dumps(logs)
    response_entry["stats"] = ret
    return response_entry