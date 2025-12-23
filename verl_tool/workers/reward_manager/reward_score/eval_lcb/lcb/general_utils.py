import ast
import json
import sys
import faulthandler
import platform
import re
# used for debugging to time steps
from datetime import datetime
import os
# to run the solution files we're using a timing based approach
import signal
import time
import numpy as np
from io import StringIO

# used for testing the code that reads from input
from unittest.mock import patch, mock_open

# from pyext import RuntimeModule
from types import ModuleType

from enum import Enum
from decimal import Decimal
import time
from typing import List

#import_string = "import sys\nsys.setrecursionlimit(1 << 25)\nsys.set_int_max_str_digits(1 << 25)\nimport time\nimport re\nimport bisect\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n"

#import_string = "import sys\nsys.setrecursionlimit(1 << 25)\nimport time\nimport re\nimport bisect\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import *\nimport fractions\nfrom typing import *\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n"
import_string = "import sys\nsys.setrecursionlimit(1 << 25)\nsys.set_int_max_str_digits(1 << 20)\nimport time\nimport re\nimport bisect\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2, comb\nimport fractions\nfrom typing import *\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n"



def convert_python_literal_to_json(s):
    obj = ast.literal_eval(s)
    return json.dumps(obj)

def convert_python_literal(arg_str):
    s = arg_str.strip()
    # If the whole string starts with '[' and ends with ']', assume it's one literal
    # and force it to be treated as a single-element tuple by adding a trailing comma.
    if s.startswith('[') and s.endswith(']'):
        wrapped_str = f"({arg_str},)"
    else:
        wrapped_str = f"({arg_str})"
    
    try:
        parsed_tuple = ast.literal_eval(wrapped_str)
    except Exception as e:
        raise ValueError(f"Error parsing argument string: {arg_str}; error: {e}") from e
    return list(parsed_tuple)

def post_process_code(code):
    code = code.split("</code>")[0]
    code = code.replace("```python", "")
    code = code.split("```")[0]
    code = code.replace("<code>", "")
    return code

def extract_functions(code_str):
    tree = ast.parse(code_str)
    extracted = []
    
    # Iterate over top-level nodes in the module.
    for node in tree.body:
        # Check for import statements.
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign)):
            code_segment = ast.get_source_segment(code_str, node)
            extracted.append(code_segment)
        # Check for top-level function definitions.
        elif isinstance(node, ast.FunctionDef):
            code_segment = ast.get_source_segment(code_str, node)
            extracted.append(code_segment)
        # Check for class definitions.
        elif isinstance(node, ast.ClassDef):
            code_segment = ast.get_source_segment(code_str, node)
            extracted.append(code_segment)
    
    return extracted

def has_code(response):
    #pattern = r"```(?:[a-zA-Z]*)\n(.*?)```"
    pattern = r"```python(?:[a-zA-Z0-9]*)\n(.*?)```"
    # Use re.DOTALL to match multiline content inside backticks
    matches = re.findall(pattern, response, re.DOTALL)
    return matches

def truncatefn(s, length=300):
    if isinstance(s, str):
        pass
    else:
        s = str(s)
    if len(s) <= length:
        return s

    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


class CODE_TYPE(Enum):
    call_based = 0
    standard_input = 1


# stuff for setting up signal timer
class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    print("timeout occured: alarm went off")
    raise TimeoutException


# used to capture stdout as a list
# from https://stackoverflow.com/a/16571630/6416660
# alternative use redirect_stdout() from contextlib
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


#def clean_if_name(code: str) -> str:
#    try:
#        astree = ast.parse(code)
#        last_block = astree.body[-1]
#        if isinstance(last_block, ast.If):
#            condition = last_block.test
#            if ast.unparse(condition).strip() == "__name__ == '__main__'":
#                code = (
#                    ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)  # type: ignore
#                )
#    except:
#        pass
#
#    return code

def clean_if_name(code: str) -> str:
    try:
        # Parse the code into an AST.
        astree = ast.parse(code)
        if not astree.body:
            return code

        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            # Extract the source text for the if-statement's condition.
            condition_segment = ast.get_source_segment(code, last_block.test)
            if condition_segment and condition_segment.strip() == "__name__ == '__main__'":
                # Extract code segments for all nodes before the if block.
                pre_if_segments = []
                for node in astree.body[:-1]:
                    segment = ast.get_source_segment(code, node)
                    if segment is not None:
                        pre_if_segments.append(segment)
                # Extract code segments for the nodes in the if block's body.
                if_body_segments = []
                for node in last_block.body:
                    segment = ast.get_source_segment(code, node)
                    if segment is not None:
                        if_body_segments.append(segment)
                # Combine the segments with a newline.
                code = "\n".join(pre_if_segments + if_body_segments)
    except Exception:
        pass

    return code

def make_function(code: str) -> str:
    try:
        import_stmts = []
        other_stmts = []
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_stmts.append(node)
            else:
                other_stmts.append(node)

        # Extract the original source segments for the import statements.
        imports_code = []
        for node in import_stmts:
            seg = ast.get_source_segment(code, node)
            if seg:
                imports_code.append(seg)
        
        # Extract the source segments for all other statements.
        body_code = []
        for node in other_stmts:
            seg = ast.get_source_segment(code, node)
            if seg:
                body_code.append(seg)

        # Create the body of the new function by indenting the extracted statements.
        indented_body = []
        for stmt in body_code:
            indented_lines = [
                "    " + line if line.strip() else line 
                for line in stmt.splitlines()
            ]
            indented_body.append("\n".join(indented_lines))
        
        # Construct the function definition string.
        function_code = "def wrapped_function():\n" + "\n".join(indented_body).replace(" global ", " nonlocal ")
        # Combine the import string, the imports, and the function code.
        main_code = (
            import_string + "\n" +
            "\n".join(imports_code) + "\n" +
            function_code
        )
        return main_code
    except Exception:
        return code


def call_method(method, inputs):

    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))

    # sys.setrecursionlimit(10000)

    # @patch('builtins.input', side_effect=inputs.split("\n"))
    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", StringIO(inputs))
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    # @patch('sys.stdout.write', print)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit as e:
            pass
        finally:
            pass

    return _inner_call_method(method)


def get_function(compiled_sol, fn_name: str):  # type: ignore
    try:
        assert hasattr(compiled_sol, fn_name)
        return getattr(compiled_sol, fn_name)
    except Exception as e:
        return


def compile_code(code: str, timeout: int):
    signal.alarm(timeout)
    try:
        tmp_sol = ModuleType("tmp_sol", "")
        exec(code, tmp_sol.__dict__)
        if "class Solution:" in code:
            # leetcode wraps solutions in `Solution`
            # this is a hack to check if it is leetcode solution or not
            # currently livecodebench only supports LeetCode but
            # else condition allows future extensibility to other platforms
            compiled_sol = tmp_sol.Solution()
        else:
            # do nothing in the other case since function is accesible
            compiled_sol = tmp_sol

        assert compiled_sol is not None
    finally:
        signal.alarm(0)
        #pass

    return compiled_sol


# def convert_line_to_decimals(line: str) -> tuple[bool, List[Float]]:
def convert_line_to_decimals(line: str):
    try:
        decimal_line = [float(elem) for elem in line.split()]
    except:
        return False, []
    return True, decimal_line


def get_stripped_lines(val: str):
    ## you don't want empty lines to add empty list after splitlines!
    val = val.strip()

    return [val_line.strip() for val_line in val.split("\n")]

def replace_newlines(text):
    result = []
    depth = 0  # tracks nesting level for [] and {}
    for char in text:
        if char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
        # Replace newline with comma only if not nested.
        if char == "\n" and depth == 0:
            result.append(",")
        else:
            result.append(char)
    return "".join(result)

def grade_ioi(
    code: str, all_inputs: list, all_outputs: list, fn_name: str, timeout: int
):
        
    # ioi-based clean up logic
    # need to wrap in try-catch logic after to catch the correct errors, but for now this is fine.
    base_code = import_string + "\n\n" + code + '\n\n'

    if "class Solution" in code:
        base_code += 'def run_main_code(self):\n'
    else:
        base_code += 'def run_main_code():\n'


    all_inputs = [
            replace_newlines(inputs) for inputs in all_inputs
    ]


    triggered = False

    try:
        all_outputs = [json.loads(output) for output in all_outputs]
        triggered = True
    except Exception:
        pass

    if not triggered:
        try:
            all_outputs = [
                json.loads(convert_python_literal_to_json(outputs)) for outputs in all_outputs
            ]
            triggered = True
        except Exception:
            pass

    if not triggered:
        all_outputs = [
            convert_python_literal(outputs) for outputs in all_outputs
        ]

    total_execution = 0
    all_results = []
    for idx, (gt_inp, gt_out) in enumerate(zip(all_inputs, all_outputs)):
        code = base_code
        if 'class Solution' in code:
            code += f"    return self.{fn_name}({gt_inp})\nSolution.run_main_code = run_main_code\n"
        else:
            code += f"    return {fn_name}({gt_inp})\n"

        compiled_sol = compile_code(code, timeout)
        if compiled_sol is None:
            return [-200], {"Compile Error while running the testcases."}  

        method = get_function(compiled_sol, "run_main_code")
        
        if method is None:
            return [-200], {"Compile Error while running the testcases."}

        signal.alarm(timeout)
        faulthandler.enable()
        try:
            # can lock here so time is useful
            start = time.time()
            prediction = method()
            total_execution += time.time() - start
            signal.alarm(0)

            tmp_result = float(prediction)

            all_results.append(tmp_result)
            if tmp_result < 0:
                return all_results, {
                    "inputs": truncatefn(gt_inp),
                    "output": truncatefn(prediction),
                    "expected": truncatefn(gt_out),
                    "error_code": -2,
                    "error_message": "Invalid Output",
                }

        except Exception as e:
            signal.alarm(0)
            if "timeoutexception" in repr(e).lower():
                all_results.append(-3)
                return all_results, {
                    "error": repr(e),
                    "error_code": -3,
                    "error_message": "Time Limit Exceeded",
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                }
            else:
                if "NameError" in repr(e):
                    all_results.append(-200)
                else:
                    all_results.append(-300)
                return all_results, {
                    "error": repr(e),
                    "error_code": -4,
                    "error_message": "Runtime Error",
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                }

        finally:
            signal.alarm(0)
            faulthandler.disable()

    return all_results, {"execution time": total_execution}




def grade_call_based(
    code: str, all_inputs: list, all_outputs: list, fn_name: str, timeout: int
):
    # call-based clean up logic
    # need to wrap in try-catch logic after to catch the correct errors, but for now this is fine.
    base_code = import_string + "\n\n" + code + '\n\n'

    if "class Solution" in code:
        base_code += 'def run_main_code(self):\n'
    else:
        base_code += 'def run_main_code():\n'

    #print(all_inputs)

    all_inputs = [
            replace_newlines(inputs) for inputs in all_inputs
    ]

    #print("Input", all_inputs)

    # print("Ouput", all_outputs)
    triggered = False

    try:
        all_outputs = [json.loads(output) for output in all_outputs]
        triggered = True
    except Exception:
        pass

    if not triggered:
        try:
            all_outputs = [
                json.loads(convert_python_literal_to_json(outputs)) for outputs in all_outputs
            ]
            triggered = True
        except Exception:
            pass

    if not triggered:
        all_outputs = [
            convert_python_literal(outputs) for outputs in all_outputs
            #json.loads(convert_python_literal_to_json('[' + outputs + ']')) for outputs in all_outputs
        ]   
    #print("Ouput", all_outputs)
    total_execution = 0
    all_results = []
    for idx, (gt_inp, gt_out) in enumerate(zip(all_inputs, all_outputs)):
        code = base_code
        #print(code)

        #if "resultsArray" in code:
        #    print(idx)

        if 'class Solution' in code:
            code += f"    return self.{fn_name}({gt_inp})\nSolution.run_main_code = run_main_code\n"
        else:
            code += f"    return {fn_name}({gt_inp})\n"
        #print(code)
        compiled_sol = compile_code(code, timeout)
        if compiled_sol is None:
            return [-200], {"Compile Error while running the testcases."}  

        method = get_function(compiled_sol, "run_main_code")
        
        if method is None:
            return [-200], {"Compile Error while running the testcases."}

        signal.alarm(timeout)
        faulthandler.enable()
        try:
            # can lock here so time is useful
            start = time.time()
            prediction = method()
            total_execution += time.time() - start
            signal.alarm(0)

            # don't penalize model if it produces tuples instead of lists
            # ground truth sequences are not tuples
            if isinstance(prediction, tuple):
                prediction = list(prediction)

            tmp_result = prediction == gt_out

            try:
                tmp_result = tmp_result or (json.dumps(prediction) == json.dumps(gt_out))
            except Exception:
                pass

            try:
                tmp_result = tmp_result or (np.allclose(float(prediction), float(gt_out), rtol=1e-5, atol=1e-6))
            except Exception:
                pass

            if isinstance(prediction, list):
                try:
                    if len(prediction) == len(gt_out):
                        all_matched = True
                        for i in range(len(prediction)):
                            if np.allclose(float(prediction[i]), float(gt_out[i]), rtol=1e-5, atol=1e-6) == False:
                                all_matched = False
                                break
                        tmp_result = tmp_result or all_matched
                except Exception:
                    pass
                    

            # handle floating point comparisons

            all_results.append(tmp_result)

            if not tmp_result:
                return all_results, {
                    "inputs": truncatefn(gt_inp),
                    "output": truncatefn(prediction),
                    "expected": truncatefn(gt_out),
                    "error_code": -2,
                    "error_message": "Wrong Answer",
                }
        except Exception as e:
            signal.alarm(0)
            if "timeoutexception" in repr(e).lower():
                all_results.append(-3)
                return all_results, {
                    "error": repr(e),
                    "error_code": -3,
                    "error_message": "Time Limit Exceeded",
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                }
            else:
                if "NameError" in repr(e):
                    all_results.append(-200)
                else:
                    all_results.append(-300)
                return all_results, {
                    "error": repr(e),
                    "error_code": -4,
                    "error_message": "Runtime Error",
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                }

        finally:
            signal.alarm(0)
            faulthandler.disable()

    return all_results, {"execution time": total_execution}


def grade_stdio(
    code: str,
    all_inputs: list,
    all_outputs: list,
    timeout: int,
):
    ## runtime doesn't interact well with __name__ == '__main__'
    code = clean_if_name(code)

    ## we wrap the given code inside another function
    code = make_function(code)
   
    #time.sleep(6)
    compiled_sol = compile_code(code, timeout)
        
    if compiled_sol is None:
        return [-200], {"Compile Error while running the testcases."}  

    method = get_function(compiled_sol, "wrapped_function")

    if method is None:
        return [-200], {"Compile Error while running the testcases."}  

    all_results = []
    total_execution_time = 0
    for idx, (gt_inp, gt_out) in enumerate(zip(all_inputs, all_outputs)):
        signal.alarm(timeout)
        faulthandler.enable()

        signal.alarm(timeout)
        with Capturing() as captured_output:
            try:
                start = time.time()
                call_method(method, gt_inp)
                total_execution_time += time.time() - start
                # reset the alarm
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                if "timeoutexception" in repr(e).lower():
                    all_results.append(-3)
                    return all_results, {
                        "error": repr(e),
                        "error_code": -3,
                        "error_message": "Time Limit Exceeded",
                        "inputs": truncatefn(gt_inp),
                        "expected": truncatefn(gt_out),
                    }
                else:
                    all_results.append(-300)
                    return all_results, {
                        "error": repr(e),
                        "error_code": -4,
                        "error_message": "Runtime Error",
                        "inputs": truncatefn(gt_inp),
                        "expected": truncatefn(gt_out),
                    }

            finally:
                signal.alarm(0)
                faulthandler.disable()

        prediction = captured_output[0]

        stripped_prediction_lines = get_stripped_lines(prediction)
        stripped_gt_out_lines = get_stripped_lines(gt_out)

        ## WA happens in multiple circumstances
        ## so cache the return to make it clean!
        WA_send_args = {
            "inputs": truncatefn(gt_inp),
            "output": truncatefn(prediction),
            "expected": truncatefn(gt_out),
            "error_code": -2,
        }

        if len(stripped_prediction_lines) != len(stripped_gt_out_lines):
            all_results.append(-2)
            WA_send_args["error_message"] = "Wrong answer: mismatched output length"
            return all_results, WA_send_args

        for output_line_idx, (
            stripped_prediction_line,
            stripped_gt_out_line,
        ) in enumerate(zip(stripped_prediction_lines, stripped_gt_out_lines)):
            WA_send_args["error_message"] = (
                f"Wrong answer at {output_line_idx=}: {truncatefn(stripped_prediction_line)} != {truncatefn(stripped_gt_out_line)}"
            )
            ## CASE 1: exact match
            if stripped_prediction_line == stripped_gt_out_line:
                continue
            
            try:
                if stripped_prediction_line.lower() == "yes" and stripped_prediction_line.lower() == stripped_gt_out_line.lower():
                    continue
                if stripped_prediction_line.lower() == "no" and stripped_prediction_line.lower() == stripped_gt_out_line.lower():
                    continue

            except Exception:
                pass
            ## CASE 2: element-wise comparision
            ## if there are floating elements
            ## use `decimal` library for good floating point comparision
            ## otherwise gotcha: np.isclose(50000000000000000, 50000000000000001) = True
            ## note that we should always be able to convert to decimals

            success, decimal_prediction_line = convert_line_to_decimals(
                stripped_prediction_line
            )

            if not success:
                all_results.append(-2)
                return all_results, WA_send_args

            success, decimal_gtout_line = convert_line_to_decimals(stripped_gt_out_line)
            
            if not success:
                all_results.append(-2)
                return all_results, WA_send_args
            
            if decimal_prediction_line == decimal_gtout_line:
                continue
            
            ok = False
            try:
                if len(decimal_prediction_line) == len(decimal_gtout_line) and np.allclose(decimal_prediction_line, decimal_gtout_line, rtol=1e-5, atol=1e-6):
                    ok = True
            except Exception:
                pass

            if ok: 
                continue
            #print(ok)
            all_results.append(-2)
            return all_results, WA_send_args
        all_results.append(True)

    return all_results, {"execution time": total_execution_time}


def run_test(problem_to_check, debug=False, timeout=6):
    """
    if test(generated_code) is not None it'll try to run the code.
    otherwise it'll just return an input and output pair.
    """
    signal.signal(signal.SIGALRM, timeout_handler)
    # Disable functionalities that can make destructive changes to the test.
    # max memory is set to 4GB
    reliability_guard()

    generation = problem_to_check['generation']
    generation = generation.replace("sys.stdin.buffer.read()", "sys.stdin.read()") # in case model uses buffer read which we don't support
    generation = has_code(generation.split("</think>")[-1])

    if len(generation) >= 1 and generation[0] == 'Your code\n':
        generation = generation[1:]
   
    if len(generation) == 0:
        return [-100], "No code found or code is incomplete."
    
    if debug:
        print(f"start = {datetime.now().time()}")

    if problem_to_check['starter_code'] == '':
        which_type = CODE_TYPE.standard_input  # Standard input
        method_name = None
        generation = generation[-1]
    else:
        starter = problem_to_check['starter_code']
        which_type = CODE_TYPE.call_based  # Call-based
        method_name = starter
        generation = generation[-1]
        #try:
        #    generation = "\n".join(extract_functions(generation[-1]))
        #except:
        #    return [-200], "Compile Error"

    if 'lib' in problem_to_check:
        generation += '\n' + problem_to_check['lib']
    generation = post_process_code(generation).replace("\t", "    ").replace('if __name__ == "__main__":', 'if True:').replace("if __name__ == '__main__':", 'if True:')

    inputs, outputs = [], []
    if isinstance(problem_to_check['input_output'], dict):
        # check inputs and outputs field
        inputs = problem_to_check['input_output']['inputs']
        outputs = problem_to_check['input_output']['outputs']
        # check if they are file based or plain-text based
        for input_idx in range(len(inputs)):
            if os.path.exists(inputs[input_idx]): # File based
                with open(inputs[input_idx], 'r', encoding='utf-8') as f:
                    inputs[input_idx] = f.read().strip()
            if os.path.exists(outputs[input_idx]): # File based
                with open(outputs[input_idx], 'r', encoding='utf-8') as f:
                    outputs[input_idx] = f.read().strip()
    elif isinstance(problem_to_check['input_output'], list):
        for in_out in problem_to_check['input_output']:
            if False: #os.path.exists(in_out['input']): # File based
                with open(in_out['input'], 'r', encoding='utf-8') as f:
                    inputs.append(f.read().strip())
                with open(in_out['output'], 'r', encoding='utf-8') as f:
                    outputs.append(f.read().strip())
            else: # plain-text based
                inputs.append(in_out['input'])
                outputs.append(in_out['output'])
    else:
        return [-100], "Invalid input-output format."

    assert(len(inputs) == len(outputs))
    
    if debug:
        print(f"loaded {len(inputs)} Input-Output pairs = {datetime.now().time()}")

    results = []
    if debug:
        print(f"loading test code = {datetime.now().time()}")
    total_timeout = 1000000 #timeout * len(inputs) + 5
    if which_type == CODE_TYPE.call_based:
        signal.alarm(total_timeout)
        try:
            if "message" in problem_to_check['question_id'] and 'lib' in problem_to_check:
                results, metadata = grade_ioi(
                    code=generation,
                    all_inputs=inputs,
                    all_outputs=outputs,
                    fn_name=method_name,
                    timeout=timeout,
                )
            elif "worldmap" in problem_to_check['question_id'] and 'lib' in problem_to_check:
                results, metadata = grade_ioi(
                    code=generation,
                    all_inputs=inputs,
                    all_outputs=outputs,
                    fn_name=method_name,
                    timeout=timeout,
                )
            elif "triples-part2" in problem_to_check['question_id'] and 'lib' in problem_to_check:
                results, metadata = grade_ioi(
                    code=generation,
                    all_inputs=inputs,
                    all_outputs=outputs,
                    fn_name=method_name,
                    timeout=timeout,
                )
            elif "migrations" in problem_to_check['question_id'] and 'lib' in problem_to_check:
                results, metadata = grade_ioi(
                    code=generation,
                    all_inputs=inputs,
                    all_outputs=outputs,
                    fn_name=method_name,
                    timeout=timeout,
                )
            else:
                results, metadata = grade_call_based(
                    code=generation,
                    all_inputs=inputs,
                    all_outputs=outputs,
                    fn_name=method_name,
                    timeout=timeout,
                )
            return results, metadata
        except Exception as e:
            # import traceback
            # with open("./test_error")
            return [-4], f"Call-based Error as {e}" 
        finally:
            signal.alarm(0)

    elif which_type == CODE_TYPE.standard_input:
        signal.alarm(total_timeout)
        try:
        #if True:
            results, metadata = grade_stdio(
                code=generation,
                all_inputs=inputs,
                all_outputs=outputs,
                timeout=timeout,
            )
            return results, metadata
        except Exception as e:
            return [-4], f"Stdin-based Error as {e}"
        finally:
            signal.alarm(0)


def reliability_guard(maximum_memory_bytes=None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)
    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(
            resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
        )
        if not platform.uname().system == "Darwin":
            resource.setrlimit(
                resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes)
            )

    faulthandler.disable()

    import builtins

    # builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None

