#!/usr/bin/env python
"""Test cases for the firejail sandbox environment.

Test coverage:
1. Execution Test - Basic command execution in sandbox environment
2. Return Result Test - STDOUT, STDERR
2. Timeout Test - Handling of long-running process termination
3. Modules Test - Verification of essential math package availability, e.g. numpy, pandas, sympy, scipy, etc.
4. Multiprocess Press Test - Stability under concurrent process execution
"""
import json
import requests
import fire
import logging
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from tqdm import tqdm

# Add parent directory to path to import FirejailPythonCodeTool
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ..tools.firejail_python_code import FirejailPythonCodeTool

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_firejail_python(
    url: str = None,
    trajectory_id: str = "test-firejail-python-001",
):
    """Test Firejail Python code execution with various test cases"""
    
    print("--- Test 1: Basic Hello World ---")
    action = """<python>print('Hello from Firejail Python!')</python>"""
    print(_send_test_request(url, trajectory_id, action, "Hello World"))
    
    print("--- Test 2: Multiple Print Statements ---")
    action = """```python
print('Line 1')
print('Line 2')
print('Line 3')
```"""
    print(_send_test_request(url, trajectory_id, action, "Multiple Prints"))
    
    print("--- Test 3: Basic Computation ---")
    action = """<python>
# Calculate sum of first 10 numbers
total = sum(range(1, 11))
print(f"Sum of numbers 1-10: {total}")

# Calculate factorial of 5
factorial = 1
for i in range(1, 6):
    factorial *= i
print(f"Factorial of 5: {factorial}")
</python>"""
    print(_send_test_request(url, trajectory_id, action, "Basic Computation"))
    
    print("--- Test 4: List Comprehension ---")
    action = """```python
# Generate squares using list comprehension
squares = [x**2 for x in range(1, 11)]
print("Squares of numbers 1-10:")
for num, square in enumerate(squares, 1):
    print(f"{num}² = {square}")
```"""
    print(_send_test_request(url, trajectory_id, action, "List Comprehension"))
    
    print("--- Test 5: Working with Input ---")
    action = """<python>
name = input("What's your name? ")
print(f"Hello, {name}!")

age = input("How old are you? ")
print(f"In 10 years, you'll be {int(age) + 10} years old.")
</python>"""
    extra_field = {"stdin": "Alice\n25\n"}
    print(_send_test_request(url, trajectory_id, action, "Input Handling", extra_field))
    
    print("--- Test 6: Error Handling ---")
    action = """<python>
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Caught division by zero error")
    
print("Execution continues after error handling")
</python>"""
    print(_send_test_request(url, trajectory_id, action, "Error Handling"))
    
    print("--- Test 7: Syntax Error ---")
    action = """<python>
prnit("This has a typo and will fail")
</python>"""
    print(_send_test_request(url, trajectory_id, action, "Syntax Error"))
    
    print("--- Test 8: Timeout Test ---")
    action = """<python>
import time
print("Starting a long computation...")
time.sleep(10)  # Should exceed the timeout
print("This shouldn't be printed")
</python>"""
    print(_send_test_request(url, trajectory_id, action, "Timeout Test"))
    
    print("--- Test 9: Forbidden Import Test ---")
    action = """<python>
import os
print("Current directory files:")
os.system("ls -la")  # This should be blocked
</python>"""
    print(_send_test_request(url, trajectory_id, action, "Forbidden Import"))
    
    print("--- Test 10: Resource Limitation Test ---")
    action = """<python>
# Try to allocate a large amount of memory
big_list = list(range(100000000))  # Should be limited by resource constraints
print(f"Created a list with {len(big_list)} elements")
</python>"""
    print(_send_test_request(url, trajectory_id, action, "Resource Limitation"))
    
    print("--- Test 11: Class Definition ---")
    action = """```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def greet(self):
        return f"Hello, my name is {self.name} and I'm {self.age} years old."

# Create instances
alice = Person("Alice", 30)
bob = Person("Bob", 25)

# Test the class
print(alice.greet())
print(bob.greet())
```"""
    print(_send_test_request(url, trajectory_id, action, "Class Definition"))
    
    print("--- Test 12: File Writing Attempt ---")
    action = """<python>
try:
    with open('test.txt', 'w') as f:
        f.write('This is a test')
    print("File written successfully")
except Exception as e:
    print(f"Error writing file: {e}")
</python>"""
    print(_send_test_request(url, trajectory_id, action, "File Writing"))
    
    return True

def test_timeout():
    """Test the timeout mechanism of the code execution environment"""
    print("\n[3] Timeout Test")
    
    tool = FirejailPythonCodeTool()
    
    start = time.time()
    action = """<python>
while True: pass
</python>"""
    
    observation, _, _ = tool.conduct_action("timeout-test", action, {})
    
    duration = time.time() - start
    print("Duration:", round(duration, 2), "seconds")
    print("Output:", observation)
    
    return {
        "duration": duration,
        "observation": observation
    }

def test_modules():
    """Test importing essential scientific and data analysis modules"""
    print("\n[4] Essential Modules Test")
    
    tool = FirejailPythonCodeTool()
    tests = {
        "numpy": "import numpy as np; print('numpy imported')",
        "pandas": "import pandas as pd; print('pandas imported')",
        "sympy": "import sympy; print('sympy imported')",
        "scipy": "import scipy; print('scipy imported')",
        "sklearn": "import sklearn; print('sklearn imported')"
    }
    
    results = {}
    for name, code in tests.items():
        print(f"  - {name}:")
        action = f"<python>{code}</python>"
        observation, _, _ = tool.conduct_action(f"module-test-{name}", action, {})
        print("    Output:", observation)
        results[name] = observation
    
    return results

def test_multiprocess():
    """Test running multiple code executions in parallel to stress test the system"""
    print("\n[5] Multiprocess Stress Test")
    
    tool = FirejailPythonCodeTool()
    codes = ["import numpy as np\nprint(np.ones((1024, 1024)).sum())"] * 1024
    results = []
    
    with ThreadPoolExecutor(max_workers=max(cpu_count() // 2, 1)) as pool:
        futures = []
        for i, code in enumerate(codes):
            action = f"<python>{code}</python>"
            futures.append(pool.submit(
                tool.conduct_action, 
                f"multiprocess-test-{i}", 
                action, 
                {}
            ))
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            observation, done, valid = future.result()
            assert "ERROR" not in observation and "Exception" not in observation, observation
            results.append({
                "observation": observation,
                "done": done,
                "valid": valid
            })
    
    print(f"Successfully completed {len(results)} parallel executions")
    return results
    
def _send_test_request(url, trajectory_id, action, test_name, extra_field=None):
    """Helper function to send test requests and process responses"""
    logger.info(f"Testing {test_name} code execution...")
    
    if extra_field is None:
        extra_field = {}
    
    # Use server API
    payload = {
        "trajectory_ids": [trajectory_id],
        "actions": [action],
        "extra_fields": [extra_field]
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise exception for error status codes
        
        result = response.json()
        logger.info(f"Response received for {test_name} test")
        
        # Print observation
        if "observations" in result and len(result["observations"]) > 0:
            observation = result["observations"][0]
            logger.info(f"\n--- {test_name} Result ---\n{observation}\n")
        else:
            logger.error(f"No observation found in response for {test_name}")
        
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"error": str(e)}

def test_firejail_python_direct():
    """Run direct tests on the FirejailPythonCodeTool without using the server API"""
    logger.info("Testing FirejailPythonCodeTool directly...")
    
    tool = FirejailPythonCodeTool()
    test_cases = [
        {
            "name": "Simple Print",
            "action": "<python>print('Direct test')</python>",
            "extra_field": {}
        },
        {
            "name": "Basic Math",
            "action": "```python\nprint(2 + 2)\n```",
            "extra_field": {}
        },
        {
            "name": "With Input",
            "action": "<python>name = input('Name? '); print(f'Hello {name}')</python>",
            "extra_field": {"stdin": "World"}
        }
    ]
    
    results = {}
    for test_case in test_cases:
        logger.info(f"Running direct test: {test_case['name']}")
        observation, done, valid = tool.conduct_action(
            "direct-test", 
            test_case['action'], 
            test_case['extra_field']
        )
        results[test_case['name']] = {
            "observation": observation,
            "done": done,
            "valid": valid
        }
        logger.info(f"\n--- {test_case['name']} Direct Result ---\n{observation}\n")
    
    return results

def test_firejail_with_matplotlib():
    """Test Firejail Python with matplotlib plotting"""
    action = """<python>
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Generate some data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    # Create a plot
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label='sin(x)')
    plt.title('Sine Function')
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.legend()
    
    # Try to save the plot
    try:
        plt.savefig('sine.png')
        print("Plot saved successfully")
    except Exception as e:
        print(f"Error saving plot: {e}")
        
    print("Matplotlib test completed")
except ImportError:
    print("Matplotlib not available in this environment")
</python>"""
    
    tool = FirejailPythonCodeTool()
    observation, done, valid = tool.conduct_action("matplotlib-test", action, {})
    logger.info(f"\n--- Matplotlib Test Result ---\n{observation}\n")
    
    return {
        "observation": observation,
        "done": done,
        "valid": valid
    }

def main():
    """Main entry point for the test script
    Run with:
        python -m verl_tool.servers.tests.test_firejail_python_code_tool firejail --url=http://localhost:5000/get_observation
        python -m verl_tool.servers.tests.test_firejail_python_code_tool direct
        python -m verl_tool.servers.tests.test_firejail_python_code_tool matplotlib
        python -m verl_tool.servers.tests.test_firejail_python_code_tool timeout
        python -m verl_tool.servers.tests.test_firejail_python_code_tool modules
        python -m verl_tool.servers.tests.test_firejail_python_code_tool multiprocess
    """
    fire.Fire({
        "firejail": test_firejail_python,
        "direct": test_firejail_python_direct,
        "matplotlib": test_firejail_with_matplotlib,
        "timeout": test_timeout,
        "modules": test_modules,
        "multiprocess": test_multiprocess,
    })

if __name__ == "__main__":
    main()