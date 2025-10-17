#!/usr/bin/env python
import json
import requests
import fire
import logging
import sys
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_python(
    url: str = None,
    trajectory_id: str = "test-python-001",
):
    """Test stateful IPython code execution"""
    
    print("--- Testing 1: Initial variable definition ---")
    action = """<python>x = 10\ny = 20\nprint(f'Defined x={x}, y={y}')</python> ..."""
    print(_send_test_request(url, trajectory_id, action, "Variable Definition"))
    
    print("--- Testing 2: Using previously defined variables ---")
    action = """<python>z = x + y\nprint(f'x={x}, y={y}, z={z}')</python> ..."""
    print(_send_test_request(url, trajectory_id, action, "State Persistence"))
    
    print("--- Testing 3: Import persistence ---")
    action = """<python>import numpy as np\narr = np.array([1, 2, 3])\nprint(f'Created numpy array: {arr}')</python> ..."""
    print(_send_test_request(url, trajectory_id, action, "Import Library"))
    
    print("--- Testing 4: Using imported library and previous variables ---")
    action = """<python>scaled_arr = arr * z\nprint(f'Original array: {arr}')\nprint(f'Scaled by z={z}: {scaled_arr}')</python> ..."""
    print(_send_test_request(url, trajectory_id, action, "Combined State"))
    
    print("--- Testing 5: Function definition ---")
    action = """<python>def multiply(a, b):\n    return a * b\n\nresult = multiply(x, y)\nprint(f'Function multiply({x}, {y}) = {result}')</python> ..."""
    print(_send_test_request(url, trajectory_id, action, "Function Definition"))
    
    print("--- Testing 6: Using previously defined function ---")
    action = """<python>new_result = multiply(z, 2)\nprint(f'Using saved function: multiply({z}, 2) = {new_result}')</python> ..."""
    print(_send_test_request(url, trajectory_id, action, "Function Reuse"))

    print("--- Testing 7: Class definition and instantiation ---")
    action = """<python>class Counter:\n    def __init__(self):\n        self.count = 0\n    def increment(self):\n        self.count += 1\n        return self.count\n\ncounter = Counter()\nprint(f'Counter initialized with count: {counter.count}')</python> ..."""
    print(_send_test_request(url, trajectory_id, action, "Class Definition"))

    print("--- Testing 8: Using class instance across cells ---")
    action = """<python>for i in range(3):\n    counter.increment()\nprint(f'Counter after 3 increments: {counter.count}')\nprint(f'All variables still accessible: x={x}, arr={arr}')</python> ..."""
    print(_send_test_request(url, trajectory_id, action, "Stateful Object"))
    
    # Test with a different trajectory_id to show isolation
    print("\n--- Testing 9: Different trajectory (should fail to access previous variables) ---")
    different_trajectory = "test-python-002"
    action = """<python>try:\n    print(f'Trying to access x from different session: {x}')\nexcept NameError as e:\n    print(f'Error as expected: {e}')</python> ..."""
    print(_send_test_request(url, different_trajectory, action, "Session Isolation"))
    
    # Back to original trajectory
    print("--- Testing 10: Original session still has state ---")
    action = """<python>print(f'Back to original session:')\nprint(f'x={x}, y={y}, z={z}')\nprint(f'counter.count={counter.count}')\nprint(f'multiply function exists: {multiply.__name__}')</python> ..."""
    print(_send_test_request(url, trajectory_id, action, "State Verification"))
    
    # Test with a completely new trajectory to show fresh state
    print("\n--- Testing 11: New trajectory starts fresh (no previous state) ---")
    new_trajectory = "test-python-003"
    action = """<python># First, try to access old variables (should fail)\ntry:\n    print(f'Trying to access x: {x}')\nexcept NameError:\n    print('NameError: x is not defined (as expected)')\n\n# Now define new variables in this fresh session\na = 100\nb = 200\nprint(f'New session variables: a={a}, b={b}')</python> ..."""
    print(_send_test_request(url, new_trajectory, action, "Fresh Session"))
    
    print("--- Testing 12: New trajectory maintains its own state ---")
    action = """<python>c = a + b\nprint(f'In new session: a={a}, b={b}, c={c}')\n# Verify old session variables are still not accessible\ntry:\n    print(f'Old x: {x}')\nexcept NameError:\n    print('Original session variables still not accessible (correct isolation)')</python> ..."""
    print(_send_test_request(url, new_trajectory, action, "New Session State"))
    
    return True
    
    
def _send_test_request(url, trajectory_id, action, test_name):
    """Helper function to send test requests and process responses"""
    logger.info(f"Testing {test_name} code execution...")
    
    # Use server API
    payload = {
        "trajectory_ids": [trajectory_id],
        "actions": [action],
        "extra_fields": [{}]
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

def main():
    """Main entry point for the test script
    Run with:
        python -m verl_tool.servers.tests.test_python_code_tool python --url=http://localhost:5000/get_observation
    """
    fire.Fire({
        "python": test_python,
    })

if __name__ == "__main__":
    main()