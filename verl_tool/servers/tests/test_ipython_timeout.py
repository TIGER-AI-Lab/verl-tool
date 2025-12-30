#!/usr/bin/env python
import json
import requests
import fire
import logging
import sys
import os
import uuid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_python(
    url: str = None,
    trajectory_id: str = "test-python-001",
):
    """Test Python code execution"""
    
    trajectory_id = str(uuid.uuid4())
    
    print("--- Testing 1 ---")
    action = """<python>import random, itertools, math, sys, time, collections, heapq, itertools, copy, math</python> ..."""
    print(_send_test_request(url, trajectory_id, action, "Python"))
    
    print("--- Testing 2 ---")
    action = """<python>def min_cost_flow(N, graph, s, t, maxf):
    # graph: adjacency list of edges: each edge is [to, rev, cap, cost]
    INF = 10**12
    res = 0
    potential = [0]*N
    dist = [0]*N
    prevv = [0]*N
    preve = [0]*N
    flow = 0
    while flow < maxf:
        # Dijkstra
        dist = [INF]*N
        dist[s] = 0
        hq = [(0, s)]
        while hq:
            d, v = heapq.heappop(hq)
            if dist[v] < d:
                continue
            for i, e in enumerate(graph[v]):
                to, rev, cap, cost = e
                if cap>0 and dist[to] > dist[v] + cost + potential[v] - potential[to]:
                    dist[to] = dist[v] + cost + potential[v] - potential[to]
                    prevv[to] = v
                    preve[to] = i
                    heapq.heappush(hq, (dist[to], to))
        if dist[t] == INF:
            # cannot flow more
            break
        for v in range(N):
            if dist[v] < INF:
                potential[v] += dist[v]
        # add as much as possible (capacity 1 per path)
        d = maxf - flow
        v = t
        while v != s:
            d = min(d, graph[prevv[v]][preve[v]][2])
            v = prevv[v]
        flow += d
        res += d * potential[t]
        v = t
        while v != s:
            e = graph[prevv[v]][preve[v]]
            e[2] -= d
            # rev edge
            graph[v][e[1]][2] += d
            v = prevv[v]
    return flow, res

def add_edge(graph, fr, to, cap, cost):
    graph[fr].append([to, len(graph[to]), cap, cost])
    graph[to].append([fr, len(graph[fr])-1, 0, -cost])

def solve_bruteforce(m,n,T):
    # enumerate all assignments of jobs to machines (m^n combos) small only for testing
    best = float('inf')
    best_assign=None
    for assign in itertools.product(range(m), repeat=n):
        # compute cost per machine
        total = 0
        per_machine = [[] for _ in range(m)]
        for j, i in enumerate(assign):
            per_machine[i].append(j)
        for i in range(m):
            jobs = per_machine[i]
            if not jobs:
                continue
            # sort processing times descending
            times = sorted([T[j][i] for j in jobs], reverse=True)
            # sum_{k} times[k-1] * k
            total += sum(times[k]* (k+1) for k in range(len(times))) # times index 0->largest weight 1
        if total < best:
            best = total
            best_assign = assign
    return best

def solve_mincost(m,n,T):
    # Build graph
    N = 1 + n + m*n + 1
    s = 0
    car_offset = 1
    slot_offset = 1+n
    t = N-1
    graph = [[] for _ in range(N)]
    # source to car nodes
    for j in range(n):
        add_edge(graph, s, car_offset+j, 1, 0)
    # slot nodes to sink
    for i in range(m):
        for k in range(1, n+1):
            slot_id = slot_offset + i*n + (k-1)
            add_edge(graph, slot_id, t, 1, 0)
    # car to slots
    for j in range(n):
        for i in range(m):
            base = slot_offset + i*n
            # p = T[j][i]
            p = T[j][i]
            # for each k
            # to reduce edges, might add only up to n but it's fine
            for k in range(1, n+1):
                slot_id = base + (k-1)
                cost = p * k
                add_edge(graph, car_offset+j, slot_id, 1, cost)
    flow, cost = min_cost_flow(N, graph, s, t, n)
    if flow != n:
        print("Flow insufficient")
    return cost

def test_random():
    for n in range(1,8):
        for m in range(2,4):
            for _ in range(200):
                T = [[random.randint(1,10) for _ in range(m)] for __ in range(n)]
                brute = solve_bruteforce(m,n,T)
                flow = solve_mincost(m,n,T)
                if brute != flow:
                    print("Mismatch! n,m", n,m)
                    print(T)
                    print(brute, flow)
                    return False
    print("All OK")
    return True

test_random()</python> ..."""
    print(_send_test_request(url, trajectory_id, action, "Python"))
    
    print("--- Testing 3 ---")
    action = """```python\ndef min_cost_flow(N, graph, s, t, maxf):
    INF = 10**12
    res = 0
    potential = [0]*N
    dist = [0]*N
    prevv = [0]*N
    preve = [0]*N
    flow = 0
    import heapq
    while flow < maxf:
        dist = [INF]*N
        dist[s] = 0
        hq = [(0, s)]
        while hq:
            d, v = heapq.heappop(hq)
            if dist[v] < d:
                continue
            for i, e in enumerate(graph[v]):
                to, rev, cap, cost = e
                if cap > 0 and dist[to] > dist[v] + cost + potential[v] - potential[to]:
                    dist[to] = dist[v] + cost + potential[v] - potential[to]
                    prevv[to] = v
                    preve[to] = i
                    heapq.heappush(hq, (dist[to], to))
        if dist[t] == INF:
            # cannot send more
            return flow, res
        for v in range(N):
            if dist[v] < INF:
                potential[v] += dist[v]
        # Add as much as possible (capacity 1 here)
        d = maxf - flow
        v = t
        while v != s:
            d = min(d, graph[prevv[v]][preve[v]][2])
            v = prevv[v]
        flow += d
        res += d * potential[t]
        v = t
        while v != s:
            e = graph[prevv[v]][preve[v]]
            e[2] -= d
            graph[v][e[1]][2] += d
            v = prevv[v]
    return flow, res

def add_edge(graph, fr, to, cap, cost):
    graph[fr].append([to, len(graph[to]), cap, cost])
    graph[to].append([fr, len(graph[fr])-1, 0, -cost])

def solve_inst(m,n,T):
    N = 1 + n + m*n + 1
    s = 0
    car_off = 1
    slot_off = 1 + n
    t = N - 1
    graph = [[] for _ in range(N)]
    # source to car
    for j in range(n):
        add_edge(graph, s, car_off+j, 1, 0)
    # slot to sink
    for i in range(m):
        base = slot_off + i*n
        for k in range(n):
            slot_id = base + k
            add_edge(graph, slot_id, t, 1, 0)
    # car to slots
    for j in range(n):
        for i in range(m):
            base = slot_off + i*n
            p = T[j][i]
            # precompute p times k+1 for all k
            # naive loop
            for k in range(n):
                slot_id = base + k
                cost = p * (k+1)
                add_edge(graph, car_off+j, slot_id, 1, cost)
    flow, total = min_cost_flow(N, graph, s, t, n)
    # flow should be n
    avg = total / n
    return avg

# test sample
sample_m, sample_n = 2,2
T = [[3,2],[1,4]]
print(solve_inst(sample_m,sample_n,T))``` ..."""
    print(_send_test_request(url, trajectory_id, action, "Python"))
    
    print("--- Testing 3.1 ---")
    action = """```python\ndef test_random2():
    for n in range(1,8):
        for m in range(2,5):
            for _ in range(200):
                T = [[random.randint(1,10) for _ in range(m)] for __ in range(n)]
                avg_flow = solve_inst(m,n,T)
                total_flow = avg_flow * n
                brute_total = solve_bruteforce(m,n,T)
                if total_flow != brute_total:
                    print("Mismatch", m,n,T, brute_total, total_flow)
                    return False
    print("All tests passed")
    return True

test_random2()\n``` ..."""
    print(_send_test_request(url, trajectory_id, action, "Python"))
    
    print("--- Testing 3.2 ---")
    action = """```python\ndef solve_inst_total(m,n,T):
    N = 1 + n + m*n + 1
    s = 0
    car_off = 1
    slot_off = 1 + n
    t = N - 1
    graph = [[] for _ in range(N)]
    for j in range(n):
        add_edge(graph,s,car_off+j,1,0)
    for i in range(m):
        base = slot_off + i*n
        for k in range(n):
            slot_id = base + k
            add_edge(graph,slot_id,t,1,0)
    for j in range(n):
        for i in range(m):
            base = slot_off + i*n
            p = T[j][i]
            for k in range(n):
                slot_id = base + k
                cost = p * (k+1)
                add_edge(graph,car_off+j,slot_id,1,cost)
    flow,total = min_cost_flow(N,graph,s,t,n)
    return total

def test_random3():
    for n in range(1,8):
        for m in range(2,5):
            for _ in range(200):
                T = [[random.randint(1,10) for _ in range(m)] for __ in range(n)]
                brute = solve_bruteforce(m,n,T)
                flow_total = solve_inst_total(m,n,T)
                if brute != flow_total:
                    print("Mismatch", m,n)
                    print(T)
                    print("brute", brute, "flow", flow_total)
                    return False
    print("All ok")
    return True

test_random3()\n``` ..."""
    print(_send_test_request(url, trajectory_id, action, "Python"))
    
    print("--- Testing 3.3 ---")
    action = """```python\ndef test_random3_one():
    for _ in range(20):
        n = random.randint(1,6)
        m = random.randint(2,4)
        T = [[random.randint(1,10) for _ in range(m)] for __ in range(n)]
        brute = solve_bruteforce(m,n,T)
        flow_total = solve_inst_total(m,n,T)
        if brute != flow_total:
            print("Mismatch")
            print(m,n,T)
            print(brute, flow_total)
            return False
    print("single set ok")
    return True

test_random3_one()\n``` ..."""
    print(_send_test_request(url, trajectory_id, action, "Python"))
    
    print("--- Testing 3.4 ---")
    action = """```python\ndef solve_inst_total(m,n,T):
    N = 1 + n + m*n + 1
    s = 0
    car_off = 1
    slot_off = 1 + n
    t = N - 1
    graph = [[] for _ in range(N)]
    for j in range(n):
        add_edge(graph,s,car_off+j,1,0)
    for i in range(m):
        base = slot_off + i*n
        for k in range(n):
            slot_id = base + k
            add_edge(graph,slot_id,t,1,0)
    for j in range(n):
        for i in range(m):
            base = slot_off + i*n
            p = T[j][i]
            for k in range(n):
                slot_id = base + k
                cost = p * (k+1)
                add_edge(graph,car_off+j,slot_id,1,cost)
    flow,total = min_cost_flow(N,graph,s,t,n)
    return total

def test_random3_one():
    for _ in range(200):
        n = random.randint(1,6)
        m = random.randint(2,4)
        T = [[random.randint(1,10) for _ in range(m)] for __ in range(n)]
        brute = solve_bruteforce(m,n,T)
        flow_total = solve_inst_total(m,n,T)
        if brute != flow_total:
            print("Mismatch")
            print("m,n:", m, n)
            print("T:", T)
            print("brute:", brute, "flow:", flow_total)
            return False
    print("All cases ok")
    return True

test_random3_one()\n``` ..."""
    print(_send_test_request(url, trajectory_id, action, "Python"))
    
    print("--- Testing 9 ---") # test finish
    action = ""
    print(_send_test_request(url, trajectory_id, action, "test_finish", finish=[True]))

    return True
    
    
def _send_test_request(url, trajectory_id, action, test_name, **kwargs):
    """Helper function to send test requests and process responses"""
    logger.info(f"Testing {test_name} code execution...")
    
    # Use server API
    payload = {
        "trajectory_ids": [trajectory_id],
        "actions": [action],
        "extra_fields": [{}],
        **kwargs
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
        python -m verl_tool.servers.tests.test_ipython_timeout python --url=http://localhost:5000/get_observation
    """
    fire.Fire({
        "python": test_python,
    })

if __name__ == "__main__":
    main()
