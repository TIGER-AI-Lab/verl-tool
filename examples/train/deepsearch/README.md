### Data Preparation
```bash
python examples/data_preprocess/deepsearch.py --dataset_path=VerlTool/deepsearch
```

### Test the Tool Server
```bash
# Start the tool server
host=localhost
port=5000
tool_type=google_search # separate by comma if you want to start multiple tool servers
workers_per_tool=4 # number of workers for the tool server, meaning how many threads will be used to handle a single tool request with multiple trajectories
python -m verl_tool.servers.serve --host $host --port $port --tool_type $tool_type --workers_per_tool $workers_per_tool & # run in background
```
```bash
python -m verl_tool.servers.tests.test_bing_search_tool bing_search --url=http://localhost:5000/get_observation
```
### Training
```bash
bash examples/train/deepsearch/train.sh > logs/deepsearch_3b_debug.log 2>&1 &
```
