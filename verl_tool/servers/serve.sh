host=127.0.0.1
port=31400
tool_server_url=http://$host:$port/get_observation
python -m verl_tool.servers.serve --host $host --port $port --tool_type "crop_image" --workers_per_tool 8 &
server_pid=$!