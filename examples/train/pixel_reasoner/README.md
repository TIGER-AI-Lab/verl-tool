## Pixel Reasoner

### Data Preprocessing
```bash
python examples/data_preprocess/pixel_reasoner.py --dataset_path=TIGER-Lab/PixelReasoner-RL-Data --local_dir=data/pixel_reasoner --version max_8192 --include_videos=True --filter_len=8192
```
note the data processing will filter out those samples with length larger than 8192 and will take a while (0.5 to 1 hour) to finish. If you don't want to filter, remove the `--filter_len` argument. But there are some samples with length larger than 8192, which may cause problems in training, so make sure you set the `max_prompt_length` during the training properly.

### Training
```bash
bash examples/train/pixel_reasoner/train_qwen25vl.sh
```
It should be able to run under 8 GPUs with 80GB memory. 

### Notes
- If you kill a training job, for current version of verl, it seems the gpu memory will not be released immediately. You may need to kill mannually (e.g. `pkill -f -9 ray`)
- Check original paper for more details: [Pixel Reasoner](https://arxiv.org/abs/2505.15966)
- [PR](https://github.com/TIGER-AI-Lab/verl-tool/pull/63)