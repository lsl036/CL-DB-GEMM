#!/bin/bash

# 启动程序并保存 PID
nohup ./clblast_sample_sgemm_c > output.log 2>&1 &
PID=$!
echo "Program started with PID: $PID"

# 使用 pcm 捕捉硬件信息
pcm -pid $PID
