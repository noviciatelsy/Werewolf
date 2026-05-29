# server.py
import asyncio
import websockets
import argparse
import json
import os
import re
import sys
import subprocess
import time
from pathlib import Path

# 配置
DEFAULT_WS_HOST = "0.0.0.0"
DEFAULT_WS_PORT = 8765
DEFAULT_LOG_FILE = "debug_output.txt"  # run_werewolf.py 中的 log 文件名

clients = set()

# 简单的解析器：尝试匹配 "[Speaker]: text" 或 "Player X: text" 等常见格式
SPEAKER_REGEXES = [
    re.compile(r'^\[(?P<speaker>[^\]]+)\]:\s*(?P<text>.*)$'),   # [Player 1]: hello
    re.compile(r'^(?P<speaker>Player\s*\d+):\s*(?P<text>.*)$'), # Player 1: hello
    re.compile(r'^(?P<speaker>[A-Za-z0-9 _\-]+):\s*(?P<text>.+)$') # Generic: Author: text
]

async def ws_handler(ws, path):
    print(f"[ws] client connected: {ws.remote_address}")
    clients.add(ws)
    try:
        # Keep the connection alive until closed by client
        await ws.wait_closed()
    finally:
        clients.remove(ws)
        print(f"[ws] client disconnected: {ws.remote_address}")

async def broadcast_json(obj):
    if not clients:
        return
    s = json.dumps(obj, ensure_ascii=False)
    await asyncio.gather(*[c.send(s) for c in clients], return_exceptions=True)

def parse_line(line):
    line = line.strip()
    if not line:
        return None
    for rx in SPEAKER_REGEXES:
        m = rx.match(line)
        if m:
            speaker = m.groupdict().get("speaker")
            text = m.groupdict().get("text")
            return {"speaker": speaker, "text": text}
    # 无法解析到 speaker/text，返回 raw
    return {"raw": line}

async def tail_and_broadcast(log_path, poll_interval=0.1):
    """Tail file (like tail -f) 并把新行广播到所有 websocket 客户端"""
    # 等待文件生成
    p = Path(log_path)
    while not p.exists():
        print(f"[tail] waiting for {log_path} to appear...")
        await asyncio.sleep(0.5)

    print(f"[tail] start tailing {log_path}")
    # 以只读模式打开并 seek 到文件末尾
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        # move to EOF
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                await asyncio.sleep(poll_interval)
                continue
            payload = parse_line(line)
            if payload is None:
                continue
            # 添加 timestamp
            payload["_ts"] = time.time()
            # 打印到 server 控制台
            print(f"[tail->ws] {payload}")
            await broadcast_json(payload)

def launch_run_werewolf(script_path):
    """可选：由 server 启动 run_werewolf.py（非阻塞）"""
    print(f"[launcher] launching {script_path} ...")
    # 使用与当前 python 相同的解释器启动
    cmd = [sys.executable, script_path]
    # 这里不捕获 stdout/stderr（run_werewolf.py 自己写 log 文件）
    proc = subprocess.Popen(cmd, cwd=os.path.dirname(script_path) or ".")
    print(f"[launcher] pid = {proc.pid}")
    return proc

async def main_async(args):
    # 启动 websocket 服务
    print(f"[ws] starting websocket server on {args.host}:{args.port}")
    server = await websockets.serve(ws_handler, args.host, args.port, max_size=2**20)

    # 启动 run_werewolf.py（可选）
    proc = None
    if args.launch_script:
        if not Path(args.launch_script).exists():
            print(f"[error] launch script {args.launch_script} not found.")
        else:
            proc = launch_run_werewolf(args.launch_script)

    # 启动 tail task
    tail_task = asyncio.create_task(tail_and_broadcast(args.log_file, poll_interval=0.05))

    # 等待直到被终止
    try:
        await tail_task
    except asyncio.CancelledError:
        pass
    finally:
        server.close()
        await server.wait_closed()
        if proc:
            proc.terminate()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=DEFAULT_WS_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_WS_PORT)
    parser.add_argument("--log-file", type=str, default=DEFAULT_LOG_FILE,
                        help="path to the log file to tail (run_werewolf writes debug_output.txt)")
    parser.add_argument("--launch-script", type=str, default="",
                        help="optional: path to run_werewolf.py to launch it from server")
    args = parser.parse_args()

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("Exiting...")

if __name__ == "__main__":
    main()
