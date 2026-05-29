from typing import List
import os
import re
import logging
from tenacity import retry, stop_after_attempt, wait_random_exponential
import requests
import sys

from .openai import OpenAIChat, END_OF_MESSAGE, STOP

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 256
DEFAULT_MODEL = "qwen-plus"

STOP = ("<EOS>", "[EOS]", "(EOS)")
END_OF_MESSAGE = "<EOS>"


class QwenChat(OpenAIChat):
    stateful = False
    type_name = "qwen-chat"
    log_prefix = "QWEN"
    MODEL_ALIASES = {
        # 通用模型
        "qwen-plus": "qwen-plus",
        "qwen-max": "qwen-max",
        "qwen-turbo": "qwen-turbo",
        "qwen-turbo-latest": "qwen-turbo-latest",
        
        # 1.8B 模型 - 正确的名称是 qwen1.5-1.8b-chat (注意没有短横线)
        "qwen-1.8b": "qwen1.5-1.8b-chat",
        "qwen-1.8b-chat": "qwen1.5-1.8b-chat",
        "qwen1.8b": "qwen1.5-1.8b-chat",
        "qwen1.8b-chat": "qwen1.5-1.8b-chat",
        "qwen1.5-1.8b": "qwen1.5-1.8b-chat",
        "qwen1.5-1.8b-chat": "qwen1.5-1.8b-chat",
        # Qwen2 变体
        "qwen2-1.5b": "qwen2-1.5b-instruct",
        "qwen2-1.5b-instruct": "qwen2-1.5b-instruct",
        
        # 72B 模型 - 正确的名称是 qwen1.5-72b-chat (注意没有短横线)
        "qwen-72b": "qwen1.5-72b-chat",
        "qwen-72b-chat": "qwen1.5-72b-chat",
        "qwen72b": "qwen1.5-72b-chat",
        "qwen72b-chat": "qwen1.5-72b-chat",
        "qwen1.5-72b": "qwen1.5-72b-chat",
        "qwen1.5-72b-chat": "qwen1.5-72b-chat",
        # Qwen2 变体
        "qwen2-72b": "qwen2-72b-instruct",
        "qwen2-72b-instruct": "qwen2-72b-instruct",
        
        # 其他模型（可能不支持，但保留别名）
        "qwen-3b": "qwen-3b",
        "qwen3b": "qwen-3b",
        "qwen-7b": "qwen-7b",
        "qwen7b": "qwen-7b",
        "qwen-14b": "qwen-14b",
        "qwen14b": "qwen-14b",
        "qwen-32b": "qwen-32b",
        "qwen32b": "qwen-32b",
    }

    def __init__(self, args, temperature: float = DEFAULT_TEMPERATURE, max_tokens: int = DEFAULT_MAX_TOKENS,
                 model: str = DEFAULT_MODEL, **kwargs):
        super().__init__(args, temperature=temperature, max_tokens=max_tokens, model=model, **kwargs)
        self.model = self._resolve_model_name(model)
        self.temperature = args.temperature if args else temperature
        self.max_tokens = args.max_tokens if args else max_tokens
        self.api_key = (
            os.getenv("QWEN_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
            or getattr(args, "qwen_api_key", None)
            # 备选：使用 Gemini API Key（如果两个模型使用相同的代理）
            or os.getenv("GEMINI_API_KEY")
        )
        if not self.api_key:
            logging.warning("QWEN_API_KEY/DASHSCOPE_API_KEY 环境变量未设置，QwenChat将无法调用接口")
        
        # 调试信息
        if self.api_key:
            print(f"[DEBUG][QWEN] Using API Key: {self.api_key[:20]}...", file=sys.stderr)
        print(f"[DEBUG][QWEN] Model: {self.model}", file=sys.stderr)

        custom_url = os.getenv("QWEN_API_URL") or os.getenv("QWEN_API_BASE")
        self.api_candidates = self._build_api_candidates(custom_url)

    def _resolve_model_name(self, model_name: str) -> str:
        if not model_name:
            return DEFAULT_MODEL
        normalized = model_name.lower().strip()
        resolved = self.MODEL_ALIASES.get(normalized)
        if resolved:
            return resolved
        normalized_no_suffix = re.sub(r"-(instruct|chat|sft)$", "", normalized)
        return self.MODEL_ALIASES.get(normalized_no_suffix, model_name)

    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=2, max=16))
    def _get_response(self, messages, method=None, max_tokens=None, T=None, speaker="Unknown"):
        max_toks = self.max_tokens if max_tokens is None else max_tokens
        temperature = self.temperature if T is None else T

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_toks,
            "stop": STOP
        }

        last_error = None
        for api_url in self.api_candidates:
            try:
                print(f"[DEBUG][QWEN] Trying API URL: {api_url}", file=sys.stderr)
                resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                choice0 = (data.get('choices') or [{}])[0]
                message_obj = choice0.get('message') or {}
                content = (
                    message_obj.get('content')
                    or choice0.get('text')
                    or data.get('output_text')
                    or ""
                ).strip()
                # 如果是 DashScope 的直接生成端点，响应格式可能不同
                if not content and 'output' in data:
                    output = data.get('output', {})
                    content = output.get('text', "")
                print(f"[DEBUG][QWEN] Success with URL: {api_url}", file=sys.stderr)
                break
            except requests.HTTPError as e:
                last_error = e
                if e.response is not None:
                    status_code = e.response.status_code
                    print(f"[DEBUG][QWEN] URL {api_url} failed with status {status_code}", file=sys.stderr)
                    if status_code == 404:
                        continue  # 尝试下一个端点
                    print(f"[QwenChat ERROR] {e}", file=sys.stderr)
                    print(f"[QwenChat DEBUG] {e.response.text}", file=sys.stderr)
                raise
            except Exception as e:
                last_error = e
                print(f"[QwenChat ERROR] {e}", file=sys.stderr)
                print(f"[DEBUG][QWEN] URL {api_url} failed with exception", file=sys.stderr)
                # 如果不是 404，继续尝试下一个端点
                if isinstance(e, requests.HTTPError) and e.response and e.response.status_code == 404:
                    continue
                raise
        else:
            if isinstance(last_error, requests.HTTPError) and last_error.response is not None:
                print(f"[QwenChat DEBUG] {last_error.response.text}", file=sys.stderr)
                last_error.response.raise_for_status()
            elif isinstance(last_error, Exception):
                raise last_error
            else:
                raise RuntimeError("QwenChat: no available API endpoint responded")

        if content.endswith("<EOS"):
            content = content + ">"
        if not content.endswith(END_OF_MESSAGE):
            content = content + END_OF_MESSAGE

        print("Model reply:", content, file=sys.stderr)
        self._log_model_reply(speaker, content)
        return content

    def _build_api_candidates(self, custom_url):
        if custom_url:
            return [custom_url.rstrip("/")]

        # 尝试多个可能的 API 端点
        candidates = [
            # 标准 DashScope 端点
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            # DashScope 直接生成端点
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            # 备选：使用统一的 api2.aigcbest.top（如果支持 Qwen）
            "https://api2.aigcbest.top/v1/chat/completions",
        ]
        
        # 如果设置了环境变量，优先使用
        alt_url = os.getenv("QWEN_API_URL") or os.getenv("QWEN_API_BASE")
        if alt_url and alt_url not in candidates:
            candidates.insert(0, alt_url.rstrip("/"))
        
        return candidates