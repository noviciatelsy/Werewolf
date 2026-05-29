import os
import sys

import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

from .openai import OpenAIChat, STOP, END_OF_MESSAGE

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 256
DEFAULT_MODEL = "gemini-1.5-flash"


class GeminiChat(OpenAIChat):
    """
    Gemini backend 支持两种 API 格式：
    1. OpenAI 兼容格式 (/v1/chat/completions) - 用于 gemini-1.5-flash, gemini-2.0-flash
    2. Gemini 原生格式 (/v1beta/models/{model}:generateContent) - 用于 gemini-3-pro-preview-thinking 等
    自动根据模型名称选择 API 格式。
    """

    stateful = False
    type_name = "gemini-chat"
    log_prefix = "GEMINI"

    def __init__(
        self,
        args,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model: str = DEFAULT_MODEL,
        **kwargs,
    ):
        super().__init__(args, temperature=temperature, max_tokens=max_tokens, model=model, **kwargs)
        self.temperature = getattr(args, "temperature", temperature) if args else temperature
        self.max_tokens = getattr(args, "max_tokens", max_tokens) if args else max_tokens
        self.model = model or DEFAULT_MODEL

        # 优先使用环境变量 GEMINI_API_KEY，否则使用固定的 API Key
        # 注意：不使用 OPENAI_API_KEY，因为它可能是其他服务的 key
        self.api_key = (
            os.getenv("GEMINI_API_KEY")
            or getattr(args, "gemini_api_key", None)
            or "sk-purft85XG65PTUKbmubzrDxkIhIMaapySvqItrMdTPmprBcE"  # 固定使用正确的 API Key
        )
        if not self.api_key:
            raise ValueError("未找到 GEMINI_API_KEY 用于 Gemini backend")
        
        # 调试：打印实际使用的 API Key（只显示前20个字符）
        print(f"[DEBUG][GEMINI] Using API Key: {self.api_key[:20]}...", file=sys.stderr)

        # 判断是否使用 Gemini 原生 API 格式
        # 如果模型名包含 "thinking" 或 "3-pro"，使用原生 API
        use_native_api = "thinking" in self.model.lower() or "3-pro" in self.model.lower()
        
        if use_native_api:
            # 使用 Gemini 原生 API 格式 (/v1beta/models/{model}:generateContent)
            base_url = (
                os.getenv("GEMINI_API_BASE")
                or os.getenv("GEMINI_API_URL")
                or "https://api2.aigcbest.top"
            ).rstrip("/")
            # 如果 GEMINI_API_URL 已经包含完整路径，直接使用；否则构建原生 API 路径
            if "/v1beta" in (os.getenv("GEMINI_API_BASE") or "") or "/v1beta" in (os.getenv("GEMINI_API_URL") or ""):
                self.api_url = os.getenv("GEMINI_API_URL") or os.getenv("GEMINI_API_BASE") or f"{base_url}/v1beta/models/{self.model}:generateContent"
            else:
                self.api_url = f"{base_url}/v1beta/models/{self.model}:generateContent"
            self.use_native_api = True
        else:
            # 使用 OpenAI 兼容格式 (/v1/chat/completions)
            self.api_url = (
                os.getenv("GEMINI_API_URL")
                or os.getenv("GEMINI_API_BASE")
                or "https://api2.aigcbest.top/v1/chat/completions"  # 默认固定 API 地址
            )
            self.use_native_api = False

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_random_exponential(min=2, max=20),
        retry=retry_if_exception_type(requests.RequestException),
    )
    def _convert_messages_to_gemini_contents(self, messages):
        """将 OpenAI 格式的 messages 转换为 Gemini 原生 API 格式的 contents"""
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # 转换角色名称
            if role == "system":
                role = "user"  # Gemini 使用 user 角色表示系统消息
            elif role == "assistant":
                role = "model"
            # user 角色保持不变
            
            parts = [{"text": content}]
            contents.append({"role": role, "parts": parts})
        
        return contents

    def _get_response(self, messages, method=None, max_tokens=None, T=None, speaker="Unknown"):
        max_toks = self.max_tokens if max_tokens is None else max_tokens
        temperature = self.temperature if T is None else T

        use_stop = self._supports_stop_parameter(self.model)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        response_text = ""
        try:
            # 调试：打印请求信息（不打印完整的 API Key）
            print(f"[DEBUG][GEMINI] Request URL: {self.api_url}", file=sys.stderr)
            print(f"[DEBUG][GEMINI] Model: {self.model}", file=sys.stderr)
            print(f"[DEBUG][GEMINI] Use Native API: {self.use_native_api}", file=sys.stderr)
            print(f"[DEBUG][GEMINI] API Key (first 20 chars): {self.api_key[:20]}...", file=sys.stderr)
            
            if self.use_native_api:
                # 使用 Gemini 原生 API 格式
                contents = self._convert_messages_to_gemini_contents(messages)
                payload = {
                    "contents": contents,
                    "generationConfig": {
                        "temperature": temperature,
                        "maxOutputTokens": max_toks,
                    }
                }
                
                resp = requests.post(self.api_url, headers=headers, json=payload, timeout=180)
                resp.raise_for_status()
                completion = resp.json()
                
                # Gemini 原生 API 格式响应
                candidates = completion.get("candidates", [])
                if candidates:
                    candidate = candidates[0]
                    parts = candidate.get("content", {}).get("parts", [])
                    if parts:
                        response_text = parts[0].get("text", "")
                finish_reason = candidates[0].get("finishReason", "unknown") if candidates else "unknown"
            else:
                # 使用 OpenAI 兼容格式（已验证可用）
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_toks,
                }
                if use_stop:
                    payload["stop"] = STOP
                
                resp = requests.post(self.api_url, headers=headers, json=payload, timeout=180)
                resp.raise_for_status()
                completion = resp.json()
                
                # OpenAI 兼容格式响应
                choice = (completion.get("choices") or [{}])[0]
                message_obj = choice.get("message") or {}
                response_text = (
                    message_obj.get("content")
                    or choice.get("text")
                    or completion.get("output_text")
                    or ""
                )
                finish_reason = choice.get("finish_reason", "unknown")
            
            print(
                f"[DEBUG][GEMINI] finish_reason={finish_reason}, content_len={len(response_text)}",
                file=sys.stderr,
            )
        except requests.RequestException as exc:
            print(f"[GEMINI ERROR] {exc}", file=sys.stderr)
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                print(f"[GEMINI RAW RESPONSE] {exc.response.text}", file=sys.stderr)
                print(f"[GEMINI REQUEST URL] {self.api_url}", file=sys.stderr)
            raise

        response_text = response_text.strip()
        if not response_text:
            print(f"[WARNING][GEMINI] empty response for model {self.model}", file=sys.stderr)
            response_text = END_OF_MESSAGE
        elif not response_text.endswith(END_OF_MESSAGE):
            response_text = response_text + END_OF_MESSAGE

        self._log_model_reply(speaker, response_text)
        return response_text

