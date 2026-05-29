from typing import List
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .base import IntelligenceBackend
from ..message import Message, SYSTEM_NAME as SYSTEM

import transformers
from transformers import pipeline

# 定义是否可用
try:
    import transformers
    from transformers import pipeline
except ImportError:
    is_transformers_available = False
else:
    is_transformers_available = True


class TransformersConversational(IntelligenceBackend):
    """
    Interface to the Transformers ConversationalPipeline
    """
    stateful = False
    type_name = "transformers:conversational"

    def __init__(self, model: str, device: int = -1, **kwargs):
        super().__init__(model=model, device=device, **kwargs)
        self.model = model
        self.device = device

        assert is_transformers_available, "Transformers package is not installed"
        self.chatbot = pipeline(task="conversational", model=self.model, device=self.device)

    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
    def _get_response(self, text: str) -> str:
        # 直接传字符串，pipeline返回结果中一般在generated_text里
        outputs = self.chatbot(text)
        # outputs 是列表，取第一个的 generated_text
        if isinstance(outputs, list):
            output = outputs[0]
        else:
            output = outputs
        # 不同模型结构返回可能不一样，尽量取 'generated_text'
        if isinstance(output, dict) and 'generated_text' in output:
            response = output['generated_text']
        else:
            # fallback，可能直接是字符串
            response = str(output)
        return response

    @staticmethod
    def _msg_template(agent_name, content):
        return f"[{agent_name}]: {content}"

    def query(self, agent_name: str, role_desc: str, history_messages: List[Message], global_prompt: str = None,
              request_msg: Message = None, *args, **kwargs) -> str:

        all_messages = []
        if global_prompt:
            all_messages.append((SYSTEM, global_prompt))
        all_messages.append((SYSTEM, role_desc))

        for msg in history_messages:
            all_messages.append((msg.agent_name, msg.content))
        if request_msg:
            all_messages.append((SYSTEM, request_msg.content))

        # 拼接成一个长字符串，格式：
        # [Agent1]: message1
        # [Agent2]: message2
        conversation_text = "\n".join(self._msg_template(name, content) for name, content in all_messages)

        # 传入文本，获取回复
        response = self._get_response(conversation_text)
        return response
