#!/usr/bin/env python3
"""
优化版本的OpenAI后端，减少LLM调用次数
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .openai import OpenAIChat
from ..message import Message, SYSTEM_NAME
import re

class OptimizedOpenAIChat(OpenAIChat):
    """
    优化版本的OpenAI后端，减少LLM调用次数
    """
    
    def query(self, arg, agent_name: str, role_desc: str, history_messages, msgs, ques, global_prompt: str = None,
              request_msg=None, turns=0, day_night="daytime", role="", alives=None, *args, **kwargs):
        """
        优化版本的query方法，减少LLM调用次数
        """
        # 构建简化的prompt
        conversations = []
        for message in history_messages:
            conversations.append({"role": "user", "content": f"{message.agent_name}: {message.content}"})
        
        # 简化的系统提示
        system_prompt = {
            "role": "system", 
            "content": f"{role_desc}\n\nYou are {agent_name}, a {role} in a werewolf game. "
                      f"Current phase: {day_night}, turn {turns}. "
                      f"Alive players: {alives if alives else 'unknown'}. "
                      f"Respond concisely in 1-2 sentences."
        }
        
        # 获取Moderator的最新指令
        moderator_instruction = ""
        for message in reversed(history_messages):
            if message.agent_name == "Moderator" and agent_name in str(message.visible_to):
                moderator_instruction = message.content
                break
        
        # 简化的请求prompt
        if moderator_instruction:
            request_prompt = [{
                "role": "system",
                "content": f"Moderator's instruction: {moderator_instruction}\n\n"
                          f"Please respond with your action. Be concise and direct."
            }]
        else:
            request_prompt = [{
                "role": "system", 
                "content": "Please respond with your action based on the current game situation. Be concise and direct."
            }]
        
        # 构建最终请求
        request = [system_prompt] + conversations + request_prompt
        
        print(f"Optimized request: {request}", file=sys.stderr)
        
        # 单次LLM调用
        response = self._get_response(request, max_tokens=100, T=0.7, *args, **kwargs, speaker=agent_name)
        
        # 清理响应
        response = re.sub(rf"^\s*(\[)?[a-zA-Z0-9\s]*(\])?:\s*", "", response)
        response = re.sub(rf"{self.END_OF_MESSAGE}$", "", response).strip()
        
        print(f"Optimized response: {response}", file=sys.stderr)
        
        return response

# 使用示例
def create_optimized_agent():
    """
    创建使用优化后端的agent
    """
    from ..agent import Player
    
    backend = OptimizedOpenAIChat(
        backend_type="openai-chat",
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=100
    )
    
    return Player(
        name="Player 4",
        role_desc="You are Player 4, a witch in the werewolf game.",
        backend=backend
    )

if __name__ == "__main__":
    print("优化版本OpenAI后端已创建")
    print("主要优化：")
    print("1. 单次LLM调用（原来需要10+次）")
    print("2. 简化的prompt")
    print("3. 直接响应Moderator指令")
    print("4. 减少token使用")






