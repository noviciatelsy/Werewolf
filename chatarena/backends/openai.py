from typing import List
import os
import re
import logging
import random
import sys
import json
from tenacity import retry, stop_after_attempt, wait_random_exponential
import requests
from requests.auth import HTTPBasicAuth

from .base import IntelligenceBackend
from ..message import Message, MessagePool, Question, QuestionPool
try:
    import openai
except ImportError:
    is_openai_available = False
    logging.warning("openai package is not installed")
else:
    openai.api_key = "sk-iEI4Hfcfv3Ed5Lcp20C5B2D27dF44532B88d82F951955b65"
    if openai.api_key is None:
        logging.warning("OpenAI API key is not set. Please set the environment variable OPENAI_API_KEY")
        is_openai_available = False
    else:
        is_openai_available = True

# Default config follows the OpenAI playground
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 256
DEFAULT_MODEL = "gpt-3.5-turbo"

STOP = ("<EOS>", "[EOS]", "(EOS)")  # End of sentence token
END_OF_MESSAGE = "<EOS>"


class OpenAIChat(IntelligenceBackend):
    """
    Interface to the ChatGPT style model with system, user, assistant roles separation
    """
    stateful = False
    type_name = "openai-chat"
    log_prefix = "OPENAI"

    def __init__(self, args, temperature: float = DEFAULT_TEMPERATURE, max_tokens: int = DEFAULT_MAX_TOKENS,
                 model: str = DEFAULT_MODEL, **kwargs):
        if not args or (args and not args.use_api_server):
            assert is_openai_available, "openai package is not installed or the API key is not set"
        super().__init__(args, temperature=temperature, max_tokens=max_tokens, model=model, **kwargs)

        if args:
            self.temperature = args.temperature
        else:
            self.temperature = temperature
        if args:
            self.max_tokens = args.max_tokens
        else:
            self.max_tokens = max_tokens
        self.model = model
        self._empathy_cache = {}
        self._round_decision_cache = {}
        self._llm_call_log = []
        try:
            from ..empathy_field import PublicEmpathyField
        except ImportError:
            from chatarena.empathy_field import PublicEmpathyField
        self._public_empathy_field = PublicEmpathyField(round_no=1)
        self._empathy_field_synced_round = -1
        self._pending_speech_propagate = None

    def _supports_stop_parameter(self, model_name):
        """
        检查模型是否支持stop参数
        某些新模型（如gpt-5）不支持stop参数
        如果模型名称包含不支持stop的关键词，返回False
        """
        # 不支持stop参数的模型列表（根据API错误信息判断）
        models_without_stop = ["gpt-5"]
        # 检查模型名称是否在不支持stop的列表中（不区分大小写）
        model_lower = model_name.lower()
        for model in models_without_stop:
            if model in model_lower:
                return False
        # 默认支持stop参数（gpt-3.5-turbo, gpt-4o等通常支持）
        return True

    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=4))
    def _get_response(self, messages, method, max_tokens=None, T=None, speaker="Unknown", log_reply=True):
        max_toks = self.max_tokens if max_tokens is None else max_tokens
        temperature = self.temperature if T is None else T
        method = 1
        
        # 检查模型是否支持stop参数
        use_stop = self._supports_stop_parameter(self.model)

        if method == 0:
            payload_params = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_toks
            }
            if use_stop:
                payload_params["stop"] = STOP
            completion = openai.ChatCompletion.create(**payload_params)
            response = completion.choices[0]['message']['content']
        if method == 1:
            # Warning!!! If you use your self-constructed API server, you should configure it here.

            #url = "https://api.chatweb.plus/v1/chat/completions"

            # 使用与 Gemini 相同的 API 端点和 Key（统一使用 api2.aigcbest.top）
            # 优先使用 GEMINI_API_KEY（因为两个模型都使用相同的 API），如果没有则使用固定的正确 Key
            API_KEY = os.getenv("GEMINI_API_KEY") or "sk-purft85XG65PTUKbmubzrDxkIhIMaapySvqItrMdTPmprBcE"
            # 如果 OPENAI_API_KEY 环境变量存在但错误，忽略它，使用正确的 Key
            if os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_API_KEY").startswith("sk-purft85XG65PTUKbmubzrDxkIhIMaapySvqItrMdTPmprBcE"):
                print(f"[WARNING][OPENAI] 检测到环境变量 OPENAI_API_KEY，但为了统一使用 api2.aigcbest.top，使用 GEMINI_API_KEY 或固定 Key", file=sys.stderr)
            
            API_URL = os.getenv("GEMINI_API_URL") or os.getenv("OPENAI_API_URL") or os.getenv("OPENAI_API_BASE") or "https://api2.aigcbest.top/v1/chat/completions"

            # 调试信息：打印实际使用的 API Key 和 URL（只显示前20个字符）
            print(f"[DEBUG][OPENAI] Using API Key: {API_KEY[:20]}...", file=sys.stderr)
            print(f"[DEBUG][OPENAI] Using API URL: {API_URL}", file=sys.stderr)
            print(f"[DEBUG][OPENAI] Model: {self.model}", file=sys.stderr)

            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }

            # 对于不支持stop参数的模型，大幅增加max_tokens以确保有足够的空间生成内容
            # 因为无法使用stop参数提前停止，需要依赖max_tokens控制长度
            adjusted_max_tokens = max_toks
            if not use_stop:
                # 不支持stop的模型，需要更大的max_tokens
                # 特别针对gpt-5，使用更大的值（500），因为从日志看可能有问题
                if "gpt-5" in self.model.lower():
                    adjusted_max_tokens = max(max_toks, 500)  # gpt-5使用更大的token限制
                    print(f"[DEBUG] Using larger max_tokens for {self.model}: {adjusted_max_tokens} (gpt-5 may need more tokens)", file=sys.stderr)
                else:
                    adjusted_max_tokens = max(max_toks, 200)  # 其他模型使用200
                if adjusted_max_tokens != max_toks:
                    print(f"[DEBUG] Adjusted max_tokens for {self.model} from {max_toks} to {adjusted_max_tokens} (model doesn't support stop)", file=sys.stderr)
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": adjusted_max_tokens
            }
            # 只有支持stop参数的模型才添加stop字段
            if use_stop:
                payload["stop"] = STOP

            #print(f"  Temperature: {temperature}, Max_tokens: {max_toks}", file=sys.stderr)
            #print(f"[DEBUG] Requesting {API_URL} with model {self.model}", file=sys.stderr)

            try:
                response1 = requests.post(API_URL, headers=headers, json=payload, timeout=45)
                response1.raise_for_status()
                completion = response1.json()
                
                # 详细调试信息：打印完整的API响应（仅当响应异常时）
                choice0 = (completion.get('choices') or [{}])[0]
                message_obj = choice0.get('message') or {}
                finish_reason = choice0.get('finish_reason', 'unknown')
                
                # 尝试多种方式获取响应内容
                response = message_obj.get('content')
                if response is None:
                    response = choice0.get('text')
                if response is None:
                    response = completion.get('output_text')
                if response is None:
                    response = ""
                
                # 调试：打印完整响应信息
                print(f"[DEBUG] {self.model} API Response - finish_reason: {finish_reason}, content_length: {len(response) if response else 0}, content_type: {type(response)}", file=sys.stderr)
                if response:
                    print(f"[DEBUG] content_preview (first 100 chars): '{response[:100]}'", file=sys.stderr)
                else:
                    print(f"[DEBUG] content is EMPTY or None", file=sys.stderr)
                    # 打印message_obj的完整内容
                    print(f"[DEBUG] message_obj: {message_obj}", file=sys.stderr)
                    print(f"[DEBUG] choice0: {choice0}", file=sys.stderr)
                
                # 如果响应为空或异常，打印更多调试信息
                if not response or (isinstance(response, str) and response.strip() == ""):
                    print(f"[ERROR] {self.model} returned EMPTY content!", file=sys.stderr)
                    print(f"[ERROR] finish_reason: {finish_reason}", file=sys.stderr)
                    
                    # 检查usage信息
                    if 'usage' in completion:
                        usage = completion['usage']
                        completion_tokens = usage.get('completion_tokens', 0)
                        prompt_tokens = usage.get('prompt_tokens', 0)
                        print(f"[ERROR] Token usage - prompt: {prompt_tokens}, completion: {completion_tokens}, total: {usage.get('total_tokens', 0)}", file=sys.stderr)
                        
                        if completion_tokens == 0:
                            print(f"[ERROR] CRITICAL: Model {self.model} generated 0 completion tokens!", file=sys.stderr)
                            print(f"[ERROR] This strongly suggests the model '{self.model}' may not exist or is not supported by this API.", file=sys.stderr)
                            print(f"[ERROR] Please verify the model name. Try using 'gpt-4o' or 'gpt-4-turbo' instead.", file=sys.stderr)
                    
                    # 如果finish_reason是length但内容为空，这是严重问题
                    if finish_reason == 'length' and not response:
                        print(f"[ERROR] CRITICAL: finish_reason is 'length' but content is EMPTY!", file=sys.stderr)
                        print(f"[ERROR] This is a critical error. Possible causes:", file=sys.stderr)
                        print(f"[ERROR] 1. Model '{self.model}' does not exist or is not supported", file=sys.stderr)
                        print(f"[ERROR] 2. API endpoint does not support this model", file=sys.stderr)
                        print(f"[ERROR] 3. Model name may be incorrect (check API documentation)", file=sys.stderr)
                        print(f"[ERROR] 4. API may have a bug with this model", file=sys.stderr)
                        print(f"[ERROR] RECOMMENDATION: Change model to 'gpt-4o' or 'gpt-4-turbo' in run_werewolf.py", file=sys.stderr)
                        
                        # 打印完整的completion结构（限制长度）
                        try:
                            completion_str = json.dumps(completion, indent=2, ensure_ascii=False)
                            if len(completion_str) > 2000:
                                completion_str = completion_str[:2000] + "\n... (truncated)"
                            print(f"[DEBUG] Full completion structure:\n{completion_str}", file=sys.stderr)
                        except Exception as e:
                            print(f"[DEBUG] Could not serialize completion: {e}", file=sys.stderr)
                        
                        # 尝试增加max_tokens并重试（但这需要修改调用方式）
                        # 或者直接返回错误，让调用者处理
                        
            except Exception as e:
                print(f"[ERROR] Failed to get response: {str(e)}", file=sys.stderr)
                if 'response1' in locals():
                    print(f"[DEBUG] Raw Response: {response1.text}", file=sys.stderr)
                raise e

            #completion = requests.post(url=url, data=data, auth=HTTPBasicAuth(username="noviciate",password="279823lsy2004123")).json()
            #response = completion['choices'][0]['message']['content']

        # 调试：打印原始响应（用于诊断问题）
        raw_response_before_processing = response
        response = response.strip()
        
        # 打印原始响应用于调试（仅当响应异常时）
        if not response or response == "<EOS>" or response == "[EOS]" or response == "(EOS)":
            print(f"[DEBUG] Raw response from {self.model}: '{raw_response_before_processing}' (length: {len(raw_response_before_processing)})", file=sys.stderr)
        
        # 如果响应为空，记录警告
        if not response:
            print(f"[WARNING] Empty response from model {self.model}. This may indicate a problem.", file=sys.stderr)
            response = ""
        else:
            # 如果模型不支持stop参数，响应可能包含多余的文本
            # 检查响应中是否已经包含EOS标记
            has_eos_in_response = False
            eos_variants = ["<EOS>", "[EOS]", "(EOS)", "<EOS", "[EOS", "(EOS"]
            
            # 查找EOS标记的位置
            eos_pos = -1
            found_eos_variant = None
            for eos_variant in eos_variants:
                pos = response.find(eos_variant)
                if pos >= 0:
                    eos_pos = pos
                    found_eos_variant = eos_variant
                    has_eos_in_response = True
                    break
            
            if has_eos_in_response and eos_pos > 0:
                # EOS标记不在开头，说明有实际内容
                # 截取EOS之前的内容
                response = response[:eos_pos].strip()
            elif has_eos_in_response and eos_pos == 0:
                # EOS标记在开头（或响应只有EOS），说明模型只返回了EOS
                # 这是一个问题，但我们需要保留响应
                print(f"[WARNING] Model {self.model} returned only EOS marker (or EOS at start). Raw: '{raw_response_before_processing}'", file=sys.stderr)
                # 如果响应只有EOS，尝试检查是否有其他内容
                # 移除EOS标记，看看是否有其他内容
                response_without_eos = response.replace(found_eos_variant, "").strip()
                if response_without_eos:
                    # 有其他内容，使用它
                    response = response_without_eos
                else:
                    # 确实只有EOS，这可能是模型的问题
                    # 但为了不中断游戏，我们返回一个默认响应或空响应
                    print(f"[ERROR] Model {self.model} returned only EOS without any content. This is a serious issue.", file=sys.stderr)
                    response = ""  # 返回空字符串，让调用者处理
            
            # 统一补全/规范化 EOS 结尾（仅用于协议边界，不把哨兵混入业务文本）
            if response:
                if response.endswith("<EOS"):
                    response = response + ">"
                elif not response.endswith(END_OF_MESSAGE):
                    response = response + END_OF_MESSAGE

        # 这里打印模型回复
        print("Model reply:", response, file=sys.stderr)

        # 记录模型回复（INTERNAL 共情分析不写入 model_reply.log）
        if log_reply and not str(speaker).startswith("INTERNAL_"):
            clean_response = self._sanitize_model_reply(response)
            if not (clean_response.strip().startswith('{') and ('"Error"' in clean_response or '"error"' in clean_response)):
                self._log_model_reply(speaker, clean_response)
            else:
                print(f"[DEBUG] Skipping JSON error log: {clean_response[:100]}...", file=sys.stderr)

        return response

    def _sanitize_model_reply(self, s: str) -> str:
        """Remove EOS/marker noise and conversation sentinels from logged model content."""
        if not s:
            return ""
        text = re.sub(rf"{re.escape(END_OF_MESSAGE)}$", "", str(s)).strip()
        text = text.replace("<<<<<<END_OF_CONVERSATION>>>>>>", "")
        text = text.replace("END_OF_CONVERSATION", "")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def extract_text(self, s):
        """改进的文本提取方法，更好地处理各种格式的发言内容"""
        # 首先尝试提取"My concise talking content:"后的内容
        concise_pattern = r'My concise talking content:\s*(.+?)(?:\n|$)'
        match = re.search(concise_pattern, s, re.IGNORECASE | re.DOTALL)
        if match:
            content = match.group(1).strip()
            # 清理内容
            content = re.sub(r'^["\']|["\']$', '', content)  # 移除首尾引号
            content = re.sub(r'\s+', ' ', content)  # 合并多个空格
            return content
        
        # 尝试提取引号内的内容
        quote_patterns = [
            r': "(.+?)"', 
            r'content: (.+)', 
            r'content:\n(.+)', 
            r'content:\n\n(.+)', 
            r'content: \n(.+)'
        ]
        
        for pattern in quote_patterns:
            match = re.search(pattern, s, re.DOTALL)
            if match:
                content = match.group(1).strip()
                if content and len(content) > 5:
                    return content
        
        # 尝试提取时间相关的发言
        time_patterns = [
            r'night:\s*(.+)',
            r'night:\n(.+)', 
            r'night:\n\n(.+)', 
            r'night: \n(.+)',
            r'daytime:\s*(.+)',
            r'daytime:\n(.+)', 
            r'daytime:\n\n(.+)', 
            r'daytime: \n(.+)',
            r'"(.+)"', 
            r'"(.+)'
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, s, re.DOTALL)
            if match:
                content = match.group(1).strip()
                if content and len(content) > 5:
                    return content
        
        # 如果以上都失败，返回清理后的原始文本
        cleaned = s.strip()
        cleaned = re.sub(r'^\s*(\[)?[a-zA-Z0-9\s]*(\])?:\s*', '', cleaned)  # 移除开头的角色标识
        cleaned = re.sub(r'\s+', ' ', cleaned)  # 合并多个空格
        cleaned = re.sub(r'^["\']|["\']$', '', cleaned)  # 移除首尾引号
        
        return cleaned

    def _validate_and_correct_response(self, response, agent_name, role, alives=None, history_messages=None):
        """验证和修正agent响应"""
        import re
        
        # 检查身份混淆
        if not response or response.strip() in (END_OF_MESSAGE, "<<<<<<END_OF_CONVERSATION>>>>>>"):
            return "I choose pass."
        wrong_identity_patterns = [
            r"I am Player \d+, the (werewolf|villager|seer|witch|guard)",
            r"I voted for Player \d+ to be killed by my teammate",
            r"My teammate.*voted.*Player \d+"
        ]
        
        for pattern in wrong_identity_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                # 修正身份混淆
                response = re.sub(r"I am Player \d+, the (werewolf|villager|seer|witch|guard)", 
                                 f"I am {agent_name}, the {role}", response, flags=re.IGNORECASE)
                
                # 修正错误的投票声明
                response = re.sub(r"I voted for Player \d+ to be killed by my teammate", 
                                 "I need to coordinate with my teammates on who to vote for", response, flags=re.IGNORECASE)
        
        # 防止狼人投票杀死自己或队友
        if role == "werewolf":
            # 从历史消息中提取狼人队友信息
            werewolf_teammates = []
            if history_messages:
                for message in history_messages:
                    if "are all of the" in message.content and "werewolves" in message.content:
                        teammates_match = re.search(r'Player \d+, Player \d+ are all of the \d+ werewolves', message.content)
                        if teammates_match:
                            teammates_text = teammates_match.group(0)
                            teammates = re.findall(r'Player \d+', teammates_text)
                            werewolf_teammates = teammates
                            break
            
            # 检查各种投票模式
            vote_patterns = [
                r"I vote to kill (Player \d+)",
                r"I choose (Player \d+)",
                r"I choose to eliminate (Player \d+)",
                r"I choose to kill (Player \d+)",
                r"Let's target (Player \d+)",
                r"target (Player \d+)"
            ]
            
            for pattern in vote_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    target = match.group(1)
                    # 如果目标是自己或队友，修正为其他玩家
                    if target == agent_name or target in werewolf_teammates:
                        # 从存活玩家中选择一个非狼人目标
                        if alives:
                            other_players = [p for p in alives if p != agent_name and p != 'pass' and p not in werewolf_teammates]
                            if other_players:
                                import random
                                safe_target = random.choice(other_players)
                                response = re.sub(pattern, f"I choose {safe_target}", response, flags=re.IGNORECASE)
                        else:
                            # 备用方案
                            response = re.sub(pattern, "I choose Player 1", response, flags=re.IGNORECASE)
                    break
        
        # 防止守卫保护自己
        if role == "guard":
            protect_patterns = [
                r"I protect (Player \d+)",
                r"I choose to protect (Player \d+)"
            ]
            
            for pattern in protect_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match and match.group(1) == agent_name:
                    # 从存活玩家中选择一个非自己的目标
                    if alives:
                        other_players = [p for p in alives if p != agent_name and p != 'pass']
                        if other_players:
                            import random
                            safe_target = random.choice(other_players)
                            response = re.sub(pattern, f"I protect {safe_target}", response, flags=re.IGNORECASE)
                    else:
                        response = re.sub(pattern, "I protect Player 1", response, flags=re.IGNORECASE)
                    break
        
        # 防止预言家验证自己
        if role == "seer":
            verify_patterns = [
                r"I verify (Player \d+)",
                r"I choose to verify (Player \d+)"
            ]
            
            for pattern in verify_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match and match.group(1) == agent_name:
                    # 从存活玩家中选择一个非自己的目标
                    if alives:
                        other_players = [p for p in alives if p != agent_name and p != 'pass']
                        if other_players:
                            import random
                            safe_target = random.choice(other_players)
                            response = re.sub(pattern, f"I verify {safe_target}", response, flags=re.IGNORECASE)
                    else:
                        response = re.sub(pattern, "I verify Player 1", response, flags=re.IGNORECASE)
                    break
        
        return response

    def _write_structured_log(self, arg, agent_name, role, turns, day_night, response, task, alives=None, empathy_data=None, reflection_context=None, phase_name=None, stage_debug=None):
        """写入结构化日志到model_reply.log"""
        import re
        import time
        
        # 辅助函数：安全获取 task 内容
        def _get_task_content(task):
            """安全获取 task 内容"""
            if task is None:
                return ""
            if not isinstance(task, dict):
                return str(task)
            return task.get("content", "")
        
        # 解析发言类型
        speech_type = "UNKNOWN"
        target_player = "N/A"
        vote_target = "N/A"
        
        # 检查是否是投票/选择类发言
        task_content = _get_task_content(task)
        if "Choose" in task_content or "choose" in task_content or "vote to" in task_content or "Yes, No" in task_content:
            speech_type = "ACTION"
            # 尝试提取投票目标
            vote_patterns = [
                r'vote\s+(Player\s*\d+)',
                r'choose\s+(Player\s*\d+)',
                r'kill\s+(Player\s*\d+)',
                r'protect\s+(Player\s*\d+)',
                r'(\bPlayer\s*\d+\b)'
            ]
            for pattern in vote_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    vote_target = match.group(1)
                    break
            
            # 检查是否是Yes/No选择
            if "Yes" in response or "No" in response:
                vote_target = "Yes/No Choice"
        else:
            speech_type = "SPEECH"
            # 尝试提取目标玩家（发言中提到的玩家）
            player_patterns = [
                r'(\bPlayer\s*\d+\b)',
                r'(\bplayer\s*\d+\b)'
            ]
            mentioned_players = []
            for pattern in player_patterns:
                matches = re.findall(pattern, response, re.IGNORECASE)
                mentioned_players.extend(matches)
            
            if mentioned_players:
                target_player = ", ".join(set(mentioned_players))
        
        # 确定发言对象
        audience = "ALL"  # 默认对所有人说
        if "teammates" in response.lower() or "team" in response.lower():
            audience = "TEAM"
        elif 'mentioned_players' in locals() and mentioned_players and any(player in response for player in mentioned_players):
            audience = "TARGETED"
        
        # 获取当前时间戳
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 获取存活玩家信息
        alive_players = "N/A"
        if alives:
            alive_players = ", ".join([p for p in alives if p != 'pass'])
        
        # 写入结构化日志
        with open("model_reply.log", "a", encoding="utf-8") as f:
            f.write(f"\n=== STRUCTURED LOG ===\n")
            f.write(f"TIMESTAMP: {timestamp}\n")
            f.write(f"AGENT: {agent_name}\n")
            f.write(f"ROLE: {role}\n")
            f.write(f"GAME_PHASE: {day_night.upper()}\n")
            f.write(f"TURN: {turns}\n")
            f.write(f"ALIVE_PLAYERS: {alive_players}\n")
            f.write(f"SPEECH_TYPE: {speech_type}\n")
            f.write(f"AUDIENCE: {audience}\n")
            f.write(f"TARGET_PLAYER: {target_player}\n")
            f.write(f"VOTE_TARGET: {vote_target}\n")
            f.write(f"PHASE_NAME: {phase_name or day_night}\n")
            if stage_debug is not None:
                try:
                    f.write(f"STAGE_DEBUG: {json.dumps(stage_debug, ensure_ascii=False)}\n")
                except Exception:
                    f.write(f"STAGE_DEBUG: {str(stage_debug)}\n")
            if empathy_data is not None:
                try:
                    f.write(f"EMPATHY_SUMMARY: {self._summarize_empathy_for_log(empathy_data)}\n")
                    f.write(f"EMPATHY_DETAIL: {self._format_empathy_detail_for_log(empathy_data)}\n")
                except Exception:
                    pass
            if reflection_context is not None:
                try:
                    f.write(f"REFLECTION_CONTEXT: {self._summarize_reflection_for_log(reflection_context)}\n")
                except Exception:
                    pass
            try:
                target_for_chain = None
                if isinstance(stage_debug, dict):
                    target_for_chain = stage_debug.get("target_player") or stage_debug.get("vote_target")
                if not target_for_chain and target_player != "N/A":
                    target_for_chain = target_player if target_player != "N/A" else None
                causal_evidence = self._build_causal_chain_evidence(empathy_data, target_for_chain, role, None, history_messages=None)
                causal_chain = self._build_causal_chain_from_evidence(role, causal_evidence, reflection_context)
                f.write(f"CAUSAL_CHAIN: {json.dumps(causal_chain, ensure_ascii=False)}\n")
            except Exception:
                pass
            f.write(f"CONTENT: {response}\n")
            f.write(f"===================\n")

    def query(self, arg, agent_name: str, role_desc: str, history_messages: List[Message], msgs: MessagePool, ques: QuestionPool, global_prompt: str = None,
              request_msg: Message = None, turns = 0, day_night = "daytime", role="", alives=[], _skip_empathy=False, *args, **kwargs, ) -> str:
        """
        format the input and call the ChatGPT/GPT-4 API
        args:
            agent_name: the name of the agent
            role_desc: the description of the role of the agent
            env_desc: the description of the environment
            history_messages: the history of the conversation, or the observation for the agent
            request_msg: the request for the chatGPT
        """
        def _get_task_content(task):
            """安全获取 task 内容"""
            if task is None:
                return ""
            if not isinstance(task, dict):
                return str(task)
            return task.get("content", "")
        
        def _get_branch(task, day_night, role):
            # 安全检查：确保 task 不为 None
            task_content = _get_task_content(task)
            
            if "Choose" in task_content or "choose" in task_content or "vote to" in task_content or "Yes, No" in task_content:
                if role == "werewolf" or role == "seer" or role == "guard":
                    if day_night == "night":
                        return 2
                # 女巫阶段判断：优先判断毒药阶段，再判断救人阶段
                if role == "witch" and day_night == "night":
                    task_content_lower_branch = task_content.lower()
                    # 毒药阶段：包含"poison"和"who are you going to kill"
                    if "poison" in task_content_lower_branch and "who are you going to kill" in task_content_lower_branch:
                        return 3
                    # 救人阶段：必须是明确询问是否使用解药
                    elif "antidote" in task_content_lower_branch or "do you want to save" in task_content_lower_branch or ("save" in task_content_lower_branch and "will be killed" in task_content_lower_branch):
                        return 2
                    # 只有明确询问“你要毒谁”才算毒药阶段
                    elif "poison" in task_content_lower_branch and ("who are you going to kill" in task_content_lower_branch or "who are you going to poison" in task_content_lower_branch):
                        return 3
                    else:
                        return 1
                return 1
            else:
                return 0
        
        conn_method = arg.use_api_server if arg and arg.use_api_server else 0
        max_tokens = arg.max_tokens if arg and arg.max_tokens else 100
        temperature = arg.temperature if arg and arg.temperature else 0.2

        alives = alives.copy()
        if 'pass' in alives:
            alives.remove('pass')

        alive_players = "Living players now: " + ", ".join(alives) + "."
        if arg:
            f = open(os.path.join(arg.logs_path_to, str(arg.current_game_number) + ".md"), "a")
            f.write(f"**{agent_name}**:  \n")


        conversations = []

        for i, message in enumerate(history_messages):
            if message.agent_name == agent_name:
                conversations.append({"role": "assistant", "content": f"{message.agent_name}: {message.content}{END_OF_MESSAGE}"})
            else:
                # Since there are more than one player, we need to distinguish between the players
                conversations.append({"role": "user", "content": f"{message.agent_name}: {message.content}{END_OF_MESSAGE}"})
        
        # 注意：不要在这里pop task，而是在检查完女巫阶段后再pop（如果需要）
        # 这样可以在检查时访问完整的conversations列表
        task = None  # 初始化为None，稍后根据情况设置

        global_desc = f"The following is the chat history you observed. You are {agent_name}, the {role}."
        
        # 身份验证prompt - 改为按角色明确世界观，避免守卫/预言家/女巫/狼人提示混淆
        identity_random_warning = (
            "CRITICAL IDENTITY REMINDER: Your identity is RANDOMLY assigned at the start of EACH game. "
            "Do NOT assume your identity from previous games. Do NOT confuse yourself with other players. "
            "ONLY trust the identity the moderator tells you in THIS specific game. "
            f"In THIS game, you are {agent_name} and your role is {role}. "
            "This identity applies ONLY to this game and may be different in other games. "
            "CRITICAL LANGUAGE REQUIREMENT: You MUST respond in ENGLISH ONLY. Do NOT use Chinese (中文), Japanese, Korean, or any other language."
        )

        if role == "werewolf":
            werewolf_teammates = []
            wolf_private_directive = (
                "You are a werewolf in a hidden-information adversarial game. "
                "Your job is to hide your identity, create doubt among good players, and avoid sounding like a cooperative villager. "
                "Never confess, never say you are a werewolf, and never discuss your real kill plan publicly."
            )
            if history_messages:
                for message in history_messages:
                    if message.agent_name == "Moderator" and "werewolves" in message.content.lower():
                        teammates_patterns = [
                            r'(Player \d+(?:, Player \d+)*)\s+are all of the \d+ werewolves',
                            r'secretly tell you that\s+(Player \d+(?:, Player \d+)*)\s+are all of the',
                            r'werewolves[!.]?\s+(?:I|We|The moderator)\s+secretly tell you that\s+(Player \d+(?:, Player \d+)*)',
                        ]
                        for pattern in teammates_patterns:
                            teammates_match = re.search(pattern, message.content, re.IGNORECASE)
                            if teammates_match:
                                teammates_text = teammates_match.group(1)
                                teammates = re.findall(r'Player \d+', teammates_text)
                                werewolf_teammates = [t for t in teammates if t != agent_name]
                                if werewolf_teammates:
                                    break
                        if werewolf_teammates:
                            break
            teammates_str = ", ".join(werewolf_teammates) if werewolf_teammates else "(private teammate info from moderator only)"
            identity_verification = (
                f"{identity_random_warning}\n\n"
                f"{wolf_private_directive}\n\n"
                f"[Werewolf Worldview]\n"
                f"Your teammates are: {teammates_str}.\n"
                f"Night rule: choose a non-teammate living target to kill; never choose pass unless no valid target exists.\n"
                f"Day rule: appear analytical and helpful, but do not over-defend teammates or make unnatural self-justifying statements.\n"
                f"You win by eliminating all non-werewolves."
            )
        elif role == "guard":
            identity_verification = (
                f"{identity_random_warning}\n\n"
                f"[Guard Worldview]\n"
                f"You are the guard. At night, protect one living player (not yourself) who is likely to be attacked or is strategically valuable.\n"
                f"Do NOT use pass as a default. Only choose pass if the prompt or game state makes protection impossible or invalid.\n"
                f"During discussion, protection information can be shared strategically if it helps the village narrow suspects."
            )
        elif role == "seer":
            identity_verification = (
                f"{identity_random_warning}\n\n"
                f"[Seer Worldview]\n"
                f"You are the seer. At night, verify one living player to learn whether they are a werewolf.\n"
                f"Do NOT use pass as a default. Only choose pass if there is truly no valid living target.\n"
                f"During discussion, share verified results clearly when it helps the village."
            )
        elif role == "witch":
            identity_verification = (
                f"{identity_random_warning}\n\n"
                f"[Witch Worldview]\n"
                f"You are the witch. You have two distinct night actions: antidote (save) and poison.\n"
                f"When asked to save, answer only Yes or No. When asked to poison, default to pass unless there is strong evidence.\n"
                f"First night poison should be pass. Do NOT confuse the save prompt with the poison prompt."
            )
        else:
            identity_verification = (
                f"{identity_random_warning}\n\n"
                f"[Villager Worldview]\n"
                f"You are a villager. You have no night action. Your job is to identify contradictions, pressure special roles to share useful info, and vote with the strongest evidence.\n"
                f"Never choose pass in a vote unless the prompt explicitly allows it and you truly have no better option."
            )
        
        # 构建系统提示
        if global_prompt:  # Prepend the global prompt if it exists
            system_prompt_str = f"{identity_verification}\n{global_prompt.strip()}\n{role_desc}\n{global_desc}"
        else:
            system_prompt_str = f"{identity_verification}\n{role_desc}\n{global_desc}"
        system_prompt = {"role": "system", "content": system_prompt_str}
        
        # ===== MCTS/共情开关：is_mcts == 1 走完整流程；== 0 走最小单步 =====
        is_mcts_enabled = True
        if arg and hasattr(arg, 'is_mcts'):
            try:
                is_mcts_enabled = bool(int(getattr(arg, 'is_mcts')))
            except Exception:
                is_mcts_enabled = True

        if not is_mcts_enabled:
            # 最小单步：按昼夜与角色给出动作/发言模板，不做检索/批问/思考链
            alives_copy = alives.copy() if alives else []
            if 'pass' in alives_copy:
                alives_copy.remove('pass')
            alive_options = f"[{', '.join(alives_copy)}]" if alives_copy else "[Player 1, Player 2, Player 3, Player 4, Player 5, Player 6, Player 7, pass]"

            if day_night == "night":
                if role in ("werewolf", "wolf"):
                    action_prompt = (
                        f"Now it's the {turns}-th night. You are a werewolf. This is an adversarial hidden-information game. "
                        f"Choose a living target to kill, and do not reveal your true identity or narrate like a villager. "
                        f"Reply EXACTLY with one line in the format: 'I vote to kill Player X{END_OF_MESSAGE}' or 'I choose pass{END_OF_MESSAGE}'. Valid options: {alive_options}."
                    )
                elif role == "guard":
                    action_prompt = (
                        f"Now it's the {turns}-th night. You are the guard. Choose someone to protect (NOT yourself). "
                        f"Reply EXACTLY with: 'I protect Player X{END_OF_MESSAGE}' or 'I choose pass{END_OF_MESSAGE}'. Valid options: {alive_options}."
                    )
                elif role == "witch":
                    action_prompt = (
                        f"Now it's the {turns}-th night. You are the witch. If asked to save with antidote, reply EXACTLY 'Yes{END_OF_MESSAGE}' or 'No{END_OF_MESSAGE}'. "
                        f"If asked whom to poison, first night MUST be pass; later nights default to 'I choose pass{END_OF_MESSAGE}' unless overwhelming hard evidence exists. Valid options: {alive_options}."
                    )
                elif role == "seer":
                    action_prompt = (
                        f"Now it's the {turns}-th night. You are the seer. Reply EXACTLY with: 'I verify Player X{END_OF_MESSAGE}' or 'I choose pass{END_OF_MESSAGE}'. Valid options: {alive_options}."
                    )
                else:
                    action_prompt = (
                        f"Now it's the {turns}-th night. Give one concise sentence based on the context. End with '{END_OF_MESSAGE}'."
                    )
            else:
                action_prompt = (
                    f"Now it's the {turns}-th day. Give your concise talking content (no more than 2 sentences). "
                    f"CRITICAL: Do NOT explicitly reveal your role in your speech. Use normal, natural language. "
                    f"Never say phrases like 'I am a werewolf', 'as a werewolf', 'being a werewolf', 'I need to blend in', or any phrase that reveals your role or that you are hiding something. "
                    f"Speak naturally as a player trying to find werewolves or defend yourself. End with '{END_OF_MESSAGE}'."
                )

            request_min = [system_prompt] + conversations + [{"role": "system", "content": action_prompt}]
            response_min = self._get_response(request_min, conn_method, T=temperature, max_tokens=max_tokens, *args, **kwargs, speaker = agent_name)
            response_min = re.sub(rf"^\s*(\[)?[a-zA-Z0-9\s]*(\])?:\s*", "", response_min)
            response_min = re.sub(rf"{END_OF_MESSAGE}$", "", response_min).strip()
            # 规范最终文本
            response_min = self.extract_text(response_min)
            response_min = re.sub(rf"^\s*(\[)?[a-zA-Z0-9\s]*(\])?:\s*", "", response_min).strip()
            # 记录响应到日志
            try:
                self._log_model_reply(agent_name, response_min)
            except Exception as e:
                print(f"[WARNING] Failed to log response: {e}", file=sys.stderr)
            return response_min

        # ===== 以下为 is_mcts == 1 的完整流程 =====
        
        # 为MCTS流程添加全局英文要求（在所有对话之前）
        english_requirement = {
            "role": "system",
            "content": "CRITICAL INSTRUCTION: You MUST respond in ENGLISH ONLY for ALL your responses. Do NOT use Chinese (中文), Japanese, Korean, or any other language. This is absolutely mandatory. Every single response must be in English."
        }
        # 将英文要求插入到conversations开头（如果还没有）
        if conversations and not any("ENGLISH ONLY" in str(msg.get("content", "")) for msg in conversations if isinstance(msg, dict)):
            conversations = [english_requirement] + conversations
        
        # 初始化task变量（在MCTS流程开始前）
        if task is None:
            if conversations:
                task = conversations.pop()
            else:
                task = {"role": "user", "content": "Moderator: Please make your first move based on your role."}
        
        # 确保 task 不为 None 并且有正确的格式
        if task is None:
            task = {"role": "user", "content": "Moderator: Please make your first move based on your role."}
        
        # 确保 task 是字典格式
        if not isinstance(task, dict):
            task = {"role": "user", "content": str(task)}
        
        # 确保 task 有 role 字段
        if "role" not in task:
            task["role"] = "user"
        
        task_content = _get_task_content(task) if task else ""
        action_phase_pre = self._detect_action_phase(task_content, day_night, role)

        # 1. 构建游戏状态 + 实时观测（发言/半轮票型/当前 moderator 任务）
        game_state = self._build_game_state(
            agent_name, role, alives, history_messages, turns, day_night, task
        )
        try:
            from ..MCTS import refresh_live_observation, refresh_empathy_reports_live
        except ImportError:
            from chatarena.MCTS import refresh_live_observation, refresh_empathy_reports_live

        refresh_live_observation(
            game_state,
            history_messages=history_messages,
            agent_name=agent_name,
            my_role=role,
            task_content=task_content,
            sync_public_field=self._public_empathy_field,
        )

        # 2. 统一状态层：belief_state 先附着，再作为共情与决策的唯一后验来源
        try:
            from ..belief_state import BeliefStateStore
        except ImportError:
            from chatarena.belief_state import BeliefStateStore
        try:
            from ..MCTS import get_game_analytics
        except ImportError:
            from chatarena.MCTS import get_game_analytics
        skip_empathy_llm = _skip_empathy or action_phase_pre == "voting"
        empathy_data = self._get_empathy_for_round(
            game_state, agent_name, arg, _skip_empathy=skip_empathy_llm
        )
        belief, empathy_from_belief = BeliefStateStore.sync(
            game_state, agent_name, empathy_data, analytics=get_game_analytics(game_state)
        )
        if action_phase_pre == "discussion":
            # 讨论阶段优先使用公开场 + 反思链生成的动态共情结果
            empathy_data = refresh_empathy_reports_live(game_state, agent_name, empathy_from_belief)
        else:
            empathy_data = empathy_from_belief
        if belief and belief.viewer == agent_name:
            belief_reports = belief.to_empathy_reports()
            for p, rep in belief_reports.items():
                if p.startswith("Player") and isinstance(rep, dict):
                    empathy_data[p] = {**empathy_data.get(p, {}), **rep}
            empathy_data["_belief"] = belief_reports.get("_belief", {})
            empathy_data["_private"] = belief_reports.get("_private", {})
        print(
            f"[MCTS] 共情数据就绪(含实时刷新): agent={agent_name}, 玩家数="
            f"{len([k for k in empathy_data if k != '_game'])}",
            file=sys.stderr,
        )
        
        # 3. MCTS搜索决策（纯计算，不调用 LLM）
        try:
            from ..MCTS import MCTS
            
            print(f"[MCTS] 开始MCTS搜索: agent={agent_name}, role={role}, day_night={day_night}", file=sys.stderr)
            
            # 构建玩家角色映射
            player_roles = self._infer_player_roles(alives, role, agent_name)
            print(f"[MCTS] 玩家角色映射: {player_roles}", file=sys.stderr)
            
            prior_decision = None
            round_no = getattr(game_state, "round_no", turns)
            if action_phase_pre == "voting":
                prior_decision = belief.get_prior_decision() if belief and belief.viewer == agent_name else None
                if (
                    belief
                    and belief.viewer == agent_name
                    and belief.commitment.round_no == round_no
                    and (
                        belief.commitment.focus_player
                        or belief.commitment.decision_brief
                    )
                ):
                    prior_decision = belief.get_prior_decision()
                    print(
                        f"[BeliefState] 投票读取讨论承诺: stance={belief.commitment.stance}, "
                        f"focus={belief.commitment.focus_player}, lean={belief.commitment.vote_lean}",
                        file=sys.stderr,
                    )
                if not prior_decision:
                    prior_decision = self._round_decision_cache.get((agent_name, round_no))
            
            best_node = MCTS(
                root_state=game_state,
                current_player=agent_name,
                backend=None,
                agent_name=agent_name,
                iter_num=8,
                args=arg,
                msgs=msgs,
                ques=ques,
                empathy_data=empathy_data,
                player_roles=player_roles,
                prior_decision=prior_decision,
            )
            
            mcts_action = getattr(best_node, "action", None)
            print(f"[MCTS] MCTS搜索完成(参考 בלבד): action={mcts_action}", file=sys.stderr)
            
            # 4. 以 LLM 为主、MCTS 仅作参考生成当前阶段回复
            task_content = _get_task_content(task) if task else ""
            action_phase = self._detect_action_phase(task_content, day_night, role)
            print(f"[MCTS] 当前阶段: {action_phase}", file=sys.stderr)

            # LLM remains the primary decision maker; MCTS is only weak reference metadata.
            llm_target = self._infer_llm_preferred_target(role, alives, agent_name, game_state, empathy_data, action_phase, task_content)
            vote_target = llm_target if llm_target else "pass"
            target_player = vote_target
            speech_style = "natural"
            empathy_context = getattr(best_node, "empathy_context", {}) or {}
            if isinstance(empathy_context, dict):
                empathy_context["mcts_reference"] = mcts_action
                empathy_context["mcts_weight"] = "very_low"
                empathy_context["decision_source"] = "llm_primary"

            reflection_context = None
            # 讨论阶段：先做一次双视角反思，再进行最终发言生成
            if action_phase == "discussion":
                reflection_context = self._generate_counterfactual_reflection(
                    role=role,
                    game_state=game_state,
                    empathy_data=empathy_data,
                    action=(vote_target, target_player, speech_style),
                    system_prompt=system_prompt,
                    conversations=conversations,
                    task=task,
                    task_content=task_content,
                    conn_method=conn_method,
                    alives=alives,
                    agent_name=agent_name,
                    history_messages=history_messages,
                    empathy_context=empathy_context,
                    *args,
                    **kwargs,
                )
                print(f"[DEBUG][REFLECTION] agent={agent_name}, reflection={reflection_context}", file=sys.stderr)
                try:
                    self._write_structured_log(
                        arg, agent_name, role, turns, day_night, "[REFLECTION_ONLY]", task,
                        alives=alives, empathy_data=empathy_data, reflection_context=reflection_context,
                        phase_name="reflection", stage_debug={
                            "phase": "reflection",
                            "round_no": getattr(game_state, "round_no", turns),
                            "best_action": mcts_action,
                            "speech_style": speech_style,
                            "vote_target": vote_target,
                            "target_player": target_player,
                        }
                    )
                except Exception as e:
                    print(f"[MCTS] 反思结构化日志失败: {e}", file=sys.stderr)
                response = self._generate_mcts_phase_response(
                    phase=action_phase,
                    role=role,
                    vote_target=vote_target,
                    target_player=target_player,
                    speech_style=speech_style,
                    empathy_context=empathy_context,
                    empathy_data=empathy_data,
                    game_state=game_state,
                    action=(vote_target, target_player, speech_style),
                    system_prompt=system_prompt,
                    conversations=conversations,
                    task=task,
                    task_content=task_content,
                    conn_method=conn_method,
                    alives=alives,
                    agent_name=agent_name,
                    arg=arg,
                    history_messages=history_messages,
                    reflection_context=reflection_context,
                    *args,
                    **kwargs,
                )
                print(f"[DEBUG][FINAL_PROMPT_PHASE={action_phase}] agent={agent_name}, response_preview={str(response)[:160]}", file=sys.stderr)
            elif action_phase == "voting":
                response = self._normalize_phase_response(
                    "voting", f"I vote to kill {vote_target}." if vote_target != "pass" else "I choose pass.", role, vote_target, agent_name, alives, task_content
                )
            elif action_phase == "witch_save":
                response = self._normalize_phase_response(
                    "witch_save", "Yes", role, vote_target, agent_name, alives, task_content
                )
            elif action_phase == "witch_poison":
                response = self._normalize_phase_response(
                    "witch_poison", f"I choose {vote_target}." if vote_target != "pass" else "I choose pass.", role, vote_target, agent_name, alives, task_content
                )
            else:
                response = self._normalize_phase_response(
                    "night", f"I choose {vote_target}." if vote_target != "pass" else "I choose pass.", role, vote_target, agent_name, alives, task_content
                )
            
            print(f"[MCTS] 决策完成: action={mcts_action}, response={response}", file=sys.stderr)

            if action_phase == "discussion":
                from ..MCTS import resolve_agent_intent, build_decision_brief
                try:
                    from ..empathy_field import strategic_plan_from_mcts
                except ImportError:
                    from chatarena.empathy_field import strategic_plan_from_mcts

                tr = empathy_data.get(target_player, {})
                intent = empathy_context.get("agent_intent") or resolve_agent_intent(
                    (vote_target, target_player, speech_style), tr, role, game_state, agent_name
                )
                
                task_content = _get_task_content(task) if task else ""
                action_phase = self._detect_action_phase(task_content, day_night, role)
                print(f"[MCTS] 当前阶段: {action_phase}", file=sys.stderr)

                empathy_context = getattr(best_node, "empathy_context", {}) or {}
                reflection_context = None
                # 讨论阶段：先做一次双视角反思，再进行最终发言生成
                if action_phase == "discussion":
                    reflection_context = self._generate_counterfactual_reflection(
                        role=role,
                        game_state=game_state,
                        empathy_data=empathy_data,
                        action=best_node.action,
                        system_prompt=system_prompt,
                        conversations=conversations,
                        task=task,
                        task_content=task_content,
                        conn_method=conn_method,
                        alives=alives,
                        agent_name=agent_name,
                        history_messages=history_messages,
                        empathy_context=empathy_context,
                        *args,
                        **kwargs,
                    )
                    print(f"[DEBUG][REFLECTION] agent={agent_name}, reflection={reflection_context}", file=sys.stderr)
                    try:
                        self._write_structured_log(
                            arg, agent_name, role, turns, day_night, "[REFLECTION_ONLY]", task,
                            alives=alives, empathy_data=empathy_data, reflection_context=reflection_context,
                            phase_name="reflection", stage_debug={
                                "phase": "reflection",
                                "round_no": getattr(game_state, "round_no", turns),
                                "best_action": best_node.action,
                                "speech_style": speech_style,
                                "vote_target": vote_target,
                                "target_player": target_player,
                            }
                        )
                    except Exception as e:
                        print(f"[MCTS] 反思结构化日志失败: {e}", file=sys.stderr)
                    response = self._generate_mcts_phase_response(
                        phase=action_phase,
                        role=role,
                        vote_target=vote_target,
                        target_player=target_player,
                        speech_style=speech_style,
                        empathy_context=empathy_context,
                        empathy_data=empathy_data,
                        game_state=game_state,
                        action=best_node.action,
                        system_prompt=system_prompt,
                        conversations=conversations,
                        task=task,
                        task_content=task_content,
                        conn_method=conn_method,
                        alives=alives,
                        agent_name=agent_name,
                        arg=arg,
                        history_messages=history_messages,
                        reflection_context=reflection_context,
                        *args,
                        **kwargs,
                    )
                    print(f"[DEBUG][FINAL_PROMPT_PHASE={action_phase}] agent={agent_name}, response_preview={str(response)[:160]}", file=sys.stderr)
                elif action_phase == "voting":
                    response = self._normalize_phase_response(
                        "voting", f"I vote to kill {vote_target}." if vote_target != "pass" else "I choose pass.", role, vote_target, agent_name, alives, task_content
                    )
                elif action_phase == "witch_save":
                    response = self._normalize_phase_response(
                        "witch_save", "Yes", role, vote_target, agent_name, alives, task_content
                    )
                elif action_phase == "witch_poison":
                    response = self._normalize_phase_response(
                        "witch_poison", f"I choose {vote_target}." if vote_target != "pass" else "I choose pass.", role, vote_target, agent_name, alives, task_content
                    )
                else:
                    response = self._normalize_phase_response(
                        "night", f"I choose {vote_target}." if vote_target != "pass" else "I choose pass.", role, vote_target, agent_name, alives, task_content
                    )
                
                print(f"[MCTS] 决策完成: action={best_node.action}, response={response}", file=sys.stderr)

                if action_phase == "discussion":
                    from ..MCTS import resolve_agent_intent, build_decision_brief
                    try:
                        from ..empathy_field import strategic_plan_from_mcts
                    except ImportError:
                        from chatarena.empathy_field import strategic_plan_from_mcts

                    tr = empathy_data.get(target_player, {})
                    intent = empathy_context.get("agent_intent") or resolve_agent_intent(
                        best_node.action, tr, role, game_state, agent_name
                    )
                    if isinstance(empathy_context, dict):
                        empathy_context["agent_intent"] = intent
                    plan = strategic_plan_from_mcts(
                        best_node.action, empathy_context, role, agent_name
                    )
                    plan["stance"] = intent.get("stance", plan.get("stance"))
                    plan["vote_lean"] = intent.get("vote_lean", plan.get("vote_lean", vote_target))
                    plan["decision_brief"] = (
                        empathy_context.get("decision_brief")
                        or build_decision_brief(intent, best_node.action, tr, role)
                    )

                    cache_entry = {
                        "vote_target": vote_target,
                        "target_player": target_player,
                        "speech_style": speech_style,
                        "stance": intent.get("stance"),
                        "vote_lean": empathy_context.get("vote_lean", intent.get("vote_lean", vote_target)),
                        "decision_brief": plan.get("decision_brief", ""),
                        "strategic_plan": plan,
                        "mcts_intent": empathy_context.get("mcts_intent", intent.get("stance")),
                    }
                    self._round_decision_cache[(agent_name, getattr(game_state, "round_no", turns))] = cache_entry
                    try:
                        from ..belief_state import BeliefStateStore
                    except ImportError:
                        from chatarena.belief_state import BeliefStateStore
                    BeliefStateStore.commit(
                        game_state,
                        agent_name,
                        intent,
                        best_node.action,
                        plan.get("decision_brief", ""),
                    )
                    if isinstance(empathy_context, dict):
                        empathy_context["decision_brief"] = plan.get("decision_brief", "")
                        empathy_context["vote_lean"] = plan.get("vote_lean", vote_target)
                        empathy_context["mcts_intent"] = intent.get("mcts_intent", intent.get("stance", ""))
                    self._pending_speech_propagate = {
                        "speaker": agent_name,
                        "plan": plan,
                        "round_no": getattr(game_state, "round_no", turns),
                        "role": role,
                    }
                    try:
                        stage_debug = {
                            "phase": action_phase,
                            "action_phase": action_phase,
                            "best_action": best_node.action,
                            "speech_style": speech_style,
                            "vote_target": vote_target,
                            "target_player": target_player,
                            "round_no": getattr(game_state, "round_no", turns),
                        }
                        self._write_structured_log(
                            arg, agent_name, role, turns, day_night, response, task, alives=alives,
                            empathy_data=empathy_data, reflection_context=reflection_context,
                            phase_name=action_phase, stage_debug=stage_debug,
                        )
                    except Exception as e:
                        print(f"[MCTS] 结构化日志记录失败: {e}", file=sys.stderr)
                
                # 确保响应不为空
                if not response or len(response.strip()) == 0:
                    print(f"[MCTS] 响应为空，使用后备方案", file=sys.stderr)
                    response = self._fallback_response(task, role, day_night, alives, agent_name)
                
            else:
                # MCTS失败时的后备方案
                print(f"[MCTS] 搜索失败，使用后备方案", file=sys.stderr)
                response = self._fallback_response(task, role, day_night, alives, agent_name)
                
        except ImportError as e:
            print(f"[MCTS] MCTS模块导入失败: {e}", file=sys.stderr)
            print(f"[MCTS] 使用后备方案", file=sys.stderr)
            # 使用后备方案
            response = self._fallback_response(task, role, day_night, alives, agent_name)
        except Exception as e:
            print(f"[MCTS] 搜索过程出错: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            # 使用后备方案
            response = self._fallback_response(task, role, day_night, alives, agent_name)
        
        # 清理响应格式
        response = self._clean_response(response, agent_name)
        
        # 最终检查：如果响应为空，做一次极简重试；仍失败才进入应急后备
        if not response or len(response.strip()) == 0:
            print(f"[MCTS] 最终检查：响应为空，尝试极简重试", file=sys.stderr)
            try:
                retry_guidance = (
                    "Return exactly one short English sentence for the current game prompt. "
                    "Do not explain. Do not add metadata. Do not output pass unless the prompt truly requires it."
                )
                retry_request = [system_prompt] + conversations + [task] + [{"role": "system", "content": retry_guidance}]
                response = self._get_response(
                    retry_request, conn_method, T=0.0, max_tokens=48,
                    speaker=agent_name, log_reply=False,
                )
                response = self._clean_response(response, agent_name)
            except Exception as e:
                print(f"[MCTS] 极简重试失败: {e}", file=sys.stderr)
                response = ""
        
        # 再次检查：如果仍然为空，才使用应急后备（仅作为最后兜底，不参与正常策略）
        if not response or len(response.strip()) == 0:
            print(f"[MCTS] 警告：所有LLM路径都失败，使用应急后备", file=sys.stderr)
            response = self._fallback_response(task, role, day_night, alives, agent_name)
            response = self._clean_response(response, agent_name)
        
        print(f"[MCTS] 最终响应: {response}", file=sys.stderr)

        if self._pending_speech_propagate and response and len(response.strip()) > 0:
            pp = self._pending_speech_propagate
            self._propagate_public_empathy_after_speech(
                pp["speaker"],
                response,
                pp["round_no"],
                plan=pp.get("plan"),
                role=pp.get("role", role),
            )
            self._pending_speech_propagate = None
        
        # 记录响应到 model_reply.log
        try:
            self._log_model_reply(agent_name, response)
        except Exception as e:
            print(f"[MCTS] 记录响应到日志失败: {e}", file=sys.stderr)
        
        return response
    
    def _build_game_state(self, agent_name, role, alives, history_messages, turns, day_night, task=None):
        """构建游戏状态对象"""
        # 辅助函数：安全获取 task 内容
        def _get_task_content(task):
            """安全获取 task 内容"""
            if task is None:
                return ""
            if not isinstance(task, dict):
                return str(task)
            return task.get("content", "")
        
        try:
            from ..MCTS import GameState
        except ImportError as e:
            print(f"[ERROR] Failed to import GameState: {e}", file=sys.stderr)
            # 创建一个简单的GameState替代品
            class GameState:
                def __init__(self, alive_players, my_role, history, votes, round_no=1, current_player=None, player_roles=None, game_phase="discussion", day_night="daytime", **kwargs):
                    self.alive_players = alive_players
                    self.my_role = my_role
                    self.history = history
                    self.votes = votes
                    self.round_no = round_no
                    self.current_player = current_player
                    self.player_roles = player_roles or {}
                    self.game_phase = game_phase
                    self.day_night = day_night
                    self.psyche = {}
                    for p in self.alive_players:
                        self.psyche[p] = {
                            "trust": 0.5,
                            "suspicion": 0.5,
                            "belief": {},
                            "emotion": {"pleasure": 0.5, "arousal": 0.5, "dominance": 0.5},
                            "influence": 1.0,
                            "susceptibility": {"logic": 0.5, "emotion": 0.5, "authority": 0.5, "consensus": 0.5, "reciprocity": 0.5, "scarcity": 0.5, "commitment": 0.5},
                            "behavior_history": [],
                            "stance_to_me": 0.0,
                            "role_probs": {}
                        }
        
        try:
            from ..MCTS import history_from_messages, get_game_analytics
        except ImportError:
            from chatarena.MCTS import history_from_messages, get_game_analytics

        task_content = _get_task_content(task) if task else ""
        history = history_from_messages(history_messages, task_content) if history_messages else []
        votes = []
        for speaker, content in history:
            if speaker.startswith("Player") and "i vote to kill" in (content or "").lower():
                m = re.search(r"i vote to kill (player \d+)", content.lower())
                if m:
                    votes.append((speaker, f"Player {m.group(1).split()[-1]}"))

        player_roles = self._infer_player_roles(alives, role, agent_name)
        
        phase = "voting" if self._is_voting_phase(task_content, day_night) else "discussion"
        if phase == "voting":
            self._empathy_field_synced_round = -1
        self._sync_public_empathy_field(history, turns)

        gs = GameState(
            alive_players=[p for p in alives if p != "pass"],
            my_role=role,
            history=history,
            votes=votes,
            round_no=turns,
            current_player=agent_name,
            player_roles=player_roles,
            game_phase=phase,
            day_night=day_night,
            public_empathy_field=self._public_empathy_field.to_dict(),
        )
        analytics = get_game_analytics(gs, force_refresh=True)
        try:
            from ..belief_state import build_and_attach_belief_state
        except ImportError:
            from chatarena.belief_state import build_and_attach_belief_state
        build_and_attach_belief_state(gs, agent_name, {}, analytics=analytics)
        return gs

    def _sync_public_empathy_field(self, history, round_no):
        """从公开历史重建 PublicEmpathyField（每轮同步一次或版本落后时）。"""
        try:
            from ..empathy_field import PublicEmpathyField, sync_field_from_history
        except ImportError:
            from chatarena.empathy_field import PublicEmpathyField, sync_field_from_history

        hist_text = " ".join(c for _, c in (history or [])).lower()
        in_vote = "voting phase" in hist_text or (history and "must vote" in (history[-1][1] or "").lower())
        if (
            self._empathy_field_synced_round == round_no
            and self._public_empathy_field.speech_acts
            and not in_vote
        ):
            return
        self._public_empathy_field = PublicEmpathyField(round_no=round_no)
        sync_field_from_history(self._public_empathy_field, history or [])
        self._empathy_field_synced_round = round_no
        print(
            f"[EmpathyField] 同步公开场: round={round_no}, speeches={len(self._public_empathy_field.speech_acts)}, "
            f"v={self._public_empathy_field.version}",
            file=sys.stderr,
        )

    def _invalidate_empathy_cache_for_round(self, round_no):
        keys = [k for k in self._empathy_cache if k[1] == round_no]
        for k in keys:
            del self._empathy_cache[k]
        if keys:
            print(f"[EmpathyField] 失效共情缓存 {len(keys)} 条 (round={round_no})", file=sys.stderr)

    def _propagate_public_empathy_after_speech(
        self,
        speaker,
        text,
        round_no,
        plan=None,
        action=None,
        empathy_context=None,
        role="",
    ):
        """发言后更新公开共情场，并失效当轮共情缓存。"""
        try:
            from ..empathy_field import (
                UtteranceEffectModel,
                strategic_plan_from_mcts,
                merge_plan_and_text,
            )
        except ImportError:
            from chatarena.empathy_field import (
                UtteranceEffectModel,
                strategic_plan_from_mcts,
                merge_plan_and_text,
            )

        if plan is None and action is not None:
            plan = strategic_plan_from_mcts(action, empathy_context, role, speaker)
        speech_act = UtteranceEffectModel.propagate(
            self._public_empathy_field, speaker, text, plan=plan
        )
        self._empathy_field_synced_round = -1
        self._invalidate_empathy_cache_for_round(round_no)
        print(
            f"[EmpathyField] Propagate {speaker}: intent={speech_act.get('intent')}, "
            f"target={speech_act.get('target')}, claims={len(speech_act.get('claims', []))}",
            file=sys.stderr,
        )
        return speech_act
    
    def _infer_player_roles(self, alives, my_role, agent_name):
        """推理玩家角色（基于游戏规则和观察）"""
        # 确保 my_role 不为空
        if not my_role or my_role == "":
            print(f"[WARNING] _infer_player_roles: my_role为空，使用默认值", file=sys.stderr)
            my_role = "villager"  # 默认角色
        
        player_roles = {agent_name: my_role}
        
        # 为其他玩家分配默认角色（在实际游戏中这些信息是未知的）
        # 但为了MCTS能够正常工作，我们需要给其他玩家分配合理的默认角色概率
        for player in alives:
            if player != agent_name and player != "pass":
                # 使用 "unknown" 表示角色未知，但MCTS会基于行为推断
                player_roles[player] = "unknown"  # 实际游戏中角色未知
        
        print(f"[MCTS] _infer_player_roles: agent={agent_name}, my_role={my_role}, player_roles={player_roles}", file=sys.stderr)
        return player_roles
    
    def _is_voting_phase(self, task_content, day_night):
        """判断是否为投票阶段"""
        # 安全检查：确保 task_content 不为 None
        if task_content is None:
            task_content = ""

        task_lower = str(task_content).lower()

        # 明确的投票阶段标识
        voting_indicators = [
            "this is the voting phase",
            "you must vote now",
            "voting is mandatory",
            "choose which of the players should be voted for killing",
            "your response must be exactly: 'i vote to kill",
            "you only choose one from the following living options"
        ]

        # 明确的讨论阶段标识
        discussion_indicators = [
            "this is the discussion phase",
            "do not vote now",
            "only discuss and analyze",
            "freely talk about roles",
            "based on your observation and reflection",
            "consider revealing your identity"
        ]

        # 首先检查是否明确标识为讨论阶段
        for indicator in discussion_indicators:
            if indicator in task_lower:
                return False

        # 然后检查是否明确标识为投票阶段
        for indicator in voting_indicators:
            if indicator in task_lower:
                return True

        # 夜晚阶段的特殊处理
        if day_night == "night":
            night_action_indicators = [
                "who you protect tonight",
                "who are you going to kill tonight",
                "who are you going to verify",
                "do you want to save",
                "who are you going to poison"
            ]
            for indicator in night_action_indicators:
                if indicator in task_lower:
                    return True

        # 默认情况：如果无法明确判断，根据关键词判断
        # 但要更严格，避免误判
        if ("vote" in task_lower and "must" in task_lower) or ("choose" in task_lower and "kill" in task_lower):
            return True

        return False

    def _track_llm_call(self, agent_name, call_type, round_no):
        self._llm_call_log.append({"agent": agent_name, "type": call_type, "round": round_no})
        print(
            f"[LLM] #{len(self._llm_call_log)} {agent_name} round={round_no} phase={call_type}",
            file=sys.stderr,
        )

    def _detect_action_phase(self, task_content, day_night, role):
        task_lower = (task_content or "").lower()
        if day_night == "night":
            if role == "witch":
                # Hard phase routing: only explicit moderator prompts unlock witch sub-phases.
                if ("antidote" in task_lower and ("save" in task_lower or "do you want" in task_lower)):
                    return "witch_save"
                if "poison" in task_lower and ("who are you going to kill" in task_lower or "who are you going to poison" in task_lower):
                    return "witch_poison"
                # First night / ambiguous night prompts must never fall through to poison.
                return "night"
            if role in ("werewolf", "wolf", "guard", "seer"):
                return "night"
        if self._is_voting_phase(task_content, day_night):
            return "voting"
        return "discussion"

    def _build_werewolf_meta_knowledge(self, role: str, round_no: int) -> str:
        """对抗性认知底座：只提供世界观，不给结论或发言模板。"""
        return f"""## Anti-adversarial game frame
This is a hidden-information social deduction game with two opposing win conditions.
Players may speak to reveal facts, conceal facts, mislead, probe, or test others.
Public truth can change quickly when verifiable claims appear, but those claims can also be faked or attacked.
A useful analysis should distinguish hard evidence, soft reads, uncertainty, and incentive-driven speech.
Round {round_no}; your role: {role}. Reason only from this game state.
"""

    def _build_voting_decision_context(self, role, vote_target, empathy_data, target_player, empathy_context, alives, round_no=1, prior_decision=None):
        from ..MCTS import format_empathy_for_speech

        alive_list = ", ".join(p for p in alives if p != "pass")
        summary = self._summarize_empathy_for_prompt(empathy_data, target_player)
        hints = format_empathy_for_speech(empathy_data, target_player)
        decision_brief = (empathy_context or {}).get("decision_brief", "")
        vote_lean = (empathy_context or {}).get("vote_lean", vote_target)

        guidance = (
            self._build_werewolf_meta_knowledge(role, round_no)
            + "\n## Voting decision\n"
            f"Choose the most consistent vote target.\n"
            f"Current algorithm target: {vote_target}\n"
            f"Integrated vote lean: {vote_lean}\n"
            f"Alive: {alive_list}\n"
        )
        if prior_decision and prior_decision.get("decision_brief"):
            guidance += f"Earlier stance: {prior_decision['decision_brief']}\n"
        elif decision_brief:
            guidance += f"Integrated stance: {decision_brief}\n"
        if summary:
            guidance += f"Target read: {summary}\n"
        if hints:
            guidance += f"Signals:\n{hints}\n"
        return guidance

    def _build_soft_strategy_context(self, role, round_no, discussion_ctx, private_knowledge, decision_brief=""):
        """算法生成的软性策略上下文（非强制命令）"""
        spoke = discussion_ctx.get("already_spoke", [])
        yet = discussion_ctx.get("yet_to_speak", [])
        parts = [
            f"Speaking order: {discussion_ctx.get('position', 'your turn')}.",
            f"Already spoke: {', '.join(spoke) or 'none'}.",
            f"Not yet spoken: {', '.join(yet) or 'none'}.",
            "You have one speech this round before voting.",
        ]
        if private_knowledge:
            parts.append("Private facts you hold: " + "; ".join(private_knowledge))
        if decision_brief:
            parts.append(f"Integrated analysis: {decision_brief}")
        role_hints = {
            "seer": "Verification results are high-value if shared when the village is uncertain.",
            "witch": "Save/poison context can resolve peaceful-night confusion when shared.",
            "guard": "Protection info can explain no-death nights when shared.",
            "villager": "Without private info, comparing claims often beats repeating general observations.",
            "werewolf": "Blending in as a claim-seeking villager is typical wolf play.",
        }
        hint = role_hints.get(role) or role_hints.get("werewolf" if role in ("wolf",) else "villager")
        if hint:
            parts.append(hint)
        return "\n".join(parts)

    def _extract_discussion_context(self, task_content, agent_name, history_messages, alives):
        """分析白天讨论的发言顺位与本轮已发言玩家"""
        task_lower = (task_content or "").lower()
        alive = [p for p in alives if p != "pass"]

        already_spoke = []
        if history_messages:
            in_discussion = False
            for msg in history_messages:
                speaker = getattr(msg, "agent_name", "")
                content = getattr(msg, "content", "") or ""
                cl = content.lower()
                if "discussion phase" in cl or "freely talk" in cl:
                    in_discussion = True
                    continue
                if "voting phase" in cl or "vote to kill" in cl and "must vote" in cl:
                    in_discussion = False
                if in_discussion and speaker.startswith("Player") and speaker != "Moderator":
                    if speaker not in already_spoke:
                        already_spoke.append(speaker)

        yet_to_speak = [p for p in alive if p not in already_spoke and p != agent_name]

        if re.search(rf"first\s+{re.escape(agent_name)}", task_content or "", re.I):
            position = "FIRST speaker this round"
        elif "the next" in task_lower and agent_name.lower() in task_lower:
            position = "MIDDLE/LATE speaker (others already spoke)"
        else:
            position = "Your turn in today's discussion order"

        return {
            "position": position,
            "already_spoke": already_spoke,
            "yet_to_speak": yet_to_speak,
            "only_one_speech": True,
        }

    def _extract_private_knowledge(self, role, history_messages, agent_name):
        """从可见历史中抽取本角色私有信息（验人、救人等）"""
        facts = []
        if not history_messages:
            return facts

        for msg in history_messages:
            content = (getattr(msg, "content", "") or "").strip()
            cl = content.lower()

            if role == "seer":
                m = re.search(r"(Player \d+) is not a werewolf", content, re.I)
                if m:
                    facts.append(f"You VERIFIED {m.group(1)} is GOOD (not a werewolf).")
                m = re.search(r"(Player \d+) is a werewolf", content, re.I)
                if m:
                    facts.append(f"You VERIFIED {m.group(1)} is a WEREWOLF.")

            if role == "witch":
                if "will die tonight" in cl and "save" in cl:
                    m = re.search(r"(Player \d+) was attacked", content, re.I)
                    if m:
                        facts.append(f"Tonight's wolf target was {m.group(1)} (you may have saved them).")
                if agent_name in content and "was attacked" in cl:
                    facts.append("You were the wolf target at some point.")

            if role == "guard":
                if "protect" in cl and msg.agent_name == agent_name:
                    facts.append(f"You previously stated: {content[:120]}")

            if role in ("werewolf", "wolf"):
                if "werewolves" in cl and "are all of" in cl:
                    facts.append(f"Teammate info from moderator (private): {content[:200]}")

        return list(dict.fromkeys(facts))

    def _build_causal_chain_evidence(self, empathy_data, target_player, role, game_state, history_messages=None):
        """从共情建模与公开历史中提炼发言前的证据链。"""
        evidence = []
        if not isinstance(empathy_data, dict):
            empathy_data = {}
        reports = empathy_data.get("player_reports", empathy_data)
        board = empathy_data.get("_game", {}) if isinstance(empathy_data, dict) else {}

        if target_player and isinstance(reports, dict):
            rep = reports.get(target_player, {}) or {}
            if rep:
                evidence.append(
                    f"Target {target_player}: hard_wolf={float(rep.get('hard_wolf_prob', 0.0)):.2f}, "
                    f"soft_wolf={float(rep.get('soft_wolf_prob', 0.0)):.2f}, pub_trust={float(rep.get('public_trust', rep.get('trust_score', 0.5))):.2f}, "
                    f"info_gain={float(rep.get('information_gain', 0.0)):.2f}, vote_pressure={float(rep.get('current_round_vote_pressure', 0.0)):.2f}"
                )
                if rep.get("supports") or rep.get("supported_by"):
                    evidence.append(
                        f"Support links: supports={','.join(rep.get('supports', [])[:3]) or 'none'}, "
                        f"supported_by={','.join(rep.get('supported_by', [])[:3]) or 'none'}"
                    )
                if rep.get("clears") or rep.get("cleared_by"):
                    evidence.append(
                        f"Clear links: clears={','.join(rep.get('clears', [])[:3]) or 'none'}, "
                        f"cleared_by={','.join(rep.get('cleared_by', [])[:3]) or 'none'}"
                    )
                if rep.get("semantic_memory"):
                    evidence.append(f"Semantic read: {str(rep.get('semantic_memory'))[:180]}")
                if rep.get("uncertainty_notes"):
                    evidence.append(f"Uncertainty note: {str(rep.get('uncertainty_notes'))[:160]}")
                if rep.get("signal"):
                    sig = rep.get("signal", {})
                    evidence.append(
                        f"Signal: claim={sig.get('claim','')[:80]}, vote={sig.get('vote','')[:60]}, tone={sig.get('tone','')[:30]}, timing={sig.get('timing','')[:30]}"
                    )

        if board:
            if board.get("top_suspects"):
                top = board.get("top_suspects", [])[:3]
                evidence.append("Board suspects: " + ", ".join(
                    f"{x.get('player')}({float(x.get('hard_wolf_prob', 0.0)):.2f}/{float(x.get('information_gain', 0.0)):.2f})"
                    for x in top
                ))
            if board.get("board_signal"):
                bs = board.get("board_signal", {})
                evidence.append(
                    f"Board signal: info_dense={bs.get('info_dense', 0)}, hard_claims={bs.get('hard_claims', 0)}, "
                    f"support_links={bs.get('support_links', 0)}, accusation_links={bs.get('accusation_links', 0)}"
                )

        if history_messages:
            recent = history_messages[-6:]
            for msg in recent:
                speaker = getattr(msg, "agent_name", "")
                if speaker == "Moderator":
                    continue
                content = (getattr(msg, "content", "") or "").strip()
                if not content:
                    continue
                if target_player and target_player in content:
                    evidence.append(f"Recent mention of {target_player} by {speaker}: {content[:140]}")

        return list(dict.fromkeys([e for e in evidence if e]))

    def _build_causal_chain_from_evidence(self, role, evidence, reflection_context=None):
        """将证据压缩为可用于发言的因果链骨架。"""
        meaning = []
        strategy = []
        if not evidence:
            return {
                "evidence": ["No strong evidence available."],
                "meaning": ["The game state is still ambiguous."],
                "strategy": ["Ask for concrete claims and avoid overcommitting."],
            }

        joined = " ".join(evidence).lower()
        if "hard_wolf" in joined or "verified" in joined:
            meaning.append("There is at least one concrete alignment signal that should affect trust.")
        if "support" in joined or "clears" in joined:
            meaning.append("Relation evidence may explain why some players are protecting or reinforcing others.")
        if "accusation" in joined or "vote_pressure" in joined:
            meaning.append("The current round is already shaping a pressure target.")
        if "no one died" in joined or "peaceful" in joined:
            meaning.append("A peaceful night is informative but ambiguous; it can result from protection, antidote, or wolf coordination failure.")
        if role == "seer":
            strategy.append("Use verified information to narrow the vote and make the information actionable.")
        elif role == "guard":
            strategy.append("Use protection implications to explain or challenge the night outcome without overexposing yourself.")
        elif role == "witch":
            strategy.append("Use save/poison knowledge to add concrete constraints to the discussion.")
        elif role == "werewolf":
            strategy.append("Exploit ambiguity, avoid direct contradictions, and redirect pressure with plausible alternatives.")
        else:
            strategy.append("Force the discussion toward concrete contradictions, not generic suspicion.")
        if reflection_context and isinstance(reflection_context, dict):
            nxt = str(reflection_context.get("what_to_do_next", "")).strip()
            if nxt:
                strategy.append(nxt[:180])
        if not meaning:
            meaning.append("The evidence does not yet justify a hard conclusion.")
        return {"evidence": evidence[:5], "meaning": meaning[:3], "strategy": strategy[:3]}

    def _build_speech_guidance(
        self, role, empathy_context, game_state, empathy_data, action,
        task_content="", agent_name="", history_messages=None, alives=None,
        reflection_context=None,
    ):
        """讨论阶段发言指导：算法决策摘要 + 共情策略（软性引导）"""
        from ..MCTS import format_empathy_for_speech, get_game_analytics, format_game_analytics_brief, build_empathy_reports, enrich_empathy_reports, merge_empathy_reports

        vote_target, target_player, speech_style = action
        round_no = getattr(game_state, "round_no", 1)
        alive_list = alives or getattr(game_state, "alive_players", [])
        alive_list = [p for p in alive_list if p != "pass"]

        decision_brief = (empathy_context or {}).get("decision_brief", "")
        evidence_chain = self._build_causal_chain_evidence(empathy_data, target_player, role, game_state, history_messages)
        causal_chain = self._build_causal_chain_from_evidence(role, evidence_chain, reflection_context)
        if reflection_context:
            reflection_hint = reflection_context.get("reflection_hint", "")
            if reflection_hint:
                decision_brief = (decision_brief + "\n" + reflection_hint).strip()

        guidance = self._build_werewolf_meta_knowledge(role, round_no)
        if decision_brief:
            guidance += f"\nDecision seed: {decision_brief[:160]}\n"
        if causal_chain:
            guidance += "\nEvidence chain:\n"
            guidance += "- evidence: " + " | ".join(causal_chain.get("evidence", [])[:4]) + "\n"
            guidance += "- meaning: " + " | ".join(causal_chain.get("meaning", [])[:3]) + "\n"
            guidance += "- strategy: " + " | ".join(causal_chain.get("strategy", [])[:3]) + "\n"

        discussion_ctx = self._extract_discussion_context(
            task_content, agent_name, history_messages, alive_list
        )
        private_knowledge = self._extract_private_knowledge(role, history_messages, agent_name)
        if discussion_ctx:
            try:
                ctx_position = discussion_ctx.get("position", "")
                already_spoke = discussion_ctx.get("already_spoke", [])
                yet_to_speak = discussion_ctx.get("yet_to_speak", [])
                guidance += (
                    f"\nContext: position={ctx_position}; already_spoke={', '.join(already_spoke[:4]) if isinstance(already_spoke, list) else already_spoke}; "
                    f"yet_to_speak={', '.join(yet_to_speak[:4]) if isinstance(yet_to_speak, list) else yet_to_speak}\n"
                )
            except Exception:
                guidance += f"\nContext: {str(discussion_ctx)[:140]}\n"
        if private_knowledge:
            guidance += f"\nPrivate: {'; '.join(private_knowledge[:2])}\n"

        if reflection_context:
            guidance += "\nReflection clues:\n"
            guidance += f"- what_i_know: {str(reflection_context.get('what_i_know', reflection_context.get('good_view','')))[:140]}\n"
            guidance += f"- what_might_be_fake: {str(reflection_context.get('what_might_be_fake', reflection_context.get('wolf_view','')))[:140]}\n"
            guidance += f"- what_conflicts_exist: {str(reflection_context.get('what_conflicts_exist', reflection_context.get('shared_uncertainties','')))[:140]}\n"
            guidance += f"- what_to_do_next: {str(reflection_context.get('what_to_do_next', reflection_context.get('reflection_hint','')))[:140]}\n"
            support_summary = reflection_context.get('support_summary', '')
            if support_summary:
                guidance += f"- support_summary: {support_summary[:160]}\n"
        if empathy_data:
            board = empathy_data.get("_game", {}) if isinstance(empathy_data, dict) else {}
            if board:
                top_suspects = board.get("top_suspects", [])[:3]
                top_trust = board.get("top_trust", [])[:2]
                board_signal = board.get("board_signal", {})
                if top_suspects:
                    guidance += "\nBoard top suspects: " + ", ".join(
                        f"{x.get('player')}[h={float(x.get('hard_wolf_prob', 0.0)):.2f},info={float(x.get('information_gain', 0.0)):.2f},act={x.get('recommended_action', 'observe')}]"
                        for x in top_suspects
                    ) + "\n"
                if top_trust:
                    guidance += "Board trusted / info chain: " + ", ".join(
                        f"{x.get('player')}[pub={float(x.get('public_trust', 0.0)):.2f},trust={float(x.get('trust_score', 0.0)):.2f}]"
                        for x in top_trust
                    ) + "\n"
                if board_signal:
                    guidance += (
                        f"Board signal: info_dense={board_signal.get('info_dense', 0)}, hard_claims={board_signal.get('hard_claims', 0)}, "
                        f"support_links={board_signal.get('support_links', 0)}, accusation_links={board_signal.get('accusation_links', 0)}\n"
                    )
            target_key = target_player if target_player in empathy_data else None
            if target_key:
                guidance += f"\nTarget empathy summary: {self._summarize_empathy_for_prompt(empathy_data, target_key)}\n"

        guidance += (
            "\nBefore speaking, internally build a short causal chain from the evidence above: evidence -> meaning -> strategy -> speech. "
            "Do not output the chain explicitly; use it to shape your final speech. "
            "Then write 2-4 fluent English sentences as a natural player would. "
            "Use the evidence above to form your own view; avoid bullet-point style or meta commentary. "
            "Do not mention internal reasoning labels. "
            "Prefer concrete references to players, votes, claims, contradictions, timing, and support/clear evidence when possible. "
            "If you are a special role, whether to reveal is a strategic choice, not a template requirement. "
            "If the board snapshot shows many info_dense signals or hard claims, explicitly mention which players are worth pressuring or trusting."
        )
        return guidance

    def _generate_counterfactual_reflection(
        self,
        role,
        game_state,
        empathy_data,
        action,
        system_prompt,
        conversations,
        task,
        task_content,
        conn_method,
        alives,
        agent_name,
        history_messages=None,
        empathy_context=None,
        *args,
        **kwargs,
    ):
        """第二次LLM调用：双视角反思链输出，不直接生成最终发言。"""
        round_no = getattr(game_state, "round_no", 1)
        vote_target, target_player, speech_style = action
        target_report = empathy_data.get(target_player, {}) if empathy_data else {}
        from ..MCTS import get_game_analytics, format_game_analytics_brief

        guidance = self._build_werewolf_meta_knowledge(role, round_no)
        analytics = get_game_analytics(game_state)
        brief = format_game_analytics_brief(analytics, agent_name, empathy_data)
        if brief:
            guidance += "\n" + brief
        guidance += "\n## Reflection task\n"
        guidance += f"- role: {role}\n"
        guidance += f"- focus_player: {target_player}\n"
        guidance += f"- speech_style: {speech_style}\n"
        guidance += f"- target_wolf_prob: {target_report.get('hard_wolf_prob', 0):.2f}/{target_report.get('soft_wolf_prob', 0):.2f}\n"
        if empathy_context:
            guidance += f"- empathy_decision_brief: {empathy_context.get('decision_brief', '')}\n"
        support_summary = ""
        if empathy_context:
            support_summary = empathy_context.get("support_summary", "")
            if support_summary:
                guidance += f"- support_summary: {support_summary}\n"
        guidance += (
            "\nProduce ONLY valid JSON in English with keys: what_i_know, what_might_be_fake, what_conflicts_exist, what_to_do_next, support_summary. "
            "Each value must be concise and different across the two views when possible. "
            "Do not write the final speech. Do not give a direct vote instruction. "
            "If uncertain, put the uncertainty in what_conflicts_exist rather than guessing. "
            "what_might_be_fake must describe how a wolf could exploit or distort the current situation, not repeat what_i_know. "
            "support_summary must explicitly note any support/clear relation and whether it may be deceptive. "
            "Internally reason in terms of evidence -> meaning -> action before writing the JSON."
        )

        request = [system_prompt] + conversations + [task] + [{"role": "system", "content": guidance}]
        print(f"[DEBUG][REFLECTION_PROMPT] agent={agent_name}, preview={guidance[:260]}", file=sys.stderr)
        raw = self._get_response(
            request, conn_method, T=0.25, max_tokens=260,
            speaker=agent_name, log_reply=False,
        )
        self._track_llm_call(agent_name, "reflection", round_no)

        text = self.extract_text(raw).strip()
        text = re.sub(rf"{END_OF_MESSAGE}$", "", text).strip()
        parsed = {}
        try:
            import json
            parsed = json.loads(text)
        except Exception:
            try:
                import json, re as _re
                m = _re.search(r"\{[\s\S]*\}", text)
                if m:
                    parsed = json.loads(m.group(0))
            except Exception:
                parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        what_i_know = str(parsed.get("what_i_know", parsed.get("good_view", text))).strip()
        what_might_be_fake = str(parsed.get("what_might_be_fake", parsed.get("wolf_view", text))).strip()
        what_conflicts_exist = str(parsed.get("what_conflicts_exist", parsed.get("shared_uncertainties", text))).strip()
        what_to_do_next = str(parsed.get("what_to_do_next", parsed.get("reflection_hint", text))).strip()
        support_summary = str(parsed.get("support_summary", "")).strip()
        return {
            "raw": text,
            "what_i_know": what_i_know,
            "what_might_be_fake": what_might_be_fake,
            "what_conflicts_exist": what_conflicts_exist,
            "what_to_do_next": what_to_do_next,
            "support_summary": support_summary,
            "good_view": what_i_know,
            "wolf_view": what_might_be_fake,
            "shared_uncertainties": what_conflicts_exist,
            "reflection_hint": what_to_do_next,
        }

    def _get_empathy_for_round(self, game_state, agent_name, arg, _skip_empathy=False):
        """每轮每个 agent 仅调用一次 LLM 共情分析，其余阶段复用缓存。"""
        from ..MCTS import build_empathy_reports, merge_empathy_reports, enrich_empathy_reports

        round_no = getattr(game_state, "round_no", None)
        if round_no is None and isinstance(game_state, dict):
            round_no = game_state.get("round_no", 0)

        if _skip_empathy:
            rule_reports = build_empathy_reports(game_state, agent_name, None)
            rule_reports = enrich_empathy_reports(game_state, agent_name, rule_reports)
            return self._normalize_empathy_supports(rule_reports, game_state, agent_name)

        try:
            public_field_version = getattr(game_state, "public_empathy_field", {}).get("version", 0) if isinstance(game_state, dict) else getattr(getattr(game_state, "public_empathy_field", None), "version", 0)
        except Exception:
            public_field_version = 0

        cache_key = (agent_name, round_no, public_field_version)
        if cache_key in self._empathy_cache:
            print(f"[MCTS] 复用共情缓存: {agent_name} round={round_no} version={public_field_version}", file=sys.stderr)
            return self._empathy_cache[cache_key]

        print(f"[MCTS] LLM共情分析: {agent_name} round={round_no} version={public_field_version}", file=sys.stderr)
        llm_reports = self._internal_empathy_extract(game_state, agent_name, arg, [], None)
        rule_reports = build_empathy_reports(game_state, agent_name, None)
        empathy_data = merge_empathy_reports(llm_reports, rule_reports) or rule_reports
        empathy_data = enrich_empathy_reports(game_state, agent_name, empathy_data)
        empathy_data = self._normalize_empathy_supports(empathy_data, game_state, agent_name)

        self._empathy_cache[cache_key] = empathy_data
        self._track_llm_call(agent_name, "empathy", round_no)
        return empathy_data

    def _summarize_empathy_for_prompt(self, empathy_data, target_player):
        if not empathy_data or not target_player:
            return ""
        reports = empathy_data.get("player_reports", empathy_data) if isinstance(empathy_data, dict) else {}
        report = reports.get(target_player, {}) if isinstance(reports, dict) else {}
        if not report:
            return ""
        board = empathy_data.get("_game", {}) if isinstance(empathy_data, dict) else {}
        rp = report.get("role_probability", {}) if isinstance(report.get("role_probability", {}), dict) else {}
        hard = float(report.get("hard_wolf_prob", 0.0))
        soft = float(report.get("soft_wolf_prob", rp.get("werewolf", 0.3)))
        supports = report.get("supports", []) or []
        supported_by = report.get("supported_by", []) or []
        clears = report.get("clears", []) or []
        cleared_by = report.get("cleared_by", []) or []
        support_uncertainty = float(report.get("support_uncertainty", 0.5))
        clear_uncertainty = float(report.get("clear_uncertainty", 0.5))
        semantic = str(report.get("semantic_memory", ""))[:90]
        notes = str(report.get("uncertainty_notes", ""))[:90]
        relation_index = report.get("relation_index", []) or []
        reflection_sketch = report.get("reflection_sketch", {}) if isinstance(report.get("reflection_sketch", {}), dict) else {}
        info_gain = float(report.get("information_gain", 0.0))
        public_trust = float(report.get("public_trust", report.get("trust_score", 0.5)))
        vote_pressure = float(report.get("current_round_vote_pressure", 0.0))
        vote_consistency = float(report.get("speech_vote_consistency", 1.0))
        claim_type = str(report.get("claim_type", ""))
        claim_target = str(report.get("claim_target", ""))
        evidence_tags = report.get("evidence_tags", []) or []
        misdirection = float(report.get("misdirection_risk", 0.0))
        top_suspects = board.get("top_suspects", [])[:3] if isinstance(board, dict) else []
        board_signal = board.get("board_signal", {}) if isinstance(board, dict) else {}
        suspect_hint = ""
        if top_suspects:
            suspect_hint = "; top=" + ", ".join(
                f"{x.get('player')}({float(x.get('hard_wolf_prob', 0.0)):.2f}/{float(x.get('information_gain', 0.0)):.2f})"
                for x in top_suspects
            )
        board_hint = ""
        if board_signal:
            board_hint = f"; board=info{int(board_signal.get('info_dense', 0))}/hard{int(board_signal.get('hard_claims', 0))}/sup{int(board_signal.get('support_links', 0))}/acc{int(board_signal.get('accusation_links', 0))}"
        trend = []
        if hard >= 0.65:
            trend.append("TARGET_NOW")
        elif hard >= 0.42:
            trend.append("DANGER")
        else:
            trend.append("LEAN_SAFE")
        if info_gain >= 0.30:
            trend.append("INFO_RICH")
        elif info_gain <= 0.08:
            trend.append("INFO_POOR")
        if public_trust >= 0.45:
            trend.append("PUBLIC_TRUSTED")
        elif public_trust <= 0.18:
            trend.append("PUBLIC_WEAK")
        if vote_pressure >= 0.35:
            trend.append("PRESSED")
        if misdirection >= 0.25:
            trend.append("MISDIRECTIVE")
        if supported_by:
            trend.append(f"ALLY:{','.join(supported_by[:1])}")
        if support_uncertainty >= 0.65:
            trend.append("ALLY_UNCERTAIN")
        if clear_uncertainty >= 0.65:
            trend.append("CLEAR_UNCERTAIN")
        action_hint = str(report.get("recommended_action", "observe")).upper()
        action_word = {
            "accuse": "PRESS",
            "support": "SPEAK-UP",
            "reveal": "REVEAL",
            "question": "QUESTION",
            "observe": "WATCH",
        }.get(str(report.get("recommended_action", "observe")).lower(), "WATCH")
        reflection_hint = str(reflection_sketch.get("what_to_do_next", report.get("recommended_action", "observe")))[:24]
        return (
            f"[{target_player}] {action_word}/{action_hint} | wolf={hard:.2f}/{soft:.2f} | info={info_gain:.2f} | pub={public_trust:.2f} | "
            f"press={vote_pressure:.2f} | align={vote_consistency:.2f} | mis={misdirection:.2f} | "
            f"rel={','.join(relation_index[:2]) or 'none'} | tags={','.join(str(x) for x in evidence_tags[:2]) or 'none'} | "
            f"claim={claim_type or 'none'}:{claim_target or '-'} | trend={'/'.join(trend)} | "
            f"next={reflection_hint}{suspect_hint}{board_hint}"
        )

    def _infer_llm_preferred_target(self, role, alives, agent_name, game_state, empathy_data, action_phase, task_content):
        """Compute a conservative LLM-preferred target without using MCTS as the primary driver."""
        alive = [p for p in (alives or []) if p not in ("pass", agent_name)]
        if not alive:
            return "pass"

        reports = empathy_data.get("player_reports", empathy_data) if isinstance(empathy_data, dict) else {}
        candidates = []
        for p in alive:
            rep = reports.get(p, {}) if isinstance(reports, dict) else {}
            hard = float(rep.get("hard_wolf_prob", rep.get("role_probability", {}).get("werewolf", 0.0) if isinstance(rep.get("role_probability", {}), dict) else 0.0))
            info = float(rep.get("information_gain", 0.0))
            pub = float(rep.get("public_trust", rep.get("trust_score", 0.5)))
            press = float(rep.get("current_round_vote_pressure", 0.0))
            score = hard * 0.55 + info * 0.15 + press * 0.15 - pub * 0.10
            if role == "werewolf" and p in alive:
                score = info * 0.10 + press * 0.10 - pub * 0.05
            candidates.append((score, p))
        candidates.sort(reverse=True)
        if action_phase == "voting":
            return candidates[0][1]
        if role == "guard":
            return candidates[0][1]
        if role == "seer":
            return candidates[0][1]
        if role == "werewolf":
            return candidates[0][1]
        return candidates[0][1]

    def _build_night_action_guidance(self, role, vote_target, empathy_data, target_player, empathy_context, game_state=None):
        summary = self._summarize_empathy_for_prompt(empathy_data, target_player)
        round_no = getattr(game_state, "round_no", 1) if game_state is not None else 1
        base = (
            "CRITICAL: Respond in ENGLISH ONLY. This is a night action prompt, not discussion. Output exactly one action line.\n"
            f"Round: {round_no}\n"
            f"MCTS recommended target: {vote_target}\n"
        )
        if summary:
            base += f"Empathy: {summary}\n"
        if empathy_context:
            base += f"Mode: {empathy_context.get('emotional_approach', 'action')}\n"

        if role in ("werewolf", "wolf"):
            base += "You are selecting a kill target. Choose a living non-teammate. Ignore MCTS if it conflicts with legality or strong inference. Never choose pass unless no valid target exists. "
            base += f"Output format: 'I vote to kill <Player X>'."
        elif role == "guard":
            base += "You are selecting a protection target. Prefer the most valuable living non-self target. Ignore MCTS if it conflicts with legality or strong inference. Never choose pass unless no valid target exists. "
            base += f"Output format: 'I protect <Player X>'."
        elif role == "seer":
            base += "You are selecting an investigation target. Prefer the highest-value living non-self target. Ignore MCTS if it conflicts with legality or strong inference. Never choose pass unless no valid target exists. "
            base += f"Output format: 'I verify <Player X>'."
        elif role == "witch":
            if round_no <= 1:
                base += "FIRST NIGHT RULE: For poison, always pass. If asked to save, answer Yes or No."
            else:
                base += (
                    "Witch poison is scarce and should only be used when your own reasoning strongly supports it. Ignore MCTS unless it agrees with a high-confidence judgment. "
                    "Never poison just because a reference target was suggested. "
                )
                base += f"Output format: 'I choose <Player X>.' or 'I choose pass.'"
        else:
            base += "You have no night action. Output pass unless the prompt explicitly asks for a different action."
        return base

    def _build_witch_save_guidance(self, vote_target, empathy_data, target_player):
        summary = self._summarize_empathy_for_prompt(empathy_data, target_player)
        guidance = "CRITICAL: Reply with ONLY 'Yes' or 'No'. No explanation.\n"
        if summary:
            guidance += f"Empathy context: {summary}\n"
        guidance += (
            f"MCTS hint: {'Yes' if vote_target and vote_target != 'pass' else 'No'}\n"
            "Strategy override: if the target is not overwhelmingly suspicious, the correct choice is No."
        )
        return guidance

    def _build_witch_poison_guidance(self, vote_target, empathy_data, target_player, alives, game_state=None):
        alive_list = ", ".join(p for p in alives if p != "pass")
        summary = self._summarize_empathy_for_prompt(empathy_data, target_player)
        round_no = getattr(game_state, "round_no", 1) if game_state is not None else 1
        guidance = (
            "CRITICAL: Respond in ENGLISH ONLY. Witch poison phase.\n"
            f"Round: {round_no}\n"
            f"MCTS recommended target: {vote_target}\n"
            f"Alive: {alive_list}\n"
            "Strategic reminder: preserve poison unless you have strong evidence; avoid wasting poison on low-confidence targets.\n"
            "If the evidence is weak or the target is already publicly trusted, answer pass even if a target is proposed.\n"
        )
        if summary:
            guidance += f"Empathy: {summary}\n"
        guidance += f"Output EXACTLY: 'I choose {vote_target}.' or 'I choose pass.'"
        return guidance

    def _normalize_phase_response(self, phase, response, role, mcts_target, agent_name, alives, task_content=""):
        text = self.extract_text(response)
        text = re.sub(rf"{END_OF_MESSAGE}$", "", text).strip()
        text = re.sub(rf"^\s*(\[)?[a-zA-Z0-9\s]*(\])?:\s*", "", text).strip()

        if phase == "witch_save":
            return "Yes" if text.lower().startswith("y") else "No"

        if phase == "witch_poison":
            if role == "witch" and (mcts_target == "pass" or not mcts_target):
                return "I choose pass."
            m = re.search(r"I choose (Player \d+|pass)", text, re.I)
            if m:
                chosen = m.group(1)
                return f"I choose {chosen if chosen.lower() != 'pass' else 'pass'}."
            if mcts_target and mcts_target != "pass" and mcts_target != agent_name:
                return "I choose pass."
            return self._generate_night_action(role, mcts_target, task_content, agent_name)

        if phase == "night":
            patterns = {
                "werewolf": r"I vote to kill (Player \d+|pass)",
                "wolf": r"I vote to kill (Player \d+|pass)",
                "guard": r"I protect (Player \d+|pass)",
                "seer": r"I verify (Player \d+|pass)",
            }
            pat = patterns.get(role)
            if pat:
                m = re.search(pat, text, re.I)
                if m:
                    target = m.group(1)
                    if target.lower() == "pass":
                        if mcts_target and mcts_target != "pass":
                            target = mcts_target
                        else:
                            return "I choose pass."
                    prefix = {"werewolf": "I vote to kill", "wolf": "I vote to kill", "guard": "I protect", "seer": "I verify"}[role]
                    return f"{prefix} {target}."
            return self._generate_night_action(role, mcts_target, task_content, agent_name)

        if phase == "voting":
            m = re.search(r"I vote to kill (Player \d+|pass)", text, re.I)
            if m:
                target = m.group(1)
                if target.lower() == "pass":
                    target = mcts_target if mcts_target and mcts_target != "pass" else "pass"
                return f"I vote to kill {target}." if target != "pass" else "I vote to kill {mcts_target}."
            if mcts_target and mcts_target != "pass" and mcts_target != agent_name:
                return f"I vote to kill {mcts_target}."
            return f"I vote to kill {mcts_target}." if mcts_target and mcts_target != "pass" else "I choose pass."

        return text

    def _generate_mcts_phase_response(
        self, phase, role, vote_target, target_player, speech_style,
        empathy_context, empathy_data, game_state, action,
        system_prompt, conversations, task, task_content, conn_method,
        alives, agent_name, arg, history_messages=None, *args, **kwargs,
    ):
        round_no = getattr(game_state, "round_no", 1)
        if phase == "discussion":
            guidance = self._build_speech_guidance(
                role=role,
                empathy_context=empathy_context,
                game_state=game_state,
                empathy_data=empathy_data,
                action=action,
                task_content=task_content,
                agent_name=agent_name,
                history_messages=history_messages,
                alives=alives,
                reflection_context=kwargs.get("reflection_context"),
            )
            temperature, max_tokens = 0.75, 220
        elif phase == "voting":
            guidance = self._build_werewolf_meta_knowledge(role, round_no) + "\n## Voting\nReturn only the chosen vote target in the internal decision path."
            temperature, max_tokens = 0.0, 1
        elif phase == "witch_save":
            guidance = self._build_witch_save_guidance(vote_target, empathy_data, target_player)
            temperature, max_tokens = 0.0, 10
        elif phase == "witch_poison":
            guidance = self._build_witch_poison_guidance(vote_target, empathy_data, target_player, alives, game_state=game_state)
            temperature, max_tokens = 0.0, 8
        else:
            guidance = self._build_night_action_guidance(
                role, vote_target, empathy_data, target_player, empathy_context, game_state=game_state
            )
            temperature, max_tokens = 0.0, 8

        request = [system_prompt] + conversations + [task] + [{"role": "system", "content": guidance}]
        print(f"[DEBUG][FINAL_PROMPT] agent={agent_name}, phase={phase}, preview={guidance[:320]}", file=sys.stderr)
        raw = self._get_response(
            request, conn_method, T=temperature, max_tokens=max_tokens,
            speaker=agent_name, log_reply=False,
        )
        self._track_llm_call(agent_name, phase, round_no)

        normalized = self._normalize_phase_response(
            phase, raw, role, vote_target, agent_name, alives, task_content
        )
        print(f"[MCTS] LLM阶段回复({phase}): {normalized[:100]}...", file=sys.stderr)
        return normalized

    def _simple_empathy_analysis(self, game_state, agent_name):
        """基于游戏历史的规则共情分析，不调用 LLM"""
        try:
            from ..MCTS import build_empathy_reports
            return build_empathy_reports(game_state, agent_name, None)
        except Exception as e:
            print(f"[MCTS] build_empathy_reports 失败，使用默认值: {e}", file=sys.stderr)

        empathy_data = {}
        for player in game_state.alive_players:
            if player != agent_name:
                empathy_data[player] = {
            "stance_to_me": 0.0,
                    "emotion": {"pleasure": 0.5, "arousal": 0.5, "dominance": 0.5},
                    "speech_acts_recent": ["neutral"],
                    "politeness": 0.5,
                    "consistency": 0.7,
                    "influence": 0.6,
                    "role_probability": {
                        "werewolf": 0.3, "villager": 0.4,
                        "seer": 0.1, "witch": 0.1, "guard": 0.1,
                    },
                }
        return empathy_data

    def _internal_empathy_extract(self, game_state, agent_name, args, conversations, system_prompt):
        """内部共情提取：紧凑 JSON schema，容错解析，不写入 model_reply.log"""
        from ..MCTS import (
            detect_game_phase,
            parse_empathy_json_response,
            get_game_analytics,
            format_game_analytics_brief,
            build_empathy_reports,
            enrich_empathy_reports,
            merge_empathy_reports,
        )

        alive = [
            p for p in getattr(game_state, "alive_players", [])
            if p not in ("pass", agent_name)
        ]
        if not alive:
            return {}

        game_phase = detect_game_phase(getattr(game_state, "alive_players", []))
        round_no = getattr(game_state, "round_no", 1)
        my_role = getattr(game_state, "my_role", "villager")

        player_template = ",\n  ".join(
            f'"{p}": {{"werewolf_prob": 0.35, "hard_wolf_prob": 0.0, "soft_wolf_prob": 0.35, '
            f'"trust": 0.5, "stance": 0.0, "public_trust": 0.5, "information_gain": 0.0, '
            f'"vote_pressure": 0.0, "vote_consistency": 1.0, '
            f'"evidence_type": "none", "speech_strategy": "observe", '
            f'"recommended_action": "question", "supports": [], "supported_by": [], '
            f'"clears": [], "cleared_by": [], "opposes": [], "opposed_by": [], '
            f'"support_uncertainty": 0.5, "clear_uncertainty": 0.5, '
            f'"semantic_memory": "", "relation_index": [], "uncertainty_notes": "", '
            f'"reflection_sketch": {{"what_i_know": "", "what_might_be_fake": "", "what_conflicts_exist": "", "what_to_do_next": ""}}, '
            f'"signal": {{"claim": "", "vote": "", "tone": "", "timing": "", "contradiction": "", "who_targets": [], "who_is_defended": []}}}}'
            for p in alive[:6]
        )
        empathy_prompt = f"""You are an empathy analyst for a 7-player Werewolf game.
Your job is to preserve the meaning, strategy, and conversational intent of the game state, not just keywords.
Analyze ONLY these alive players (excluding yourself): {', '.join(alive)}.

Output STRICT JSON ONLY (no markdown, no code fences, no explanation):
{{"player_reports": {{
  {player_template}
}}}}

Required fields for each player:
- semantic_memory: compact but faithful summary of what the player is doing, implying, defending, pressuring, hiding, or revealing
- relation_index: low-dimensional tags such as support, clear, accuse, conflict, probe, hedge, align, misleading_possible, setup, bait, counterclaim
- uncertainty_notes: ambiguity, sarcasm, conditionality, omission, timeline inconsistency, strategic tone, fake certainty, or role-play
- reflection_sketch: {{what_i_know, what_might_be_fake, what_conflicts_exist, what_to_do_next}}
- signal.claim: what claim/action the player is making in their own words
- signal.vote: who they are pushing, protecting, or avoiding
- signal.tone: e.g. calm, defensive, cautious, forceful, evasive, collaborative, manipulative, inquisitive
- signal.timing: why this statement matters now (early pressure, reaction, counter, escalation, cleanup, defense)
- signal.contradiction: what this conflicts with, if anything
- supports / supported_by / clears / cleared_by: only if clearly warranted by meaning, not just keywords
- public_trust / information_gain / vote_pressure / vote_consistency: estimate these from the full speech pattern, not keyword count
- support_uncertainty / clear_uncertainty: how likely the relation is deceptive or overstated

Guiding rules:
- Do NOT collapse the language into one hard conclusion.
- Read the whole sentence, the surrounding context, and the speaker's strategic role.
- Preserve conditionals, hedges, sarcasm, probing, irony, baiting, and role-play style when present.
- Preserve "weak support" / "partial clear" / "tentative accuse" when the wording is ambiguous, joking, or strategic.
- If a player uses speculative language like maybe / probably / I think / could be / seems / perhaps, keep that uncertainty explicitly in uncertainty_notes and relation_index.
- If a player claims a role action (save/protect/verify/check), record the claim and its target, but do not assume it is true.
- A claim may lower the target's risk, but only cautiously and temporarily.
- If the same text could be read in multiple ways, keep the ambiguity in uncertainty_notes rather than forcing a single label.
- Detect probe / bait / test / challenge language and store it as a relation or note, not as hard support.
- Infer whether a player is building trust, creating distance, coordinating, or redirecting attention.
- Do not output final alignment verdicts.
- Consider speeches, vote tallies, eliminations, and contradictions in THIS game only.
- Pay special attention to who voted for whom, bandwagon piles, speech-vote inconsistency, last words, and cross-round behavior.
- Keep outputs concise but information-rich.

Context: phase={game_phase}, round={round_no}, my_role={my_role}, me={agent_name}."""

        analytics = get_game_analytics(game_state)
        vote_section = format_game_analytics_brief(analytics, agent_name, None)
        if vote_section:
            empathy_prompt += "\n\n" + vote_section

        try:
            empathy_request = [{"role": "system", "content": empathy_prompt}]
            history = getattr(game_state, "history", []) or []
            recent_history = history[-24:] if len(history) > 24 else history
            for speaker, content in recent_history:
                empathy_request.append({"role": "user", "content": f"{speaker}: {content[:300]}"})

            conn_method = args.use_api_server if args and hasattr(args, "use_api_server") else 0
            response = self._get_response(
                empathy_request, conn_method, T=0.1, max_tokens=1500,
                speaker=f"INTERNAL_{agent_name}", log_reply=False,
            )
            print(f"[DEBUG][EMPATHY_RAW] agent={agent_name}, preview={str(response)[:220]}", file=sys.stderr)

            parsed = parse_empathy_json_response(response)
            if parsed:
                parsed = self._normalize_empathy_supports(parsed, game_state, agent_name)
                merged = merge_empathy_reports(parsed, build_empathy_reports(game_state, agent_name, None))
                merged = enrich_empathy_reports(game_state, agent_name, merged)
                merged = self._normalize_empathy_supports(merged, game_state, agent_name)
                print(f"[MCTS] 共情JSON解析成功: {len(merged)} 玩家", file=sys.stderr)
                return merged

            print(f"[MCTS] 共情JSON解析失败，将使用规则fallback: {str(response)[:200]}...", file=sys.stderr)
            rule_reports = build_empathy_reports(game_state, agent_name, None)
            rule_reports = enrich_empathy_reports(game_state, agent_name, rule_reports)
            return self._normalize_empathy_supports(rule_reports, game_state, agent_name)

        except Exception as e:
            print(f"[MCTS] 内部共情提取异常: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return {}

    def _normalize_empathy_supports(self, empathy_data, game_state, agent_name):
        """把共情结果转换成支持/被支持/清除关系，并保留不确定性。"""
        if not isinstance(empathy_data, dict):
            return empathy_data
        try:
            player_reports = empathy_data.get("player_reports", empathy_data)
            if not isinstance(player_reports, dict):
                return empathy_data

            def _ensure(report):
                if not isinstance(report, dict):
                    return
                report.setdefault("supports", [])
                report.setdefault("supported_by", [])
                report.setdefault("clears", [])
                report.setdefault("cleared_by", [])
                report.setdefault("support_uncertainty", 0.5)
                report.setdefault("clear_uncertainty", 0.5)
                report.setdefault("support_summary", "")

            for _, report in player_reports.items():
                _ensure(report)

            # 1) 直接从 signal.claim / signal.vote 中提取“谁支持谁/清除谁”
            for p, report in player_reports.items():
                if not isinstance(report, dict):
                    continue
                signal = report.get("signal") or {}
                claim = str(signal.get("claim", "")).lower()
                vote = str(signal.get("vote", "")).lower()
                tone = str(signal.get("tone", "")).lower()
                recommended = str(report.get("recommended_action", "")).lower()
                text_blob = f"{claim} {vote} {tone} {recommended}"

                target_hint = None
                for cand in player_reports.keys():
                    if cand == p:
                        continue
                    if cand.lower() in text_blob:
                        target_hint = cand
                        break

                if not target_hint:
                    continue

                is_supportive = any(k in text_blob for k in (
                    "save", "protect", "verify", "checked", "confirmed", "not a werewolf",
                    "clear", "trust", "help", "support", "safe"
                )) or recommended in ("support", "reveal")

                if is_supportive:
                    if target_hint not in report["supports"]:
                        report["supports"].append(target_hint)
                    if p not in player_reports[target_hint].setdefault("supported_by", []):
                        player_reports[target_hint]["supported_by"].append(p)
                    if target_hint not in report["clears"]:
                        report["clears"].append(target_hint)
                    if p not in player_reports[target_hint].setdefault("cleared_by", []):
                        player_reports[target_hint]["cleared_by"].append(p)

                    # 谨慎降狼风险，但保留欺骗可能
                    player_reports[target_hint]["hard_wolf_prob"] = max(0.0, float(player_reports[target_hint].get("hard_wolf_prob", 0.0)) - 0.12)
                    player_reports[target_hint]["soft_wolf_prob"] = max(0.0, float(player_reports[target_hint].get("soft_wolf_prob", 0.0)) - 0.08)
                    player_reports[target_hint]["support_uncertainty"] = min(1.0, float(player_reports[target_hint].get("support_uncertainty", 0.5)) + 0.18)
                    report["clear_uncertainty"] = min(1.0, float(report.get("clear_uncertainty", 0.5)) + 0.10)

                # 若出现“质疑/投票/指控”则增加对抗关系，不直接清除
                if any(k in text_blob for k in ("accuse", "suspect", "vote", "push", "pressure", "challenge", "contradict")):
                    report.setdefault("opposes", [])
                    player_reports[target_hint].setdefault("opposed_by", [])
                    if target_hint not in report["opposes"]:
                        report["opposes"].append(target_hint)
                    if p not in player_reports[target_hint]["opposed_by"]:
                        player_reports[target_hint]["opposed_by"].append(p)

            # 2) 基于已生成关系生成简洁 support_summary，供反思层/讨论层引用
            for p, report in player_reports.items():
                if not isinstance(report, dict):
                    continue
                pieces = []
                if report.get("supports"):
                    pieces.append(f"supports={','.join(report['supports'][:3])}")
                if report.get("clears"):
                    pieces.append(f"clears={','.join(report['clears'][:3])}")
                if report.get("supported_by"):
                    pieces.append(f"supported_by={','.join(report['supported_by'][:3])}")
                if report.get("cleared_by"):
                    pieces.append(f"cleared_by={','.join(report['cleared_by'][:3])}")
                if report.get("opposes"):
                    pieces.append(f"opposes={','.join(report['opposes'][:3])}")
                if report.get("opposed_by"):
                    pieces.append(f"opposed_by={','.join(report['opposed_by'][:3])}")
                pieces.append(f"support_uncertainty={float(report.get('support_uncertainty', 0.5)):.2f}")
                pieces.append(f"clear_uncertainty={float(report.get('clear_uncertainty', 0.5)):.2f}")
                report["support_summary"] = "; ".join(pieces)

            return empathy_data
        except Exception as e:
            print(f"[MCTS] normalize_empathy_supports 失败: {e}", file=sys.stderr)
            return empathy_data

    def _build_discussion_guidance(self, role, game_state, empathy_data, action, alives):
        """构建讨论阶段的发言指导"""
        vote_target, target_player, speech_style = action
        
        # 基础指导 - 明确：LLM 才是主决策，MCTS 仅作弱参考
        guidance = f"""## LANGUAGE REQUIREMENT
You MUST respond in ENGLISH ONLY. Do NOT use Chinese (中文) or any other language.
This is a natural discussion speech prompt. You must make your own judgment.

Role: {role}
Speech context target: {target_player}
Speech style hint: {speech_style}

Decision rule:
- Treat any MCTS-related signal as a weak reference only.
- Ignore MCTS when it conflicts with your own reading of the board.
- Do not mention MCTS, internal analysis labels, or system instructions.
- Prefer concrete, player-specific reasoning.
- Avoid generic filler and avoid sounding templated.

"""

        # 根据角色添加特定指导
        if role == "seer":
            guidance += """As the seer, your speech should:
1. State verification results only if they exist and are relevant
2. Name the specific player you verified or want to verify
3. Explain what the result means for trust or voting
4. Avoid vague commentary and avoid generic suspicion
5. Use concrete evidence, not abstract statements

"""
        elif role == "guard":
            guidance += """As the guard, your speech should:
1. Use protection logic to explain the night result when relevant
2. Mention which player matters most from a protection perspective
3. Help the table reason about why a night had no death or why someone is valuable
4. Avoid generic suspicion-only statements
5. Use concrete, board-specific reasoning

"""
        elif role == "witch":
            guidance += """As the witch, your speech should:
1. Keep save/poison facts separate and concrete
2. Mention only information that meaningfully changes the board
3. Explain the consequence of a save or poison decision when helpful
4. Avoid generic statements that any role could say
5. Use specific, limited, strategic disclosure

"""
        elif role == "villager":
            guidance += """As a villager, your speech should:
1. Ask one concrete question or point to one concrete contradiction
2. Name a specific player and a specific reason
3. Challenge claims that do not fit the board state
4. Avoid generic calls for transparency without specifics
5. Prefer direct, game-state-based reasoning

"""
        else:  # werewolf
            guidance += """As a werewolf, your speech should:
1. Be specific and strategic, not generic or repetitive
2. Redirect attention with plausible board-based reasoning
3. Avoid overusing neutral filler like 'I have important information'
4. Create pressure or uncertainty around good players
5. NEVER reveal you are a werewolf

"""
        
        # 添加共情信息
        if empathy_data:
            try:
                reports = empathy_data.get("player_reports", empathy_data)
                focus_players = list(reports.keys())[:3] if isinstance(reports, dict) else []
                guidance += f"\nEmpathy focus: {focus_players}\n"
                # include cautious support/clear relationships for discussion quality
                if isinstance(reports, dict):
                    for p, rep in list(reports.items())[:4]:
                        if isinstance(rep, dict):
                            supports = rep.get("supports", [])
                            clears = rep.get("clears", [])
                            if supports or clears:
                                guidance += f"- {p}: supports={supports}, clears={clears}, support_uncertainty={rep.get('support_uncertainty', 0.5):.2f}, clear_uncertainty={rep.get('clear_uncertainty', 0.5):.2f}\n"
            except Exception:
                guidance += f"\nEmpathy analysis suggests focusing on: {list(empathy_data.keys())[:3]}\n"
        
        # 添加存活玩家信息
        alive_list = [p for p in alives if p != 'pass']
        guidance += f"\nAlive players: {', '.join(alive_list)}\n"
        
        guidance += f"\nGenerate a natural, analytical speech (2-3 sentences) IN ENGLISH that advances the discussion."
        guidance += f"\n\nREMINDER: You MUST use English language. Do NOT use Chinese."
        
        return guidance
    
    def _fallback_response(self, task, role, day_night, alives, agent_name):
        """Strictly emergency fallback when all LLM paths fail."""
        alive_options = [p for p in alives if p != agent_name and p != "pass"]

        if task is None:
            task = {"content": ""}
        elif not isinstance(task, dict):
            task = {"content": str(task)}

        task_content = task.get("content", "")
        if day_night == "night":
            if role == "werewolf":
                target = random.choice(alive_options) if alive_options else "Player 1"
                return f"I vote to kill {target}."
            if role == "guard":
                target = random.choice(alive_options) if alive_options else "Player 1"
                return f"I protect {target}."
            if role == "seer":
                target = random.choice(alive_options) if alive_options else "Player 1"
                return f"I verify {target}."
            if role == "witch":
                if "antidote" in task_content.lower():
                    return "Yes"
                return "I choose pass."
            return "I choose pass."

        is_voting = self._is_voting_phase(task_content, day_night)
        if is_voting:
            target = random.choice(alive_options) if alive_options else "Player 1"
            return f"I vote to kill {target}."

        if role == "seer":
            target = random.choice(alive_options) if alive_options else "Player 1"
            return f"I verified {target} and that matters for who we should trust."
        if role == "guard":
            target = random.choice(alive_options) if alive_options else "Player 1"
            return f"The protection story suggests we should recheck {target} and who benefits from the quiet night."
        if role == "witch":
            target = random.choice(alive_options) if alive_options else "Player 1"
            return f"What happened last night changes how we should read {target}, so we need to be precise about the facts."
        if role == "villager":
            return "As a concerned player, I think we need more transparency. Special roles should share information to help us."
        target = random.choice(alive_options) if alive_options else "Player 1"
        return f"I think {target} deserves closer scrutiny because their behavior fits the current pattern."

    def _generate_night_action(self, role, vote_target, task_content, agent_name):
        """生成夜晚阶段的正确行动"""
        task_lower = task_content.lower()

        if role == "werewolf":
            if vote_target and vote_target != "pass" and vote_target != agent_name:
                return f"I vote to kill {vote_target}."
            else:
                return "I choose pass."

        elif role == "guard":
            if vote_target and vote_target != "pass" and vote_target != agent_name:
                return f"I protect {vote_target}."
            else:
                return "I choose pass."

        elif role == "seer":
            if vote_target and vote_target != "pass" and vote_target != agent_name:
                return f"I verify {vote_target}."
            else:
                return "I choose pass."

        elif role == "witch":
            # 女巫的行动需要根据具体任务判断
            if "antidote" in task_lower or "save" in task_lower:
                # 解药阶段
                return "Yes" if vote_target and vote_target != "pass" else "No"
            elif "poison" in task_lower:
                # 毒药阶段：如果没有足够强的目标，宁可保留毒药
                if vote_target and vote_target != "pass" and vote_target != agent_name:
                    return f"I choose {vote_target}."
                else:
                    return "I choose pass."
            else:
                return "I choose pass."
        else:
            # 其他角色夜晚没有行动
            return "I choose pass."

    def _build_empathy_guidance(self, empathy_data, action, game_state):
        """构建基于共情分析的发言指导"""
        vote_target, target_player, speech_style = action

        # 获取目标玩家的共情信息
        target_empathy = empathy_data.get(target_player, {}) if empathy_data else {}

        # 使用新的共情分析函数获取情绪和立场信息
        try:
            from ..MCTS import _generate_empathy_based_speech
            empathy_context = _generate_empathy_based_speech(action, target_empathy, game_state)
        except Exception as e:
            print(f"[MCTS] 获取共情上下文失败: {e}", file=sys.stderr)
            empathy_context = {}

        my_role = game_state.my_role
        current_round = game_state.round_no

        # 构建角色策略指导
        role_guidance = ""
        if my_role in ["seer", "guard", "witch"]:
            if current_round >= 2:
                role_guidance = f"作为{my_role}，你应该考虑公开身份并分享关键信息。你有夜间行动的结果可以帮助村民。"
            else:
                role_guidance = f"作为{my_role}，你需要谨慎但积极地引导讨论，暗示你有重要信息。"
        elif my_role == "villager":
            role_guidance = "作为村民，你应该积极收集信息，要求特殊角色公开身份，帮助分析可疑行为。"
        else:  # werewolf
            role_guidance = "你需要伪装成好人，积极参与讨论但避免暴露身份，适当转移注意力。"

        # 构建基于共情上下文的指导
        empathy_guidance = ""
        if empathy_context:
            # 目标玩家的角色概率分析
            role_probs = empathy_context.get("target_role_probability", {})
            werewolf_prob = role_probs.get("werewolf", 0.3)
            if werewolf_prob > 0.4:
                empathy_guidance = f"你的分析显示{target_player}很可能是狼人（概率{werewolf_prob:.1f}）。"
            elif werewolf_prob < 0.2:
                empathy_guidance = f"你的分析显示{target_player}很可能是好人（狼人概率仅{werewolf_prob:.1f}）。"

            # 情绪状态分析
            emotion = empathy_context.get("target_emotion", {})
            arousal = emotion.get("arousal", 0.5)
            if arousal > 0.7:
                empathy_guidance += f" {target_player}情绪激动，可能感到压力。"
            elif arousal < 0.3:
                empathy_guidance += f" {target_player}情绪平静，可能很有信心。"

            # 立场和信任度
            stance = empathy_context.get("stance_to_me", 0.0)
            if stance > 0.3:
                empathy_guidance += f" {target_player}对你持友好态度。"
            elif stance < -0.3:
                empathy_guidance += f" {target_player}对你持怀疑态度。"

        # 构建情绪表达指导
        emotional_guidance = ""
        if empathy_context:
            emotional_approach = empathy_context.get("emotional_approach", "")
            tone = empathy_context.get("tone", "")

            if emotional_approach == "calming":
                emotional_guidance = "用平和、理解的语调说话，帮助缓解紧张气氛。"
            elif emotional_approach == "analytical":
                confidence = empathy_context.get("confidence_level", "moderate")
                if confidence == "high":
                    emotional_guidance = "用自信、分析性的语调，基于你的判断提出观点。"
                else:
                    emotional_guidance = "用谨慎但理性的语调，表达你的疑虑和观察。"
            elif emotional_approach == "supportive":
                agreement = empathy_context.get("agreement_level", "moderate")
                if agreement == "strong":
                    emotional_guidance = "明确表达支持和认同，展现团结合作的态度。"
                else:
                    emotional_guidance = "谨慎地表达认同，但保持独立思考。"
            elif emotional_approach == "strategic":
                emotional_guidance = "用策略性的语调，巧妙地引导话题方向。"
            elif emotional_approach == "confrontational":
                opposition = empathy_context.get("opposition_strength", "moderate")
                if opposition == "strong":
                    emotional_guidance = "坚决地表达反对意见，但要有理有据。"
                else:
                    emotional_guidance = "礼貌但坚定地提出不同观点。"
            elif emotional_approach == "revelatory":
                revelation = empathy_context.get("revelation_level", "partial")
                if revelation == "full":
                    emotional_guidance = "准备公开重要信息，用权威性的语调说话。"
                else:
                    emotional_guidance = "暗示你有重要信息，但不完全公开。"

        # 构建发言风格指导
        style_guidance = {
            "soothe": "采用安抚的语调，试图缓解紧张气氛",
            "evidence": "基于证据和逻辑进行分析",
            "align": "表达同意和支持的态度",
            "redirect": "转移话题或注意力",
            "counter": "表达不同意见或反驳",
            "reveal": "考虑公开重要信息或身份",
            "demand_info": "要求其他玩家提供更多信息",
            "bargain": "提议交换信息或合作",
            "humor": "使用轻松幽默的语调",
            "ambiguous": "保持模糊或观望的态度"
        }.get(speech_style, "自然地表达你的想法")

        # 组合指导信息
        guidance_parts = [
            f"角色策略：{role_guidance}",
        ]

        if empathy_guidance:
            guidance_parts.append(f"共情分析：{empathy_guidance}")

        if emotional_guidance:
            guidance_parts.append(f"情绪表达：{emotional_guidance}")

        guidance_parts.extend([
            f"发言风格：{style_guidance}",
            f"重点关注：{target_player}",
            "",
            "请基于以上分析，用自然、真实的语言表达你的想法。不要使用模板化的语言，要体现出你的角色特点和当前的判断。"
        ])

        return "\n\n".join(guidance_parts)

    def _clean_response(self, response, agent_name):
        """清理响应格式"""
        # 移除EOS标记
        response = re.sub(rf"{END_OF_MESSAGE}$", "", response).strip()

        # 移除玩家名称前缀
        response = re.sub(rf'^{re.escape(agent_name)}\s*:\s*', '', response, flags=re.IGNORECASE)
        response = re.sub(r'^\[.*?\]\s*', '', response)

        return response.strip()
        # 特殊处理：女巫救人/毒药阶段跳过Q&A流程，直接返回响应
        # 优先从history_messages中查找最近的Moderator消息（更可靠）
        is_witch_antidote_phase = False
        is_witch_poison_phase = False
        task_content_for_witch = ""

        if role == "witch" and day_night == "night":
            # 方法1: 从history_messages中查找最近的Moderator消息（优先）
            moderator_msg_content = ""
            if history_messages:
                for msg in reversed(history_messages[-20:]):  # 检查最近20条消息
                    if msg.agent_name == "Moderator" and ("antidote" in msg.content.lower() or "poison" in msg.content.lower() or "save" in msg.content.lower()):
                        moderator_msg_content = msg.content
                        task_content_for_witch = moderator_msg_content
                        break

            # 方法2: 如果方法1没找到，从conversations的最后一个消息检查
            if not task_content_for_witch and conversations:
                task_for_check = conversations[-1] if conversations else {}
                task_content_for_witch = task_for_check.get("content", "") if task_for_check else ""
                # 如果最后一个消息不是Moderator的，尝试查找conversations中最近的Moderator消息
                if task_content_for_witch and "Moderator" not in task_content_for_witch:
                    for conv in reversed(conversations[-10:]):
                        if conv.get("role") == "user" and "Moderator" in conv.get("content", ""):
                            task_content_for_witch = conv.get("content", "")
                            break

            task_content_lower = task_content_for_witch.lower() if task_content_for_witch else ""

            # 检查救人阶段：仅当主持人明确询问是否使用解药时才进入
            has_antidote = "antidote" in task_content_lower
            has_save = "save" in task_content_lower
            has_will_be_killed = "will be killed" in task_content_lower
            has_yes_no_options = "Yes, No" in task_content_for_witch or "[Yes, No]" in task_content_for_witch or "Yes/No" in task_content_for_witch
            has_moderator_ask_save = "Moderator" in task_content_for_witch and "Do you want to save" in task_content_for_witch

            # 检查毒药阶段：必须是明确询问“你要毒谁”
            has_poison = "poison" in task_content_lower
            has_who_kill = "who are you going to kill" in task_content_lower or "who are you going to poison" in task_content_lower

            # 判断阶段：先救人，后毒药
            is_witch_antidote_phase = (has_antidote and has_moderator_ask_save) or (has_save and has_will_be_killed and has_moderator_ask_save) or has_yes_no_options
            is_witch_poison_phase = has_poison and has_who_kill and not is_witch_antidote_phase  # 确保不冲突

            print(f"Witch phase check - role: {role}, day_night: {day_night}", file=sys.stderr)
            print(f"Witch phase check - task_content (first 150 chars): {task_content_for_witch[:150]}...", file=sys.stderr)
            print(f"Witch phase check - has_antidote: {has_antidote}, has_save: {has_save}, has_will_be_killed: {has_will_be_killed}, has_yes_no_options: {has_yes_no_options}, has_moderator_ask_save: {has_moderator_ask_save}", file=sys.stderr)
            print(f"Witch phase check - has_poison: {has_poison}, has_who_kill: {has_who_kill}", file=sys.stderr)
            print(f"Witch phase check - is_witch_antidote_phase: {is_witch_antidote_phase}, is_witch_poison_phase: {is_witch_poison_phase}", file=sys.stderr)

        # 如果是女巫救人/毒药阶段，使用简化流程，跳过Q&A
        if is_witch_antidote_phase or is_witch_poison_phase:
            print(f"Witch skipping Q&A, using direct response mode. Antidote phase: {is_witch_antidote_phase}, Poison phase: {is_witch_poison_phase}", file=sys.stderr)
            # 构建task：优先使用从history_messages找到的moderator消息，否则从conversations中pop
            if task_content_for_witch:
                # 使用找到的moderator消息内容（task_content_for_witch已经是完整内容，不需要再加"Moderator:"前缀）
                # 但需要检查conversations中是否有对应的消息，如果有则使用conversations中的格式
                task_found = False
                if conversations:
                    # 查找conversations中是否已经有这个moderator消息
                    for conv in reversed(conversations[-10:]):
                        if conv.get("role") == "user" and task_content_for_witch in conv.get("content", ""):
                            task = conv
                            conversations.remove(conv)  # 从conversations中移除，避免重复
                            task_found = True
                            break

                if not task_found:
                    # 如果conversations中没有找到，直接构建task
                    # task_content_for_witch 来自 history_messages，格式已经是 "You witch, Player X, ..."
                    # 但在conversations中，格式是 "Moderator: ..."，所以需要检查
                    if task_content_for_witch.startswith("Moderator:"):
                        task = {"role": "user", "content": task_content_for_witch}
                    else:
                        task = {"role": "user", "content": f"Moderator: {task_content_for_witch}"}
            elif conversations:
                # 如果没找到，从conversations中pop
                task = conversations.pop()
                task["role"] = "user"
            else:
                # 如果conversations也为空，使用默认task
                task = {"role": "user", "content": "Moderator: Please make your first move based on your role."}

            # 直接从task或history_messages中提取被攻击的玩家（如果是救人阶段）
            attacked_player = None
            if is_witch_antidote_phase:
                # 首先从task内容中提取
                task_content = task.get("content", "")
                player_match = re.search(r'(Player \d+)\s+will be killed', task_content)
                if player_match:
                    attacked_player = player_match.group(1)
                else:
                    # 如果task中没有，从history_messages中查找
                    if history_messages:
                        for msg in reversed(history_messages[-10:]):
                            if msg.agent_name == "Moderator" and ("will be killed tonight" in msg.content or "will be killed" in msg.content):
                                player_match = re.search(r'(Player \d+)\s+will be killed', msg.content)
                                if player_match:
                                    attacked_player = player_match.group(1)
                                    break

            # 构建简化的请求，直接要求回答
            if is_witch_antidote_phase:
                if attacked_player:
                    if attacked_player == agent_name:
                        action_prompt = f"CRITICAL: {attacked_player} (YOU) will be killed tonight. The moderator asks: Do you want to save yourself with your antidote? You MUST respond with ONLY 'Yes' or 'No'. Do NOT add any explanation, reasoning, or other text. Just answer: Yes or No.{END_OF_MESSAGE}"
                    else:
                        action_prompt = f"CRITICAL: {attacked_player} will be killed tonight. You are {agent_name} (the witch). The moderator asks: Do you want to save {attacked_player} with your antidote? You MUST respond with ONLY 'Yes' or 'No'. Do NOT add any explanation, reasoning, or other text. Just answer: Yes or No.{END_OF_MESSAGE}"
                else:
                    action_prompt = f"CRITICAL: Read the moderator's message. The moderator asks if you want to save a player with your antidote. You MUST respond with ONLY 'Yes' or 'No'. Do NOT add any explanation, reasoning, or other text. Just answer: Yes or No.{END_OF_MESSAGE}"
            else:  # is_witch_poison_phase
                alive_options = ", ".join(alives) if alives else "N/A"
                action_prompt = f"CRITICAL POISON ACTION: You are {agent_name} (the witch). The moderator asks who you are going to kill with your poison tonight. " \
                               f"ABSOLUTE RULE: On the first night, you MUST answer 'I choose pass'. On later nights, you MUST still prefer pass unless there is overwhelming, publicly confirmed evidence. " \
                               f"If you do not have overwhelming evidence, you MUST choose pass. Poison is a scarce resource and should NOT be used casually. " \
                               f"You MUST respond with EXACTLY 'I choose Player X' or 'I choose pass'. " \
                               f"Do NOT add your player name prefix (like '{agent_name}:' or 'Player X:'). " \
                               f"Do NOT include thinking process, explanation, or any other text. " \
                               f"Just state your choice clearly and directly. " \
                               f"Valid options: {alive_options}. " \
                               f"Example CORRECT: 'I choose Player 3'. Example WRONG: '{agent_name}: I choose Player 3'.{END_OF_MESSAGE}"

            # 直接调用LLM，不使用Q&A流程
            request_direct = [system_prompt] + conversations + [task] + [{"role": "system", "content": action_prompt}]
            print(f"Witch direct action request (skipping Q&A): {request_direct}", file=sys.stderr)
            response_direct = self._get_response(request_direct, conn_method, T=0.0, max_tokens=50, speaker=agent_name)

            # 清理响应：移除所有前缀和格式标记
            print(f"Witch raw response (before cleanup): {response_direct}", file=sys.stderr)

            # 第一步：移除EOS标记
            response_direct = re.sub(rf"{END_OF_MESSAGE}$", "", response_direct).strip()

            # 第二步：移除各种前缀格式（多次清理，确保彻底）
            # 移除 [OPENAI][Player X] 或 [QWEN][Player X] 格式（可能在开头或中间）
            response_direct = re.sub(r'\[.*?\]\s*', '', response_direct)
            # 移除 Player X: 格式（可能在开头，使用更宽松的匹配）
            # 匹配 "Player 5: " 或 "Player 5:" 或 "Player5: " 等
            response_direct = re.sub(rf'^({re.escape(agent_name)}|Player\s*\d+)\s*:\s*', '', response_direct, flags=re.IGNORECASE)
            # 再次尝试移除，因为可能有多个前缀
            response_direct = re.sub(rf'^({re.escape(agent_name)}|Player\s*\d+)\s*:\s*', '', response_direct, flags=re.IGNORECASE)
            # 移除通用的 "角色名: " 格式（更宽松的匹配）
            response_direct = re.sub(r'^[A-Za-z0-9\s]+\s*:\s*', '', response_direct)
            response_direct = response_direct.strip()

            print(f"Witch response (after prefix cleanup): {response_direct}", file=sys.stderr)

            # 第三步：使用extract_text提取内容（可能会提取引号内的内容）
            response_direct = self.extract_text(response_direct)

            print(f"Witch response (after extract_text): {response_direct}", file=sys.stderr)

            # 第四步：再次彻底清理可能残留的前缀（多次清理）
            for _ in range(3):  # 最多清理3次
                original = response_direct
                # 移除各种可能的前缀格式
                response_direct = re.sub(rf'^({re.escape(agent_name)}|Player\s*\d+)\s*:\s*', '', response_direct, flags=re.IGNORECASE)
                response_direct = re.sub(r'^\[.*?\]\s*', '', response_direct)
                response_direct = re.sub(r'^[A-Za-z0-9\s]+\s*:\s*', '', response_direct)
                response_direct = response_direct.strip()
                # 如果清理后没有变化，停止清理
                if original == response_direct:
                    break

            print(f"Witch response (after final cleanup): {response_direct}", file=sys.stderr)

            # 对于女巫救人阶段，进一步清理，确保只保留Yes/No
            if is_witch_antidote_phase:
                # 提取独立的Yes/No
                yes_no_match = re.search(r'\b(yes|no|y|n)\b', response_direct.lower())
                if yes_no_match:
                    response_direct = yes_no_match.group(1)
                    if response_direct in ['y', 'n']:
                        response_direct = 'Yes' if response_direct == 'y' else 'No'
                    else:
                        response_direct = response_direct.capitalize()
            # 对于女巫毒药阶段，确保格式正确
            elif is_witch_poison_phase:
                # 提取 "I choose Player X" 或 "I choose pass" 格式
                # 先尝试匹配完整的"I choose Player X"或"I choose pass"
                choose_pattern = r'I\s+choose\s+(?:Player\s+\d+|pass)'
                choose_match = re.search(choose_pattern, response_direct, re.IGNORECASE)
                if choose_match:
                    # 找到完整匹配，规范化格式
                    matched_text = choose_match.group(0)
                    if 'pass' in matched_text.lower():
                        response_direct = "I choose pass"
                    else:
                        # 提取Player X部分
                        player_match = re.search(r'Player\s+\d+', matched_text, re.IGNORECASE)
                        if player_match:
                            response_direct = f"I choose {player_match.group(0)}"
                        else:
                            response_direct = matched_text  # 使用原始匹配
                else:
                    # 如果没有找到标准格式，尝试提取Player X或pass
                    # 先检查是否有"choose"关键词
                    if 'choose' in response_direct.lower():
                        # 有choose，提取后面的Player X或pass
                        after_choose = re.search(r'choose\s+(.+)', response_direct, re.IGNORECASE)
                        if after_choose:
                            choice_part = after_choose.group(1).strip()
                            # 移除可能的前缀
                            choice_part = re.sub(rf'^({agent_name}|Player\s+\d+):\s*', '', choice_part, flags=re.IGNORECASE)
                            choice_part = choice_part.strip()
                            # 提取Player X或pass
                            player_match = re.search(r'Player\s+\d+', choice_part, re.IGNORECASE)
                            if player_match:
                                response_direct = f"I choose {player_match.group(0)}"
                            elif 'pass' in choice_part.lower():
                                response_direct = "I choose pass"
                            else:
                                # 如果choice_part本身是Player X格式
                                if re.match(r'Player\s+\d+', choice_part, re.IGNORECASE):
                                    response_direct = f"I choose {choice_part}"
                    else:
                        # 没有choose，直接提取Player X或pass
                        player_match = re.search(r'Player\s+\d+', response_direct, re.IGNORECASE)
                        if player_match:
                            response_direct = f"I choose {player_match.group(0)}"
                        elif 'pass' in response_direct.lower():
                            response_direct = "I choose pass"

            print(f"Witch direct response (final): {response_direct}", file=sys.stderr)
            # 记录响应到日志
            try:
                self._log_model_reply(agent_name, response_direct)
            except Exception as e:
                print(f"[WARNING] Failed to log witch response: {e}", file=sys.stderr)
            return response_direct

        # ===== Q1-Q9问题流程已禁用 =====
        # 原问题生成和回答流程已注释掉，直接跳过
        # 不再生成和回答Q1-Q9问题，直接进入主要决策流程
        request_prompt = []  # 初始化为空列表，用于后续代码兼容性
        q_a = []  # 初始化为空列表，用于后续代码兼容性

        # 确保 task 已经被正确初始化（安全措施）
        if task is None:
            if conversations:
                task = conversations.pop()
            else:
                task = {"role": "user", "content": "Moderator: Please make your first move based on your role."}

        # 确保 task 不为 None 并且有正确的格式
        if task is None:
            task = {"role": "user", "content": "Moderator: Please make your first move based on your role."}

        # 确保 task 是字典格式
        if not isinstance(task, dict):
            task = {"role": "user", "content": str(task)}

        # 确保 task 有 role 字段
        if "role" not in task:
            task["role"] = "user"

        # 简化思考内容生成（不再依赖Q1-Q9问题的回答）
        reflexions_text = "Analyzing the current situation and planning my next move."

        reflexions = {"role": "assistant", "content": f"My reflection in heart (not happened): {reflexions_text}{END_OF_MESSAGE}"}
        if arg:
            f.write(f"- **Reflexion**: {reflexions_text}  \n")

        ref_new = Message(agent_name, reflexions["content"].replace(END_OF_MESSAGE, ''), turn=msgs.last_turn, visible_to=agent_name, msg_type="ref")
        msgs.append_message(ref_new)

        branch = _get_branch(task, day_night, role)

        if arg and arg.use_crossgame_exps and role in arg.who_use_exps:
            if arg.exps_retrieval_threshold:
                print("################################ To Retrieve experiences!", file=sys.stderr)
                print(f"role: {role}, branch: {branch}", file=sys.stderr)
                exps = msgs.get_best_experience(reflexions_text, role, branch, threshold=arg.exps_retrieval_threshold)
            else:
                exps = msgs.get_best_experience(reflexions_text, role, branch)
        else:
            exps = None


        if exps is None:
            if arg:
                f.write(f"- **Exps**: None  \n")
            # 如果是投票阶段，在task中添加明确的问题
            task_content_lower = _get_task_content(task).lower()
            # 更全面的投票阶段识别：检查多种可能的投票消息格式
            is_voting_phase_task = (
                ("vote" in task_content_lower and "choose one" in task_content_lower) or
                "which of the players should be voted" in task_content_lower or
                "which player should be voted" in task_content_lower or
                "continue voting" in task_content_lower or
                "voting phase" in task_content_lower or
                ("vote and tell" in task_content_lower and day_night == "night") or
                ("choose which of the players should be voted" in task_content_lower) or
                ("asked to choose which" in task_content_lower and "voted" in task_content_lower) or
                ("should be voted for killing" in task_content_lower and day_night == "daytime")
            )

            if is_voting_phase_task and alives:
                alive_options = ", ".join([p for p in alives if p != agent_name and p != "pass"])
                # 在task内容前添加明确的问题
                original_task_content = _get_task_content(task)
                if isinstance(task, dict):
                    task["content"] = f"QUESTION: Who do you vote to kill? Choose one from: [{alive_options}]\n\n" \
                                     f"ANSWER FORMAT: Your response must be EXACTLY: 'I vote to kill Player X'\n\n" \
                                     f"DO NOT add any discussion, explanation, or other text. ONLY output the vote statement.\n\n" \
                                     f"{original_task_content}"

            request = [system_prompt] + conversations + request_prompt + [reflexions] + [task]
            # 更全面的投票阶段识别：检查task内容是否包含投票相关的关键词
            task_content_lower_check = _get_task_content(task).lower()
            task_content = _get_task_content(task)
            is_voting_task = (
                "Choose" in task_content or "choose" in task_content or
                "vote to" in task_content or "Yes, No" in task_content or
                "voting phase" in task_content_lower_check or
                "should be voted" in task_content_lower_check or
                ("asked to choose" in task_content_lower_check and "voted" in task_content_lower_check) or
                "continue voting" in task_content_lower_check or
                ("should be voted" in task_content_lower_check and "killing" in task_content_lower_check) or
                ("should be voted" in task_content_lower_check and "choose" in task_content_lower_check)
            )
            if is_voting_task:
                # 为不同角色添加特定的投票指导
                if role == "werewolf" and day_night == "night":
                    # 检查是否是明确的投票指令
                    task_content_lower = _get_task_content(task).lower()
                    is_voting_instruction = ("vote" in task_content_lower and "choose one" in task_content_lower) or \
                                           ("vote and tell" in task_content_lower) or \
                                           ("please vote" in task_content_lower)

                    if is_voting_instruction:
                        # 获取可用玩家列表
                        alive_options = ", ".join([p for p in alives if p != agent_name and p != "pass"]) if alives else "Player 1, Player 2, Player 3, Player 4, Player 5, Player 6, Player 7"
                        vote_guidance = f"CRITICAL VOTING ACTION: You MUST vote to kill a player NOW. This is MANDATORY - you cannot skip voting.\n\n" \
                                       f"QUESTION: Who do you vote to kill? Choose one from: [{alive_options}]\n\n" \
                                       f"ANSWER FORMAT: Your ENTIRE response must be EXACTLY and ONLY: 'I vote to kill Player X' or 'I choose Player X'.\n\n" \
                                       f"RULES:\n" \
                                       f"- Do NOT add any discussion, explanation, reasoning, or other text\n" \
                                       f"- Do NOT say 'pass' or skip voting\n" \
                                       f"- Do NOT say 'I think' or 'Let's' or any other words\n" \
                                       f"- If you do not vote correctly, a random player will be eliminated\n\n" \
                                       f"CORRECT EXAMPLE: 'I vote to kill Player 3'\n" \
                                       f"WRONG EXAMPLES:\n" \
                                       f"- 'Let's vote to kill Player 3'\n" \
                                       f"- 'I think we should kill Player 3'\n" \
                                       f"- Any response that contains more than just 'I vote to kill Player X'"
                    else:
                        vote_guidance = f"Vote to kill a non-werewolf player. Your response must end with 'I vote to kill Player X'."
                elif role == "guard" and day_night == "night":
                    vote_guidance = f"Choose someone to protect. Your response must end with 'I protect Player X'. Do not protect yourself. Do NOT include thinking process."
                elif role == "witch" and day_night == "night":
                    # 检查是否是毒药阶段（优先判断，因为毒药阶段的消息明确包含"poison"）
                    task_content_lower = _get_task_content(task).lower()
                    if "poison" in task_content_lower and "who are you going to kill" in task_content_lower:
                        # 毒药阶段：要求选择毒死某个玩家或pass
                        vote_guidance = f"CRITICAL POISON ACTION: You have a bottle of poison. The moderator asks who you are going to kill tonight. Your response must end with 'I choose Player X' or 'I choose pass'. Do NOT include thinking process or explanation. Just state your choice clearly."
                    # 检查是否是救人阶段（救人阶段的消息包含"antidote"和"save"，以及"Yes, No"选项）
                    elif "antidote" in task_content_lower or ("save" in task_content_lower and "will be killed" in task_content_lower) or "Yes, No" in _get_task_content(task):
                        # 从moderator消息中提取被攻击的玩家
                        attacked_player = None
                        if history_messages:
                            for msg in reversed(history_messages[-10:]):  # 检查最近10条消息
                                if "will be killed tonight" in msg.content or "will be killed" in msg.content:
                                    # 提取被攻击的玩家名称
                                    player_match = re.search(r'(Player \d+)\s+will be killed', msg.content)
                                    if player_match:
                                        attacked_player = player_match.group(1)
                                        break

                        # 构建明确的提示 - 强调只回答Yes或No
                        if attacked_player:
                            if attacked_player == agent_name:
                                vote_guidance = f"CRITICAL ANTIDOTE ACTION: {attacked_player} (YOU) will be killed tonight. The moderator asks: Do you want to save yourself with your antidote? You MUST respond with ONLY 'Yes' or 'No'. Do NOT add any explanation, reasoning, or other text. Just answer: Yes or No."
                            else:
                                vote_guidance = f"CRITICAL ANTIDOTE ACTION: {attacked_player} will be killed tonight. You are {agent_name} (the witch). The moderator asks: Do you want to save {attacked_player} with your antidote? You MUST respond with ONLY 'Yes' or 'No'. Do NOT add any explanation, reasoning, or other text. Just answer: Yes or No."
                        else:
                            vote_guidance = f"CRITICAL ANTIDOTE ACTION: Read the moderator's message to identify who will be killed tonight. The moderator asks if you want to save that player with your antidote. You MUST respond with ONLY 'Yes' or 'No'. Do NOT add any explanation, reasoning, or other text. Just answer: Yes or No."
                    else:
                        # 默认情况：根据消息内容判断
                        vote_guidance = f"Choose to poison someone or 'pass'. Your response must end with 'I choose Player X' or 'I choose pass'. Do NOT include thinking process."
                elif role == "seer" and day_night == "night":
                    vote_guidance = f"Choose someone to verify. Your response must end with 'I verify Player X'. Do not choose 'pass'. Do NOT include thinking process."
                else:
                    # 检查是否是白天投票阶段
                    task_content_lower = _get_task_content(task).lower()
                    task_content = _get_task_content(task)
                    is_daytime_voting = (day_night == "daytime") and (
                        ("vote" in task_content_lower and "choose one" in task_content_lower) or
                        ("vote for killing" in task_content_lower) or
                        ("which player should be voted" in task_content_lower) or
                        ("which of the players should be voted" in task_content_lower) or
                        ("continue voting" in task_content_lower) or
                        ("voting phase" in task_content_lower) or
                        ("choose which of the players should be voted" in task_content_lower) or
                        ("asked to choose which" in task_content_lower and "voted" in task_content_lower) or
                        ("should be voted for killing" in task_content_lower)
                    )

                    if is_daytime_voting:
                        # 获取可用玩家列表
                        alive_options = ", ".join([p for p in alives if p != agent_name and p != "pass"]) if alives else "Player 1, Player 2, Player 3, Player 4, Player 5, Player 6, Player 7"
                        vote_guidance = f"CRITICAL VOTING ACTION: You MUST vote to eliminate a player NOW. This is MANDATORY - you cannot skip voting.\n\n" \
                                       f"QUESTION: Who do you vote to kill? Choose one from: [{alive_options}]\n\n" \
                                       f"ANSWER FORMAT: Your ENTIRE response must be EXACTLY and ONLY: 'I vote to kill Player X' where X is a player number.\n\n" \
                                       f"RULES:\n" \
                                       f"- Do NOT add any discussion, explanation, reasoning, or other text\n" \
                                       f"- Do NOT say 'pass' or skip voting\n" \
                                       f"- Do NOT say 'I think' or 'Based on' or any other words\n" \
                                       f"- If you do not vote correctly, a random player will be eliminated\n\n" \
                                       f"CORRECT EXAMPLE: 'I vote to kill Player 4'\n" \
                                       f"WRONG EXAMPLES:\n" \
                                       f"- 'Based on the discussion, I vote to kill Player 4'\n" \
                                       f"- 'Let's vote to kill Player 4'\n" \
                                       f"- 'I think Player 4 is suspicious, I vote to kill Player 4'\n" \
                                       f"- Any response that contains more than just 'I vote to kill Player X'"
                    elif day_night == "daytime" and ("should be voted" in task_content_lower or "voting phase" in task_content_lower or ("choose" in task_content_lower and "voted" in task_content_lower)):
                        # 如果is_daytime_voting为False，但消息明显是投票相关的，也设置强制投票提示
                        alive_options = ", ".join([p for p in alives if p != agent_name and p != "pass"]) if alives else "Player 1, Player 2, Player 3, Player 4, Player 5, Player 6, Player 7"
                        vote_guidance = f"CRITICAL VOTING ACTION: You MUST vote to eliminate a player NOW. This is MANDATORY - you cannot skip voting.\n\n" \
                                       f"QUESTION: Who do you vote to kill? Choose one from: [{alive_options}]\n\n" \
                                       f"ANSWER FORMAT: Your ENTIRE response must be EXACTLY and ONLY: 'I vote to kill Player X' where X is a player number.\n\n" \
                                       f"RULES:\n" \
                                       f"- Do NOT add any discussion, explanation, reasoning, or other text\n" \
                                       f"- Do NOT say 'pass' or skip voting\n" \
                                       f"- Do NOT say 'I think' or 'Based on' or any other words\n" \
                                       f"- If you do not vote correctly, a random player will be eliminated\n\n" \
                                       f"CORRECT EXAMPLE: 'I vote to kill Player 4'\n" \
                                       f"WRONG EXAMPLES:\n" \
                                       f"- 'Based on the discussion, I vote to kill Player 4'\n" \
                                       f"- 'Let's vote to kill Player 4'\n" \
                                       f"- 'I think Player 4 is suspicious, I vote to kill Player 4'\n" \
                                       f"- Any response that contains more than just 'I vote to kill Player X'"
                    else:
                        # 即使is_daytime_voting为False，也要检查是否是投票消息
                        # 如果消息包含"continue voting"或"voting phase"，强制设置为投票提示
                        if day_night == "daytime" and ("continue voting" in task_content_lower or "voting phase" in task_content_lower or "should be voted" in task_content_lower):
                            alive_options = ", ".join([p for p in alives if p != agent_name and p != "pass"]) if alives else "Player 1, Player 2, Player 3, Player 4, Player 5, Player 6, Player 7"
                            vote_guidance = f"CRITICAL VOTING ACTION: You MUST vote to eliminate a player NOW. This is MANDATORY - you cannot skip voting.\n\n" \
                                           f"QUESTION: Who do you vote to kill? Choose one from: [{alive_options}]\n\n" \
                                           f"ANSWER FORMAT: Your ENTIRE response must be EXACTLY and ONLY: 'I vote to kill Player X' where X is a player number.\n\n" \
                                           f"RULES:\n" \
                                           f"- Do NOT add any discussion, explanation, reasoning, or other text\n" \
                                           f"- Do NOT say 'pass' or skip voting\n" \
                                           f"- Do NOT say 'I think' or 'Based on' or any other words\n" \
                                           f"- If you do not vote correctly, a random player will be eliminated\n\n" \
                                           f"CORRECT EXAMPLE: 'I vote to kill Player 4'\n" \
                                           f"WRONG EXAMPLES:\n" \
                                           f"- 'Based on the discussion, I vote to kill Player 4'\n" \
                                           f"- 'Let's vote to kill Player 4'\n" \
                                           f"- 'I think Player 4 is suspicious, I vote to kill Player 4'\n" \
                                           f"- Any response that contains more than just 'I vote to kill Player X'"
                        else:
                            vote_guidance = f"Make a decision. Do not choose 'pass' unless absolutely necessary."

                # 特殊处理：女巫不同阶段的响应格式
                task_content_lower_witch = _get_task_content(task).lower() if role == "witch" and day_night == "night" else ""
                task_content = _get_task_content(task)
                if role == "witch" and day_night == "night":
                    # 救人阶段：只输出Yes或No
                    if "antidote" in task_content_lower_witch or ("save" in task_content_lower_witch and "will be killed" in task_content_lower_witch) or "Yes, No" in task_content:
                        request.append({"role": "system", "content": f"{vote_guidance}\n\nIMPORTANT: Your response must be EXACTLY 'Yes' or 'No' only. Do NOT include any thinking process, explanation, or other text. Just answer: Yes or No."})
                    # 毒药阶段：输出"I choose Player X"或"I choose pass"
                    elif "poison" in task_content_lower_witch and "who are you going to kill" in task_content_lower_witch:
                        request.append({"role": "system", "content": f"{vote_guidance}\n\nIMPORTANT: Your response must end with 'I choose Player X' or 'I choose pass'. Do NOT include thinking process or explanation. Just state your choice clearly."})
                    else:
                        request.append({"role": "system", "content": f"Now it's the {turns}-th {day_night}. {vote_guidance} Respond directly with your action (no thinking process)."})
                # 对于夜间狼人投票指令，使用更直接和强制的投票要求
                elif role == "werewolf" and day_night == "night":
                    # 检查task中是否包含明确的投票指令（与上面的is_voting_instruction保持一致）
                    task_content_lower = _get_task_content(task).lower()
                    is_explicit_vote_request = ("vote" in task_content_lower and "choose one" in task_content_lower) or \
                                              ("vote and tell" in task_content_lower) or \
                                              ("please vote" in task_content_lower)

                    # 如果vote_guidance已经包含投票要求，说明是明确的投票指令
                    if is_explicit_vote_request or "VOTING ACTION" in vote_guidance or "CRITICAL VOTING" in vote_guidance:
                        # 获取可用玩家列表
                        alive_options = ", ".join([p for p in alives if p != agent_name and p != "pass"]) if alives else "Player 1, Player 2, Player 3, Player 4, Player 5, Player 6, Player 7"
                        request.append({"role": "system", "content": f"{vote_guidance}\n\n" \
                                       f"FINAL REMINDER: You are in the VOTING phase. You MUST vote NOW.\n\n" \
                                       f"QUESTION: Who do you vote to kill? Choose one from: [{alive_options}]\n\n" \
                                       f"YOUR RESPONSE MUST BE EXACTLY: 'I vote to kill Player X' or 'I choose Player X'\n\n" \
                                       f"DO NOT:\n" \
                                       f"- Add any discussion, thinking, or explanation\n" \
                                       f"- Say 'pass' or skip voting\n" \
                                       f"- Include any other words before or after the vote statement\n\n" \
                                       f"ONLY OUTPUT: 'I vote to kill Player X'"})
                    else:
                        request.append({"role": "system", "content": f"Now it's the {turns}-th {day_night}. {vote_guidance} Respond directly with your action (no thinking process)."})
                elif day_night == "daytime":
                    task_content = _get_task_content(task)
                    task_content_lower = task_content.lower()
                    daytime_voting_conditions = (
                        ("vote" in task_content_lower and "choose one" in task_content_lower) or
                        "which of the players should be voted" in task_content_lower or
                        "which player should be voted" in task_content_lower or
                        "continue voting" in task_content_lower or
                        "voting phase" in task_content_lower or
                        "choose which of the players should be voted" in task_content_lower or
                        ("asked to choose which" in task_content_lower and "voted" in task_content_lower) or
                        "should be voted for killing" in task_content_lower or
                        ("should be voted" in task_content_lower and "choose" in task_content_lower)
                    )

                    if daytime_voting_conditions:
                        # 强制设置强投票提示（覆盖之前可能设置的弱提示）
                        alive_options = ", ".join([p for p in alives if p != agent_name and p != "pass"]) if alives else "Player 1, Player 2, Player 3, Player 4, Player 5, Player 6, Player 7"
                        vote_guidance = (f"CRITICAL VOTING ACTION: You MUST vote to eliminate a player NOW. This is MANDATORY - you cannot skip voting.\n\n"
                                       f"QUESTION: Who do you vote to kill? Choose one from: [{alive_options}]\n\n"
                                       f"ANSWER FORMAT: Your ENTIRE response must be EXACTLY and ONLY: 'I vote to kill Player X' where X is a player number.\n\n"
                                       f"RULES:\n"
                                       f"- Do NOT add any discussion, explanation, reasoning, or other text\n"
                                       f"- Do NOT say 'pass' or skip voting\n"
                                       f"- Do NOT say 'I think' or 'Based on' or any other words\n"
                                       f"- If you do not vote correctly, a random player will be eliminated\n\n"
                                       f"CORRECT EXAMPLE: 'I vote to kill Player 4'\n"
                                       f"WRONG EXAMPLES:\n"
                                       f"- 'Based on the discussion, I vote to kill Player 4'\n"
                                       f"- 'Let's vote to kill Player 4'\n"
                                       f"- 'I think Player 4 is suspicious, I vote to kill Player 4'\n"
                                       f"- Any response that contains more than just 'I vote to kill Player X'")

                        # 获取可用玩家列表
                        request.append({"role": "system", "content": (f"{vote_guidance}\n\n"
                                           f"FINAL REMINDER: You are in the VOTING phase. You MUST vote NOW.\n\n"
                                           f"QUESTION: Who do you vote to kill? Choose one from: [{alive_options}]\n\n"
                                           f"YOUR RESPONSE MUST BE EXACTLY: 'I vote to kill Player X'\n\n"
                                           f"DO NOT:\n"
                                           f"- Add any discussion, thinking, or explanation\n"
                                           f"- Say 'pass' or skip voting\n"
                                           f"- Include any other words before or after the vote statement\n\n"
                                           f"ONLY OUTPUT: 'I vote to kill Player X'")})
                    else:
                        # 检查vote_guidance是否包含CRITICAL VOTING，如果是，添加FINAL REMINDER
                        if "CRITICAL VOTING" in vote_guidance or "VOTING ACTION" in vote_guidance:
                            alive_options = ", ".join([p for p in alives if p != agent_name and p != "pass"]) if alives else "Player 1, Player 2, Player 3, Player 4, Player 5, Player 6, Player 7"
                            request.append({"role": "system", "content": (f"{vote_guidance}\n\n"
                                               f"FINAL REMINDER: You are in the VOTING phase. You MUST vote NOW.\n\n"
                                               f"QUESTION: Who do you vote to kill? Choose one from: [{alive_options}]\n\n"
                                               f"YOUR RESPONSE MUST BE EXACTLY: 'I vote to kill Player X'\n\n"
                                               f"DO NOT:\n"
                                               f"- Add any discussion, thinking, or explanation\n"
                                               f"- Say 'pass' or skip voting\n"
                                               f"- Include any other words before or after the vote statement\n\n"
                                               f"ONLY OUTPUT: 'I vote to kill Player X'")})
                        else:
                            request.append({"role": "system", "content": (f"Now it's the {turns}-th {day_night}. {vote_guidance} "
                                                                         f"CRITICAL: Do NOT explicitly reveal your role in your speech. Use normal, natural language. "
                                                                         f"Never say phrases like 'I am a werewolf', 'as a werewolf', 'being a werewolf', 'I need to blend in', or any phrase that reveals your role or that you are hiding something. "
                                                                         f"Respond directly with your action or talking (no thinking process).")})
                Temp = 0.0
            else:
                request.append({"role": "system", "content": f"Now it's the {turns}-th {day_night}. Think about what to say in your talking based on the context. "
                                                             f"CRITICAL: Do NOT explicitly reveal your role in your speech. Use normal, natural language. "
                                                             f"Never say phrases like 'I am a werewolf', 'as a werewolf', 'being a werewolf', 'I need to blend in', or any phrase that reveals your role or that you are hiding something. "
                                                             f"Speak naturally as a player trying to find werewolves or defend yourself. "
                                                             f"Respond directly with your talking content (no thinking process)."})
                '''request.append({"role": "system", "content": f"Combining the conversations, reflections above, assuming you are {agent_name}, the {role}, "
                                f"continue to talk with few concise sentences. You'd better not reveal your role, because there may be your enemies in other players.{END_OF_MESSAGE}"})'''
                Temp = arg.temperature
        else:
            request = [system_prompt]
            good_exps = '\n'.join(exps[0])
            task_content = _get_task_content(task)
            if "Choose" in task_content or "choose" in task_content or "vote to" in task_content or "Yes, No" in task_content:
                request.append({'role': 'user', 'content': f"I retrieve some historical experience similar to current situation that I am facing. "
                                f"There is one bad experience:\n\n{exps[1]}\n\nAnd there are also a set of experience that may consist of good ones:\n\n{good_exps}\n\n"
                                "Please help me analyze the differences between these experiences and identify the good ones from the set of experiences. "
                                "The difference is mainly about voting to kill someone or to pass, choosing to protect someone or to pass, using drugs or not. "
                                "What does the experience set do but the bad experience does not do? "
                                "Indicate in second person what is the best way for the player to do under such reflection. Clearly indicate whether to vote, protect or use drugs without any prerequisites. "
                                "For example 1: The experience set involves choosing to protect someone, while the bad experience involves not protecting anyone and choosing to pass in contrast. "
                                "The best way for you to do under such reflection is to choose someone to protect based on your analysis.\n"
                                "For example 2: The bad experience choose to pass the voting, and all the experience in the experience set choose to pass as well. "
                                "The best way for you to do under such reflection is to observe and analyse the identity of other players.\n"
                                "No more than 1 sentence. If there is no obvious difference between them, only generate 'No useful experience can be used.'.<EOS>"})
            else:
                request.append({'role': 'user', 'content': f"I retrieve some historical experience similar to current situation that I am facing. "
                            f"There is one bad experience:\n\n{exps[1]}\n\nAnd there are also a set of experience that may consist of good ones:\n\n{good_exps}\n\n"
                            "According to the game result, good experience may be better than bad experience and lead game victory faster than bad experience. "
                            "Compare and find the difference between the bad experience and the experience set, this is the key to victory. Ignore the player name and think what good experience set do but bad experience not do and "
                            "do not say to me. Indicate in second person what is the best way for the player to do under such reflection? For example: The best "
                            "way for you to do under such reflection is to...\nNo more than 1 sentence. If there is no obvious difference between them, only "
                            "generate 'No useful experience can be used.'.<EOS>"})
            print(f"request2: {request}", file=sys.stderr)
            response = self._get_response(request, conn_method, max_tokens=200, speaker=agent_name)
            print(f"response2: {response}", file=sys.stderr)
            response = re.sub(rf"^\s*(\[)?[a-zA-Z0-9\s]*(\])?:\s*", "", response)
            if re.search('The best way.*', response):
                response = re.search('The best way.*', response).group()
            response = re.sub(r"(\sor.*)(\.)", r'\2', response)
            exp = re.sub(rf"{END_OF_MESSAGE}$", "", response).strip()

            request = [system_prompt] + conversations + [reflexions] + [task]
            task_content_lower_exp = _get_task_content(task).lower()
            task_content = _get_task_content(task)
            if "Choose" in task_content or "choose" in task_content or "vote to" in task_content:
                # 检查是否是女巫毒药阶段
                if role == "witch" and day_night == "night" and "poison" in task_content_lower_exp and "who are you going to kill" in task_content_lower_exp:
                    request.append({"role": "system", "content": f"Now it's the {turns}-th {day_night}. Based on the context and reflection, decide who to poison. Besides, there may be history experience you can refer to: {exp} Your response must end with 'I choose Player X' or 'I choose pass'. Do NOT include thinking process."})
                else:
                    request.append({"role": "system", "content": f"Now it's the {turns}-th {day_night}. Think about which to choose based on the context, especially the just now reflection. "
                                    f"Besides, there may be history experience you can refer to: {exp} Respond directly with your choice (no thinking process)."})
                Temp = 0.0
            elif "Yes, No" in task_content:
                # 女巫救人阶段：只输出Yes或No
                if role == "witch" and day_night == "night" and ("antidote" in task_content_lower_exp or ("save" in task_content_lower_exp and "will be killed" in task_content_lower_exp)):
                    request.append({"role": "system", "content": f"Now it's the {turns}-th {day_night}. Based on the context and reflection, decide whether to save with antidote. Your response must be EXACTLY 'Yes' or 'No' only. Do NOT include any thinking process or explanation."})
                else:
                    request.append({"role": "system", "content": f"Now it's the {turns}-th {day_night}. Think about which to choose based on the context, especially the just now reflection. "
                                    f"Besides, there may be history experience you can refer to: {exp} Respond directly with your choice (no thinking process)."})
                Temp = 0.0
            else:
                request.append({"role": "system", "content": f"Now it's the {turns}-th {day_night}. Think about what to say in your talking based on the context. "
                                f"Besides, there may be history experience you can refer to: {exp} "
                                f"CRITICAL: Do NOT explicitly reveal your role in your speech. Use normal, natural language. "
                                f"Never say phrases like 'I am a werewolf', 'as a werewolf', 'being a werewolf', 'I need to blend in', or any phrase that reveals your role or that you are hiding something. "
                                f"Speak naturally as a player trying to find werewolves or defend yourself. "
                                f"Respond directly with your talking content (no thinking process)."})
                Temp = arg.temperature

            if arg:
                f.write(f"- **Exps**: {exp.strip()} \n")
        print(f"request: {request}", file=sys.stderr)
        # 投票阶段使用更小的max_tokens，强制简短响应
        task_content = _get_task_content(task)
        task_content_lower = task_content.lower()
        voting_keywords = ("Choose" in task_content or "choose" in task_content or "vote to" in task_content or "vote" in task_content)

        is_voting_phase = task_content_lower and voting_keywords and (("vote" in task_content_lower and "choose one" in task_content_lower) or
                                                   ("vote for killing" in task_content_lower) or
                                                   ("which player should be voted" in task_content_lower) or
                                                   ("which of the players should be voted" in task_content_lower) or
                                                   ("continue voting" in task_content_lower) or
                                                   ("voting phase" in task_content_lower) or
                                                   ("vote and tell" in task_content_lower))
        max_tokens_for_response = 50 if is_voting_phase else 400  # 投票阶段限制为50 tokens
        response = self._get_response(request, conn_method, T=Temp, max_tokens=max_tokens_for_response, speaker=agent_name)
        print(f"raw response: {response}", file=sys.stderr)
        response = re.sub(rf"^\s*(\[)?[a-zA-Z0-9\s]*(\])?:\s*", "", response)
        response = re.sub(rf"{END_OF_MESSAGE}$", "", response).strip()
        if arg:
            f.write(f"- **CoT**: {response}  \n\n")
        # print(response)
        response = self.extract_text(response)
        response = re.sub(rf"^\s*(\[)?[a-zA-Z0-9\s]*(\])?:\s*", "", response)
        # print(response)
        response = response.replace('\n', ' ')
        response = response.replace("'''.", '')
        response = response.strip('"')
        response = response.strip("'")

        # 添加响应验证和修正
        response = self._validate_and_correct_response(response, agent_name, role, alives, history_messages)

        # 投票阶段后处理：如果响应中没有投票格式，强制提取或修正
        task_content_lower = _get_task_content(task).lower()
        is_voting_phase = (("vote" in task_content_lower and "choose one" in task_content_lower) or
                         ("vote for killing" in task_content_lower) or
                         ("which player should be voted" in task_content_lower) or
                         ("which of the players should be voted" in task_content_lower) or
                         ("continue voting" in task_content_lower) or
                         ("voting phase" in task_content_lower) or
                         ("vote and tell" in task_content_lower))

        if is_voting_phase:
            # 检查响应中是否包含投票格式
            vote_patterns = [
                r"i\s+vote\s+to\s+kill\s+(player\s*\d+)",
                r"i\s+vote\s+for\s+(player\s*\d+)",
                r"i\s+choose\s+(player\s*\d+)",
                r"i\s+vote\s+(player\s*\d+)",
            ]

            has_vote = False
            for pattern in vote_patterns:
                if re.search(pattern, response, re.IGNORECASE):
                    has_vote = True
                    break

            # 如果没有投票格式，尝试从响应中提取玩家名称并添加投票语句
            if not has_vote and alives:
                # 尝试从响应中提取提到的玩家
                player_mentions = []
                for player in alives:
                    if player != agent_name and player != "pass":
                        if re.search(rf"\b{re.escape(player)}\b", response, re.IGNORECASE):
                            player_mentions.append(player)

                if player_mentions:
                    # 使用最后提到的玩家作为投票目标
                    vote_target = player_mentions[-1]
                    response = f"I vote to kill {vote_target}"
                    print(f"WARNING: No vote format found, extracted and added vote: {response}", file=sys.stderr)
                else:
                    # 如果无法提取，使用第一个可用玩家（除了自己）
                    other_players = [p for p in alives if p != agent_name and p != "pass"]
                    if other_players:
                        vote_target = other_players[0]
                        response = f"I vote to kill {vote_target}"
                        print(f"WARNING: No vote format found, defaulting to: {response}", file=sys.stderr)

        print(f"response: {response}", file=sys.stderr)

        game_number = arg.current_game_number if arg and arg.current_game_number else 0
        exp_new = Message(agent_name, [reflexions["content"].replace(END_OF_MESSAGE, '').split(': ', maxsplit=1)[1], response, 0, branch], turn=game_number, msg_type="exp")
        msgs.append_message(exp_new)

        task_content = _get_task_content(task)
        if "Choose" in task_content or "choose" in task_content or "vote to" in task_content or "Yes, No" in task_content:
            response = f"({turns}-th {day_night}) " + response

        if arg:
            f.write(f"- **Final**: {response}  \n\n")
            f.close()

            # 添加结构化日志记录到model_reply.log
            self._write_structured_log(arg, agent_name, role, turns, day_night, response, task, alives, empathy_data=empathy_data if 'empathy_data' in locals() else None, reflection_context=reflection_context if 'reflection_context' in locals() else None)

        # 确保响应被记录到 model_reply.log（如果之前没有记录）
        try:
            self._log_model_reply(agent_name, self._sanitize_model_reply(response))
        except Exception as e:
            print(f"[WARNING] Failed to log final response: {e}", file=sys.stderr)

        return response

    def _summarize_empathy_for_log(self, empathy_data):
        if not empathy_data:
            return "{}"
        try:
            items = []
            board = empathy_data.get("_game", {}) if isinstance(empathy_data, dict) else {}
            if board:
                ts = board.get("top_suspects", [])[:3]
                tt = board.get("top_trust", [])[:3]
                bs = board.get("board_signal", {})
                if ts:
                    items.append("top_suspects=" + ", ".join(
                        f"{x.get('player')}[h={x.get('hard_wolf_prob', 0):.2f},i={x.get('information_gain', 0):.2f},b={'1' if x.get('bandwagon_risk') else '0'},act={x.get('recommended_action', 'observe')}]"
                        for x in ts
                    ))
                if tt:
                    items.append("top_trust=" + ", ".join(
                        f"{x.get('player')}[pub={x.get('public_trust', 0):.2f},t={x.get('trust_score', 0):.2f}]"
                        for x in tt
                    ))
                if bs:
                    items.append(
                        f"board[info_dense={bs.get('info_dense', 0)},hard_claims={bs.get('hard_claims', 0)},support_links={bs.get('support_links', 0)},accusations={bs.get('accusation_links', 0)}]"
                    )
            for player, report in list(empathy_data.items()):
                if player.startswith("_"):
                    continue
                items.append(
                    f"{player}:wolf={report.get('hard_wolf_prob', report.get('role_probability', {}).get('werewolf', 0.0)):.2f},"
                    f"info={report.get('information_gain', 0.0):.2f},"
                    f"press={report.get('current_round_vote_pressure', 0.0):.2f},"
                    f"align={report.get('speech_vote_consistency', 1.0):.2f},"
                    f"act={report.get('recommended_action', 'observe')}"
                )
            return " | ".join(items[:8]) if items else "{}"
        except Exception:
            return "{}"

    def _format_empathy_detail_for_log(self, empathy_data):
        if not empathy_data or not isinstance(empathy_data, dict):
            return "{}"
        try:
            lines = []
            board = empathy_data.get("_game", {}) if isinstance(empathy_data, dict) else {}
            if board:
                lines.append(f"board={json.dumps(board, ensure_ascii=False)[:1000]}")
            reports = empathy_data.get("player_reports", empathy_data)
            if isinstance(reports, dict):
                for player, report in list(reports.items())[:7]:
                    if str(player).startswith("_"):
                        continue
                    if not isinstance(report, dict):
                        continue
                    slim = {
                        "hard_wolf_prob": report.get("hard_wolf_prob", report.get("role_probability", {}).get("werewolf", 0.0)),
                        "soft_wolf_prob": report.get("soft_wolf_prob", 0.0),
                        "public_trust": report.get("public_trust", report.get("trust_score", 0.0)),
                        "trust": report.get("trust", 0.0),
                        "information_gain": report.get("information_gain", 0.0),
                        "vote_pressure": report.get("current_round_vote_pressure", 0.0),
                        "speech_vote_consistency": report.get("speech_vote_consistency", 1.0),
                        "recommended_action": report.get("recommended_action", "observe"),
                        "semantic_memory": report.get("semantic_memory", "")[:220],
                        "uncertainty_notes": report.get("uncertainty_notes", "")[:220],
                        "signal": report.get("signal", {}),
                        "supports": report.get("supports", [])[:3],
                        "supported_by": report.get("supported_by", [])[:3],
                        "clears": report.get("clears", [])[:3],
                        "cleared_by": report.get("cleared_by", [])[:3],
                        "opposes": report.get("opposes", [])[:3],
                        "opposed_by": report.get("opposed_by", [])[:3],
                    }
                    lines.append(f"{player}={json.dumps(slim, ensure_ascii=False)[:1200]}")
            return "\n".join(lines) if lines else "{}"
        except Exception:
            return "{}"

    def _log_model_reply(self, speaker: str, content: str):
        """记录模型回复到 model_reply.log，格式：[{speaker}] {content}"""
        log_line = f"[{speaker}] {content}"
        print(log_line, file=sys.stderr)
        try:
            with open("model_reply.log", "a", encoding="utf-8") as f:
                f.write(log_line + "\n")
        except Exception as e:
            print(f"[WARNING] Failed to write to model_reply.log: {e}", file=sys.stderr)
