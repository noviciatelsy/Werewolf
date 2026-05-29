from typing import List, Union
import re
from tenacity import RetryError
import logging
import uuid
from abc import abstractmethod
import os

LOG_FILE = "werewolf_dialogue.txt"

from .MCTS import MCTS, fast_evaluate_state, generate_llm_speech, llm_empathy_extract
from .backends import IntelligenceBackend, load_backend
# GameState 现在统一从 MCTS 模块导入
from .MCTS import GameState
from .message import SYSTEM_NAME
from .config import AgentConfig, Configurable, BackendConfig
from .message import Message, MessagePool, QuestionPool

# A special signal sent by the player to indicate that it is not possible to continue the conversation, and it requests to end the conversation.
# It contains a random UUID string to avoid being exploited by any of the players.
SIGNAL_END_OF_CONVERSATION = f"<<<<<<END_OF_CONVERSATION>>>>>>{uuid.uuid4()}"


class Agent(Configurable):

    @abstractmethod
    def __init__(self, name: str, role_desc: str, global_prompt: str = None, *args, **kwargs):
        super().__init__(name=name, role_desc=role_desc, global_prompt=global_prompt, **kwargs)
        self.name = name
        self.role_desc = role_desc
        self.global_prompt = global_prompt


class Player(Agent):
    """
    Player of the game. It can takes the observation from the environment and return an action
    """

    def __init__(self, args, name: str, role_desc: str, backend: Union[BackendConfig, IntelligenceBackend],
                 global_prompt: str = None, **kwargs):

        if isinstance(backend, BackendConfig):
            backend_config = backend
            backend = load_backend(backend_config, args)
        elif isinstance(backend, IntelligenceBackend):
            backend_config = backend.to_config()
        else:
            raise ValueError(f"backend must be a BackendConfig or an IntelligenceBackend, but got {type(backend)}")

        assert name != SYSTEM_NAME, f"Player name cannot be {SYSTEM_NAME}, which is reserved for the system."

        # Register the fields in the _config
        super().__init__(name=name, role_desc=role_desc, backend=backend_config,
                         global_prompt=global_prompt, **kwargs)

        self.backend = backend

    def to_config(self) -> AgentConfig:
        return AgentConfig(
            name=self.name,
            role_desc=self.role_desc,
            backend=self.backend.to_config(),
            global_prompt=self.global_prompt,
        )

    # def __call__(self, args, observation: List[Message], messages: MessagePool, questions: QuestionPool, state = (0, "daytime", "", [])) -> str:
    #     """
    #     Call the agents to generate a response (equivalent to taking an action).
    #     """
    #     try:
    #         if self.name == "Player 1" and args and args.human_in_combat:
    #             response = input("Now you say: ")
    #             with open(os.path.join(args.logs_path_to, str(args.current_game_number) + ".md"), "w") as f:
    #                 f.write(f"Player 1: {response}  " + "\n")
    #         else:
    #             response = self.backend.query(args, agent_name=self.name, role_desc=self.role_desc,
    #                                         history_messages=observation, global_prompt=self.global_prompt,
    #                                         request_msg=None, msgs=messages, ques=questions, turns=state[0],
    #                                         day_night=state[1], role=state[2], alives=state[3])
    #     except RetryError as e:
    #         logging.warning(f"Agent {self.name} failed to generate a response. "
    #                         f"Error: {e.last_attempt.exception()}. "
    #                         f"Sending signal to end the conversation.")
    #         response = SIGNAL_END_OF_CONVERSATION
    #
    #     return response

    def __call__(self, args, observation: List[Message], messages: MessagePool, questions: QuestionPool,
                 state=(0, "daytime", "", [])) -> str:
        """
        Call the agents to generate a response (equivalent to taking an action).
        state = (turn, phase, role, alive_list)
        """
        try:
            if self.name == "Player 1" and args and hasattr(args, 'human_in_combat') and args.human_in_combat:
                response = input("Now you say: ")
                with open(os.path.join(args.logs_path_to, str(args.current_game_number) + ".md"), "a") as f:
                    f.write(f"Player 1: {response}  " + "\n")
            else:
                # 关键修复：确保role参数正确传递，并添加身份验证
                current_role = state[2] if len(state) > 2 else ""
                
                # 添加身份验证逻辑
                if current_role and current_role != "unknown":
                    # 在prompt中明确当前agent的身份
                    enhanced_prompt = f"You are {self.name}, and you are a {current_role}. Remember this throughout the conversation. Do NOT confuse your identity with other players."
                    if self.global_prompt:
                        self.global_prompt = enhanced_prompt + "\n" + self.global_prompt
                    else:
                        self.global_prompt = enhanced_prompt
                
                response = self.backend.query(args, agent_name=self.name, role_desc=self.role_desc,
                                            history_messages=observation, global_prompt=self.global_prompt,
                                            request_msg=None, msgs=messages, ques=questions, turns=state[0],
                                            day_night=state[1], role=current_role, alives=state[3])
                
                # 后处理：验证和修正响应
                response = self._validate_and_correct_response(response, self.name, current_role)
                
        except RetryError as e:
            logging.warning(f"Agent {self.name} failed to generate a response. "
                            f"Error: {e.last_attempt.exception()}. "
                            f"Sending signal to end the conversation.")
            response = SIGNAL_END_OF_CONVERSATION
        except Exception as e:
            logging.warning(f"Agent {self.name} failed to generate a response. "
                            f"Error: {e}. Sending signal to end the conversation.")
            response = SIGNAL_END_OF_CONVERSATION

        return response
    
    def _validate_and_correct_response(self, response, agent_name, role):
        """验证和修正agent响应"""
        import re
        
        # 检查身份混淆
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
        
        # 防止狼人投票杀死自己
        if role == "werewolf":
            vote_pattern = r"I vote to kill (Player \d+)"
            match = re.search(vote_pattern, response, re.IGNORECASE)
            if match and match.group(1) == agent_name:
                response = re.sub(vote_pattern, "I vote to kill Player 1", response, flags=re.IGNORECASE)
        
        return response


class Moderator(Player):
    """
    A special type of player that moderates the conversation (usually used as a component of environment).
    """

    def __init__(self, role_desc: str, backend: Union[BackendConfig, IntelligenceBackend],
                 terminal_condition: str, global_prompt: str = None, **kwargs):
        name = "Moderator"
        super().__init__(name=name, role_desc=role_desc, backend=backend, global_prompt=global_prompt, **kwargs)

        self.terminal_condition = terminal_condition

    def to_config(self) -> AgentConfig:
        return AgentConfig(
            name=self.name,
            role_desc=self.role_desc,
            backend=self.backend.to_config(),
            terminal_condition=self.terminal_condition,
            global_prompt=self.global_prompt,
        )

    def is_terminal(self, history: List[Message], *args, **kwargs) -> bool:
        """
        check whether the conversation is over
        """
        # If the last message is the signal, then the conversation is over
        if history[-1].content == SIGNAL_END_OF_CONVERSATION:
            return True

        try:
            request_msg = Message(agent_name=self.name, content=self.terminal_condition, turn=-1)
            response = self.backend.query(agent_name=self.name, role_desc=self.role_desc, history_messages=history,
                                          global_prompt=self.global_prompt, request_msg=request_msg, *args, **kwargs)
        except RetryError as e:
            logging.warning(f"Agent {self.name} failed to generate a response. "
                            f"Error: {e.last_attempt.exception()}.")
            return True

        if re.match(r"yes|y|yea|yeah|yep|yup|sure|ok|okay|alright", response, re.IGNORECASE):
            # print(f"Decision: {response}. Conversation is ended by moderator.")
            return True
        else:
            return False
