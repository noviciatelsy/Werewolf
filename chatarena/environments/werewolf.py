from typing import List, Dict, Union, Tuple
import random
import re
import json
import logging
import sys

from .base import Environment, TimeStep
from ..message import Message, MessagePool, Question, QuestionPool
from ..agent import SIGNAL_END_OF_CONVERSATION
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.DEBUG)


class Werewolf(Environment):
    type_name = "werewolf"

    @property
    def current_phase(self):
        return self._current_phase

    @property
    def current_turn(self):
        return self._current_turn

    @property
    def alive_list(self):
        return self._alive_list

    @property
    def dead_list(self):
        return [self.player_names[i] for i, alive in enumerate(self._is_alive) if not alive]

    @property
    def characters(self):
        return self._characters

    @property
    def last_actions(self):
        if self.message_pool._messages:
            return [msg.content for msg in self.message_pool._messages[-10:]]
        return []

    def __init__(self, args, player_names: List[str], topic_codes: Dict[str, List[str]] = None, **kwargs):
        super().__init__(player_names=player_names, topic_codes=topic_codes, **kwargs)
        
        self.args = args
        self.message_pool = MessagePool(args)
        self.question_pool = QuestionPool(args)

        if not args or (args and not args.role_config):
            with open("./config/1.json", "r") as f:
                self._character_config = {k: v for k, v in json.load(f).items() if v != 0}
        else:
            with open(args.role_config, "r") as f:
                self._character_config = {k: v for k, v in json.load(f).items() if v != 0}
        self._alive_list = self.player_names + ["pass"]
        self._characters = [k for k, v in self._character_config.items() for _ in range(v)]
        random.shuffle(self._characters)        # ["werewolf", "seer", "witch", "villager"...]
        self._is_alive = [True for _ in range(len(self._characters))]
        self._identity_mapping = {              # {"werewolf":[2,3], "villager":[1]...}
            "werewolf": [],
            "villager": [],
            "guard": [],
            "witch": [],
            "seer": [],
            "hunter": [],
            "idiot": [],
            "thief": [],
            "cupid": [],
            "girl": [],
            "sheriff": [],
            "elder": [],
            "scapegoat": [],
            "piper": []
        }
        for i, x in enumerate(self._characters):
            self._identity_mapping[x].append(i)
        self.werewolf_numbers = len(self._identity_mapping["werewolf"])
        # The following: [2,3,1...]
        self._night_order = [x for k, v in self._identity_mapping.items() if k != "villager" and len(v) > 0 for x in v]
        # The following: [3,4,1...] must be shuffled exery day!
        self._day_order = [i for i in range(len(self._characters))]
        random.shuffle(self._day_order)

        self._current_turn = 0
        self._next_player_idx = 0
        self._current_phase = "night"           # night, daytime
        self._number_of_nights = 0
        self._initialized = False
        self._players_votes = None
        self._lives = 0
        self._number_of_rounds = 0
        self._guard_to = None
        self._witch_antidote = True
        self._witch_poison = True
        self._witch_antidote_to = None
        self._witch_poison_to = None
        self._current_first_alive = -1
        self._killed_list = []
        self._night_kill_list = []

        # self.reset()  # To initialize the game

    def _print_infos(self):
        print(f"alive list: {self._alive_list}", file=sys.stderr)
        print(f"is alive: {self._is_alive}", file=sys.stderr)
        print(f"identity mapping: {str(self._identity_mapping)}", file=sys.stderr)
        print(f"night order: {self._night_order}", file=sys.stderr)
        print(f"day order: {self._day_order}", file=sys.stderr)

    def get_next_player(self) -> str:
        """
            get the next player
        """
        if self._number_of_rounds == 9999:             # last statement
            return self.player_names[self._next_player_idx]

        if self._current_phase == "daytime":
            return self.player_names[self._day_order[self._next_player_idx]]
        elif self._current_phase == "night":
            return self.player_names[self._night_order[self._next_player_idx]]

    def reset(self):
        self.message_pool.reset()

        self._current_turn = 0
        self._next_player_idx = 0
        self._current_phase = "night"
        self._number_of_nights = 0
        self._lives = len(self._characters)
        self._players_votes = {name: 0 for name in self.player_names}
        self._players_votes["pass"] = 0

        self._description = [str(len(v)) + ' ' + k + "(s), " for k, v in self._identity_mapping.items() if len(v) > 0]
        _list = list(self._description[-1])
        _list.pop()
        _list[-1] = '.'
        self._description[-1] = ''.join(_list)
        self._description = ''.join(self._description)

        self._print_infos()
        # 首先强调身份是随机分配的
        self._moderator_speak("IMPORTANT: All player identities are RANDOMLY assigned at the start of each game. "
                              "Do NOT assume identities from previous games. Each game has a fresh, random identity assignment. "
                              "Only trust the identity I tell you in THIS game.", importance=6)
        self._moderator_speak(f"Now the game starts! In this game, we have {self._description}")
        for i, player in enumerate(self.player_names):
            self._moderator_speak(f"IMPORTANT: Your identity has been RANDOMLY assigned for THIS game. "
                                  f"You are {self._characters[i]}! "
                                  f"Do NOT confuse your identity with other players or assume identities from previous games. "
                                  f"In THIS game, you are {player} and your role is {self._characters[i]}. "
                                  f"Remember: identities are randomly assigned each game, so your role may be different from previous games.", 
                                  visible_to=player, importance=6)
        self._moderator_speak("It's dark, everyone close your eyes. I will talk with you/your team secretly at night.")
        werewolves = ', '.join([self.player_names[i] for i in self._identity_mapping["werewolf"]])
        print(f"wolves: {werewolves}", file=sys.stderr)
        print(f"alive list: {self._alive_list}", file=sys.stderr)
        self._moderator_speak(f"Werewolves, please open your eyes! "
                              f"I secrecly tell you that {werewolves} are all of the {len(self._identity_mapping['werewolf'])} werewolves! "
                              f"Keep in mind you are teammates. The rest players are not werewolves. Now vote and tell your teammates which of the players should be killed tonight. The first werewolf, "
                              f"you, randomly choose one from the following living options please: [{', '.join(self._alive_list)}]. ",
                              # f"For example: I choose Player...",
                              visible_to=[self.player_names[i] for i in self._identity_mapping["werewolf"]],
                              importance=5)
        self._current_turn = 1

        self._initialized = True
        init_timestep = TimeStep(observation=self.get_observation(),
                                 reward=self.get_zero_rewards(),
                                 terminal=False)

        return init_timestep

    def print(self):
        self.message_pool.print()

    def get_observation(self, player_name=None) -> List[Message]:
        """
            get observation for the player
        """
        n_last = self.args.message_window if self.args and self.args.message_window else 10
        if player_name is None:
            return self.message_pool.get_all_messages()
        else:
            # return self.message_pool.get_visible_messages(player_name, turn=self._current_turn)
            return self.message_pool.get_last_k_messages(player_name, self._current_turn, n_last)

    def _moderator_speak(self, text: str, visible_to: Union[str, List[str]] = "all", importance = 1):
        """
            moderator say something
        """
        message = Message(agent_name="Moderator", content=text, turn=self._current_turn, visible_to=visible_to, importance=importance)
        self.message_pool.append_message(message)

    def _get_number_of_people(self) -> Tuple[int, int, int]:
        _all = len(self._characters)
        _live = self._lives
        _werewolf = len(self._identity_mapping["werewolf"])
        return _all, _live, _werewolf

    def _get_next_alive(self, crt: int) -> int:         # find in self._day_order, crt is self._next_player_idx
        number_of_players = len(self._characters)
        for idx in range(crt + 1, crt + number_of_players):
            current_idx = idx % number_of_players
            if self._is_alive[self._day_order[current_idx]]:
                return current_idx
        return crt

    def _text2vote(self, text, player_name=None) -> str:
        """
        convert text to vote, return a player's name
        Improved version with better pattern matching
        """
        # 处理END_OF_CONVERSATION信号
        if "END_OF_CONVERSATION" in text or "<<<<<<END_OF_CONVERSATION" in text:
            # 如果上层已经把结束信号带进来了，这里保底返回一个合法非空动作，避免吞掉模型回复
            return "I choose pass"
        
        # 清理文本，保留原始文本用于匹配
        original_text = text
        text_lower = text.lower().strip()
        
        # 方法1: 优先匹配明确的投票模式（最高优先级）
        # 匹配 "I vote to kill Player X" 或 "I choose Player X" 等明确格式
        explicit_vote_patterns = [
            r"i\s+vote\s+to\s+kill\s+(player\s*\d+)",
            r"i\s+vote\s+for\s+(player\s*\d+)",
            r"i\s+choose\s+(player\s*\d+)",
            r"i\s+vote\s+(player\s*\d+)",
            r"vote\s+to\s+kill\s+(player\s*\d+)",
            r"vote\s+for\s+(player\s*\d+)",
            r"choose\s+(player\s*\d+)",
            r"kill\s+(player\s*\d+)",
            r"eliminate\s+(player\s*\d+)",
            r"protect\s+(player\s*\d+)",
            r"verify\s+(player\s*\d+)",
            r"target\s+(player\s*\d+)",
            r"vote\s+(player\s*\d+)"
        ]
        
        for pattern in explicit_vote_patterns:
            match = re.search(pattern, text_lower)
            if match:
                player_ref = match.group(1).strip()
                # 尝试匹配玩家名称
                for name in self.player_names:
                    name_lower = name.lower()
                    # 精确匹配
                    if name_lower == player_ref or name_lower.replace(" ", "") == player_ref.replace(" ", ""):
                        print(f"Matched player '{name}' from pattern '{pattern}' in text: '{text}'", file=sys.stderr)
                        return name
                    # 部分匹配（Player X格式）
                    if f"player {name_lower.split()[-1]}" in player_ref or name_lower.split()[-1] in player_ref:
                        print(f"Matched player '{name}' from pattern '{pattern}' in text: '{text}'", file=sys.stderr)
                        return name
        
        # 方法2: 匹配文本中的玩家名称（按出现顺序，优先选择最后出现的）
        # 分割文本为句子
        sentences = re.split(r'[!.?:]', original_text)
        # 从后往前查找（最后一个句子优先级最高）
        for sentence in reversed(sentences):
            sentence_lower = sentence.lower().strip()
            if not sentence_lower:
                continue
            
            # 检查是否包含"myself"
            if "myself" in sentence_lower and player_name is not None:
                print(f"Matched 'myself' to player '{player_name}' in text: '{text}'", file=sys.stderr)
                return player_name
            
            # 检查是否包含玩家名称
            for name in self.player_names:
                name_lower = name.lower()
                name_variants = [
                    name_lower,
                    name_lower.replace(" ", ""),
                    name_lower.replace(" ", "_"),
                    f"player {name_lower.split()[-1]}" if " " in name_lower else name_lower
                ]
                # 检查是否在句子中包含玩家名称（作为完整单词）
                for variant in name_variants:
                    # 使用单词边界确保完整匹配
                    if re.search(r'\b' + re.escape(variant) + r'\b', sentence_lower):
                        print(f"Matched player '{name}' from sentence: '{sentence}' in text: '{text}'", file=sys.stderr)
                        return name
        
        # 方法3: 在整个文本中查找玩家名称（备用方法）
        for name in self.player_names:
            name_lower = name.lower()
            name_variants = [
                name_lower,
                name_lower.replace(" ", ""),
                name_lower.replace(" ", "_"),
                f"player {name_lower.split()[-1]}" if " " in name_lower else name_lower
            ]
            for variant in name_variants:
                if re.search(r'\b' + re.escape(variant) + r'\b', text_lower):
                    print(f"Matched player '{name}' from text: '{text}'", file=sys.stderr)
                    return name
        
        # 如果没有找到，返回pass
        print(f"No player matched in text: '{text}', returning 'pass'", file=sys.stderr)
        return "pass"

    def get_rewards(self, chameleon_win: bool) -> Dict[str, float]:
        pass

    def is_terminal(self) -> bool:
        """
            check if the conversation is over
        """
        # If the last message is the signal, then the conversation is over
        if self.message_pool.last_message.content == SIGNAL_END_OF_CONVERSATION:
            return True

    def _kill_by_name(self, kill_list: List[str]):
        if '' in kill_list:
            kill_list.remove('')
        if "pass" in kill_list:
            kill_list.remove("pass")
        killed_identity = [self.player_names.index(name) for name in kill_list]
        for idx in killed_identity:
            if not self._is_alive[idx]:  # 修复：检查玩家是否还活着
                continue
            self._is_alive[idx] = False
            # 修复：检查玩家是否在alive_list中再移除
            if self.player_names[idx] in self._alive_list:
                self._alive_list.remove(self.player_names[idx])
            identity = self._characters[idx]
            # 修复：检查身份映射中是否存在该索引
            if idx in self._identity_mapping[identity]:
                self._identity_mapping[identity].remove(idx)
            '''if idx in self._night_order:
                self._night_order.remove(idx)'''
            self._lives -= 1

    def _judge_is_alive(self, name: str) -> bool:
        if name == "pass" or name == '':
            return False
        idx = self.player_names.index(name)
        return self._is_alive[idx]


    def _check_game_over(self):
        def _get_winner_names(is_villager=True):
            werewolf_camp = [self.player_names[idx] for idx, role in enumerate(self._characters) if role=="werewolf"]
            villager_camp = [name for name in self.player_names if name not in werewolf_camp]
            if is_villager:
                return villager_camp
            else:
                return werewolf_camp
        
        def _give_rewards(winner_names, camp):
            self.message_pool.give_rewards(winner_names)
            self.question_pool.give_rewards(last_turn=self._current_turn, camp=camp)
        
        if self._lives > 0 and len(self._identity_mapping["werewolf"]) == 0:
            self._moderator_speak("Game over, the villager wins!")
            _give_rewards(_get_winner_names(True), "villager")
            self._moderator_speak(SIGNAL_END_OF_CONVERSATION)
            return True
        # if self._current_phase == "night" and self._lives <= 2 and len(self._identity_mapping["werewolf"]) > 0:
        if len(self._identity_mapping["werewolf"]) > 0 and len(self._identity_mapping["villager"]) == 0:
            self._moderator_speak("Game over, the werewolf wins!")
            _give_rewards(_get_winner_names(False), "werewolf")
            self._moderator_speak(SIGNAL_END_OF_CONVERSATION)
            return True
        return False

    def step(self, player_name: str, action: str) -> TimeStep:
        # If not initialized, reset the environment
        if not self._initialized:
            self.reset()

        assert player_name == self.get_next_player(), f"Wrong player! It is {self.get_next_player()} turn."
        if self._current_phase == "daytime":
            assert self._get_number_of_people()[2] > 0
            rewards = self.get_zero_rewards()
            if self._number_of_rounds == 0:
                print(f"action: {action}", file=sys.stderr)
                message = Message(agent_name=player_name, content=action, turn=self._current_turn)
                self.message_pool.append_message(message)
                print(f"_next_player_idx: {self._next_player_idx}", file=sys.stderr)
                self._next_player_idx = self._get_next_alive(self._next_player_idx)
                print(f"_next_player_idx: {self._next_player_idx}", file=sys.stderr)

                print(f"_current_first_alive: {self._current_first_alive}", file=sys.stderr)
                if self._next_player_idx == self._current_first_alive:
                    self._next_player_idx = self._get_next_alive(-1)
                    print(f"_next_player_idx: {self._next_player_idx}", file=sys.stderr)
                    self._number_of_rounds = 1
                    print(f"alive list: {self._alive_list}", file=sys.stderr)
                    self._moderator_speak(f"Now you {self.get_next_player()} are asked to choose which of the players should be voted for killing based on the discussion? Don't mention your role. "
                                          f"This is the VOTING phase. You MUST vote now. Voting is MANDATORY - you cannot skip. "
                                          f"If no one votes or all votes are invalid, a random player will be eliminated. "
                                          f"You only choose one from the following living options please: [{', '.join(self._alive_list)}]. "
                                          f"Your response must be EXACTLY: 'I vote to kill Player X' where X is a player number. "
                                          f"For example: I vote to kill Player 3")
                else:
                    # 根据回合数调整提示词的激进程度
                    if self._current_turn >= 2:
                        # 第2轮及以后，强烈鼓励信息分享
                        self._moderator_speak(f"The next {self.get_next_player()}, you, "
                                              f"continue talking with other players based on your observation and reflection with few sentences. "
                                              f"CRITICAL: If you are a seer, guard, or witch, you MUST reveal your identity and share your information NOW. The village needs this information to win. "
                                              f"IMPORTANT: This is the DISCUSSION phase. Do NOT vote now. Only discuss and analyze. You will vote later when the moderator asks you to vote. "
                                              f"MANDATORY FOR SPECIAL ROLES: Share your verification results, protection information, or save/poison actions immediately. "
                                              f"MANDATORY FOR VILLAGERS: Demand information from other players. Ask who is the seer, guard, and witch. "
                                              f"REMINDER: Information sharing is ESSENTIAL for good players to win. Silence guarantees werewolf victory. "
                                              f"If you have information that can help eliminate players from suspicion (e.g., you successfully protected someone, used an antidote to save someone, or verified someone as good), sharing this information helps the village narrow down suspects and make better voting decisions.",
                                              visible_to=self.get_next_player())
                    else:
                        # 第1轮，温和鼓励信息分享
                        self._moderator_speak(f"The next {self.get_next_player()}, you, "
                                              f"continue talking with other players based on your observation and reflection with few sentences. Consider revealing your identity if it helps the village. "
                                              f"IMPORTANT: This is the DISCUSSION phase. Do NOT vote now. Only discuss and analyze. You will vote later when the moderator asks you to vote. "
                                              f"CRITICAL: Do NOT explicitly reveal your role in your speech if you are a werewolf. Use normal, natural language. "
                                              f"Never say phrases like 'I am a werewolf', 'as a werewolf', 'being a werewolf', 'I need to blend in', or any phrase that reveals your role or that you are hiding something. "
                                              f"Speak naturally as a player trying to find werewolves or defend yourself. "
                                              f"REMINDER: If you have information that can help eliminate players from suspicion (e.g., you successfully protected someone, used an antidote to save someone, or verified someone as good), sharing this information helps the village narrow down suspects and make better voting decisions.",
                                              visible_to=self.get_next_player())
            elif self._number_of_rounds == 1:
                print(f"action: {action}", file=sys.stderr)
                message = Message(agent_name=player_name, content=action, turn=self._current_turn)
                self.message_pool.append_message(message)
                vote = self._text2vote(action, player_name)
                print(f"vote result: {vote}", file=sys.stderr)
                if vote in self.player_names or vote == "pass":
                    self._players_votes[vote] += 1
                self._next_player_idx = self._get_next_alive(self._next_player_idx)
                print(f"_next_player_idx: {self._next_player_idx}", file=sys.stderr)

                print(f"_current_first_alive: {self._current_first_alive}", file=sys.stderr)
                if self._next_player_idx == self._current_first_alive:
                    to_kill = max(self._players_votes, key=self._players_votes.get)
                    print(f"to kill: {to_kill}", file=sys.stderr)
                    print(f"is alive: {self._is_alive}", file=sys.stderr)
                    if to_kill != "pass" and not self._is_alive[self.player_names.index(to_kill)]:
                        self._moderator_speak(f"Only the living can be killed, {to_kill} is dead, "
                                              f"hence no one will be killed!")
                        to_kill = "pass"
                    else:
                        print(f"players votes: {str(self._players_votes)}", file=sys.stderr)
                        for name, vote in self._players_votes.items():
                            if name != to_kill and vote == self._players_votes[to_kill]:
                                to_kill = "pass"
                                self._moderator_speak("No consensus, no one will be killed!")
                                break
                    if to_kill == "pass":
                        self._current_turn += 1
                        self._number_of_rounds = 0
                        self._players_votes = {name: 0 for name in self.player_names}
                        self._players_votes["pass"] = 0
                        self._current_phase = "night"
                        # self._night_order = [x for k, v in self._identity_mapping if len(v) > 0 for x in v]
                        self._next_player_idx = 0
                        print(f"alive list: {self._alive_list}", file=sys.stderr)
                        self._moderator_speak(f"It's dark, everyone close your eyes.")
                        self._moderator_speak(f"Werewolves, please open your eyes! "
                                              f"Now vote and tell your teammates which of the players should be killed tonight. "
                                              f"You {self.get_next_player()} only choose one from the following living options please: [{', '.join(self._alive_list)}]. ",
                                              # f"For example: I choose Player...",
                                              visible_to=[self.player_names[i] for i in self._identity_mapping["werewolf"]],
                                              importance=1 if self._judge_is_alive(self.get_next_player()) else 0)
                    else:
                        print(f"to kill: {to_kill}", file=sys.stderr)
                        self._moderator_speak(f"{to_kill} will be killed! You can make a brief last statement.", importance=6)
                        self._next_player_idx = self.player_names.index(to_kill)
                        self._number_of_rounds = 9999
                else:
                    self._moderator_speak(f"The next {self.get_next_player()}, you, continue voting the players should be killed based on the discussion? Don't mention your role. "
                                          f"This is the VOTING phase. You MUST vote now. Voting is MANDATORY - you cannot skip. "
                                          f"If no one votes or all votes are invalid, a random player will be eliminated. "
                                          f"Only choose one from the following living options please: [{', '.join(self._alive_list)}]. "
                                          f"Your response must be EXACTLY: 'I vote to kill Player X' where X is a player number. "
                                          f"For example: I vote to kill Player 3",
                                          visible_to=self.get_next_player())
            else:
                print(f"action: {action}", file=sys.stderr)
                message = Message(agent_name=player_name, content=action, turn=self._current_turn)
                self.message_pool.append_message(message)
                self._print_infos()
                self._kill_by_name([player_name])
                self._print_infos()
                if self._check_game_over():
                    return TimeStep(observation=self.get_observation(), reward=rewards, terminal=True)
                self._current_turn += 1
                self._number_of_rounds = 0
                self._players_votes = {name: 0 for name in self.player_names}
                self._players_votes["pass"] = 0
                self._current_phase = "night"
                # self._night_order = [x for k, v in self._identity_mapping.items() if k != "villager" and len(v) > 0 for x in v]
                self._current_first_alive = self._get_next_alive(-1)
                self._next_player_idx = 0
                print(f"alive list: {self._alive_list}", file=sys.stderr)
                self._moderator_speak(f"It's dark, everyone close your eyes.")
                self._moderator_speak(f"Werewolves, please open your eyes! "
                                      f"Now vote and tell your teammates which of the players should be killed tonight. "
                                      f"You {self.get_next_player()} only choose one from the following living options please: [{', '.join(self._alive_list)}]. ", 
                                      # f"For example: I choose Player...",
                                      visible_to=[self.player_names[i] for i in self._identity_mapping["werewolf"]],
                                      importance=1 if self._judge_is_alive(self.get_next_player()) else 0)

            terminal = False
            timestep = TimeStep(observation=self.get_observation(), reward=rewards, terminal=terminal)
        elif self._current_phase == "night":
            rewards = self.get_zero_rewards()
            # werewolf、guard、witch、seer
            # assert self._get_number_of_people()[2] > 0
            if self._next_player_idx < self.werewolf_numbers:
                print(f"action: {action}", file=sys.stderr)
                message = Message(agent_name=player_name, content=action, turn=self._current_turn,
                                  visible_to=[self.player_names[i] for i in self._identity_mapping["werewolf"]],
                                  importance=1 if self._judge_is_alive(player_name) else 0)
                self.message_pool.append_message(message)
                if self._judge_is_alive(player_name):
                    vote = self._text2vote(action, player_name)
                    print(f"vote result: {vote}", file=sys.stderr)
                    if vote in self.player_names or vote == "pass":
                        self._players_votes[vote] += 1
                self._next_player_idx += 1
                terminal = False

                print(f"_next_player_idx: {self._next_player_idx}", file=sys.stderr)
                print(f"number of wolves: {len(self._identity_mapping['werewolf'])}", file=sys.stderr)
                if self._next_player_idx == self.werewolf_numbers:
                    self._moderator_speak(f"You guard, {self.get_next_player()}, please open your eyes! "
                                          f"Now tell me who you protect tonight? "
                                          f"You only choose one from the following living options please: [{', '.join(self._alive_list)}]. ",
                                          # f"For example: I choose to protect Player...",
                                          visible_to=[self.player_names[i] for i in self._identity_mapping["guard"]],
                                          importance=1 if self._judge_is_alive(self.get_next_player()) else 0)
                    # self.print()
                else:
                    self._moderator_speak(f"The next werewolf, you {self.get_next_player()}, please vote and tell your teammates that which of the players should be killed tonight. "
                                          f"You only choose one from the following living options please: [{', '.join(self._alive_list)}]. ",
                                          # f"For example: I choose Player...",
                                          visible_to=[self.player_names[i] for i in self._identity_mapping["werewolf"]],
                                          importance=1 if self._judge_is_alive(self.get_next_player()) else 0)
            elif self._next_player_idx == self.werewolf_numbers:
                print(f"action: {action}", file=sys.stderr)
                message = Message(agent_name=player_name, content=action, turn=self._current_turn,
                                  visible_to=[self.player_names[i] for i in self._identity_mapping["guard"]],
                                  importance=1 if self._judge_is_alive(player_name) else 0)
                self.message_pool.append_message(message)
                if self._judge_is_alive(player_name):
                    vote = self._text2vote(action, player_name=player_name)
                    if "myself" in action:
                        vote = player_name
                    print(f"vote result: {vote}", file=sys.stderr)
                    if vote == '':
                        vote = "pass"
                    if self._guard_to == vote:
                        self._guard_to = "pass"
                    else:
                        self._guard_to = vote
                else:
                    self._guard_to = "pass"
                print(f"guard to: {self._guard_to}", file=sys.stderr)
                self._next_player_idx += 1
                print(f"_next_player_idx: {self._next_player_idx}", file=sys.stderr)
                terminal = False

                # 修复狼人投票统计逻辑
                print(f"players votes: {str(self._players_votes)}", file=sys.stderr)
                
                # 排除"pass"选项，只考虑实际玩家
                player_votes = {k: v for k, v in self._players_votes.items() if k != "pass"}
                
                if player_votes:
                    werewolf_kill = max(player_votes, key=player_votes.get)
                    print(f"werewolf_kill: {werewolf_kill}", file=sys.stderr)
                    
                    # 检查是否有平票
                    max_votes = player_votes[werewolf_kill]
                    tied_players = [name for name, votes in player_votes.items() if votes == max_votes]
                    
                    if len(tied_players) > 1:
                        # 如果有平票，随机选择一个
                        import random
                        werewolf_kill = random.choice(tied_players)
                        print(f"Tied vote, randomly selected: {werewolf_kill}", file=sys.stderr)
                    
                    # 检查是否被守卫保护
                    print(f"_guard_to: {self._guard_to}", file=sys.stderr)
                    if werewolf_kill == self._guard_to:
                        werewolf_kill = "pass"
                        print(f"Target protected by guard, werewolf_kill set to pass", file=sys.stderr)
                else:
                    werewolf_kill = "pass"
                    print(f"No valid votes, werewolf_kill set to pass", file=sys.stderr)
                if werewolf_kill != "pass":
                    self._print_infos()
                    self._killed_list.append(werewolf_kill)
                    self._print_infos()
                self._players_votes = {name: 0 for name in self.player_names}
                self._players_votes["pass"] = 0

                print(f"killed list: {self._killed_list}", file=sys.stderr)
                # 女巫阶段：优先告知被攻击的玩家，然后根据是否有解药决定流程
                if len(self._killed_list) > 0:
                    attacked_player = self._killed_list[0]
                    if self._witch_antidote:
                        # 有解药：询问是否救人
                        self._moderator_speak(f"You witch, {self.get_next_player()}, please open your eyes! "
                                              f"{attacked_player} was attacked by werewolves and will die tonight (unless saved by you or protected by the guard). "
                                              f"You have a bottle of antidote. Do you want to save {attacked_player}? "
                                              f"CRITICAL: You MUST respond with ONLY 'Yes' or 'No'. Do NOT add any explanation or other text. "
                                              f"Must choose only one from the following options: [Yes, No]",
                                              visible_to=[self.player_names[i] for i in self._identity_mapping["witch"]],
                                              importance=6 if self._judge_is_alive(self.get_next_player()) else 0)
                        self._number_of_rounds = 0  # 女巫解药阶段
                    else:
                        # 没有解药：告知被攻击的玩家，然后询问毒药
                        print(f"alive list: {', '.join(self._alive_list)}", file=sys.stderr)
                        self._moderator_speak(f"You witch, {self.get_next_player()}, please open your eyes! "
                                              f"{attacked_player} was attacked by werewolves and will die tonight (unless protected by the guard). "
                                              f"You have no antidote left (already used). "
                                              f"IMPORTANT: {attacked_player} is already being attacked by werewolves and will die. "
                                              f"You have a bottle of poison, who are you going to kill tonight? "
                                              f"CRITICAL: You MUST respond with EXACTLY 'I choose Player X' or 'I choose pass'. "
                                              f"Do NOT add your player name prefix (like 'Player X:'). Do NOT add any explanation or other text. "
                                              f"Just state your choice clearly. Valid options: [{', '.join(self._alive_list)}]. "
                                              f"WARNING: Do NOT poison {attacked_player} as they are already being attacked by werewolves and will die. "
                                              f"Example CORRECT: 'I choose Player 5' (if you suspect Player 5 is a werewolf). "
                                              f"Example WRONG: 'I choose {attacked_player}' (they are already being attacked and will die).",
                                              visible_to=[self.player_names[i] for i in self._identity_mapping["witch"]],
                                              importance=6 if self._judge_is_alive(self.get_next_player()) else 0)
                        self._number_of_rounds = 1  # 女巫毒药阶段
                else:
                    # 没有被攻击的玩家：直接询问毒药
                    print(f"alive list: {', '.join(self._alive_list)}", file=sys.stderr)
                    self._moderator_speak(f"You witch, {self.get_next_player()}, please open your eyes! "
                                          f"You have a bottle of poison, who are you going to kill tonight? "
                                          f"CRITICAL: You MUST respond with EXACTLY 'I choose Player X' or 'I choose pass'. "
                                          f"Do NOT add your player name prefix (like 'Player X:'). Do NOT add any explanation or other text. "
                                          f"Just state your choice clearly. Valid options: [{', '.join(self._alive_list)}]. "
                                          f"Example CORRECT: 'I choose Player 3'. Example WRONG: 'Player 5: I choose Player 3'.",
                                          visible_to=[self.player_names[i] for i in self._identity_mapping["witch"]],
                                          importance=6 if self._judge_is_alive(self.get_next_player()) else 0)
                    self._number_of_rounds = 1  # 女巫毒药阶段
            elif self._next_player_idx == self.werewolf_numbers + 1:
                print(f"action: {action}", file=sys.stderr)
                message = Message(agent_name=player_name, content=action, turn=self._current_turn,
                                  visible_to=[self.player_names[i] for i in self._identity_mapping["witch"]],
                                  importance=1 if self._judge_is_alive(player_name) else 0)
                self.message_pool.append_message(message)
                
                if self._number_of_rounds == 0:
                    # 女巫解药阶段：决定是否救被狼人杀死的玩家
                    if self._judge_is_alive(player_name):
                        print(f"Witch antidote action (raw): {action}", file=sys.stderr)
                        # 清理action，移除可能的前缀（如"[QWEN][Player X]"等）
                        action_clean = re.sub(r'^\[.*?\]\s*', '', action)  # 移除[QWEN][Player X]这样的前缀
                        action_clean = re.sub(r'^Player\s+\d+:\s*', '', action_clean, flags=re.IGNORECASE)  # 移除Player X:前缀
                        action_clean = action_clean.strip()
                        print(f"Witch antidote action (cleaned): {action_clean}", file=sys.stderr)
                        # 改进女巫救人响应解析：更准确地识别Yes/No，优先提取独立的Yes/No
                        action_lower = action_clean.lower().strip()
                        
                        # 方法1: 提取独立的Yes/No单词（最高优先级）
                        # 使用正则表达式匹配独立的yes/no单词
                        yes_no_pattern = r'\b(yes|no|y|n)\b'
                        yes_no_matches = re.findall(yes_no_pattern, action_lower)
                        
                        # 如果找到独立的yes/no，优先使用
                        if yes_no_matches:
                            first_match = yes_no_matches[0]
                            if first_match in ['yes', 'y']:
                                is_yes = True
                                is_no = False
                                print(f"Found explicit Yes/No: '{first_match}', decision: Save", file=sys.stderr)
                            elif first_match in ['no', 'n']:
                                is_yes = False
                                is_no = True
                                print(f"Found explicit Yes/No: '{first_match}', decision: Don't save", file=sys.stderr)
                            else:
                                is_yes = False
                                is_no = False
                        else:
                            # 方法2: 检查是否以Yes/No开头
                            if action_lower.startswith("yes") or action_lower.startswith("y "):
                                is_yes = True
                                is_no = False
                            elif action_lower.startswith("no") or action_lower.startswith("n "):
                                is_yes = False
                                is_no = True
                            else:
                                # 方法3: 检查前几个词中是否包含Yes/No
                                words = action_lower.split()[:5]  # 检查前5个词
                                if "yes" in words or "y" in words:
                                    is_yes = True
                                    is_no = False
                                elif "no" in words or "n" in words:
                                    is_yes = False
                                    is_no = True
                                else:
                                    # 方法4: 检查是否包含明确的救人/不救人的表达
                                    save_patterns = [
                                        "will use" in action_lower and "antidote" in action_lower,
                                        "choose to use" in action_lower and "antidote" in action_lower,
                                        "use" in action_lower and "antidote" in action_lower and "save" in action_lower,
                                        "save" in action_lower and "antidote" in action_lower,
                                        "i will save" in action_lower or "i'll save" in action_lower,
                                        "decide to save" in action_lower
                                    ]
                                    not_save_patterns = [
                                        "will not" in action_lower and "save" in action_lower,
                                        "choose not" in action_lower and "save" in action_lower,
                                        "do not" in action_lower and "save" in action_lower,
                                        "won't save" in action_lower,
                                        "will not use" in action_lower and "antidote" in action_lower,
                                        "decide not to save" in action_lower
                                    ]
                                    
                                    is_yes = any(save_patterns) and not any(not_save_patterns)
                                    is_no = any(not_save_patterns) or (not any(save_patterns) and ("pass" in action_lower or "silent" in action_lower or "wait" in action_lower))
                        
                        if is_yes and not is_no:
                            if self._witch_antidote:
                                self._witch_antidote_to = self._killed_list[0]
                                self._witch_antidote = False
                                self._killed_list = []  # 清空死亡列表，因为被救了
                                print(f"Witch saved {self._witch_antidote_to}, antidote used", file=sys.stderr)
                                print(f"_witch_antidote_to: {self._witch_antidote_to}", file=sys.stderr)
                                print(f"_witch_antidote: {self._witch_antidote}", file=sys.stderr)
                                print(f"_killed_list: {self._killed_list}", file=sys.stderr)
                            else:
                                self._moderator_speak("Failed, your antidote has run out!",
                                                      visible_to=[self.player_names[i] for i in self._identity_mapping["witch"]],
                                                      importance=3)
                                self._witch_antidote_to = None
                        elif is_no:
                            # 女巫选择不救人，保持死亡列表
                            print(f"Witch chose not to save {self._killed_list[0]}", file=sys.stderr)
                        else:
                            # 无法明确识别，默认视为No（不救人）
                            print(f"Witch response unclear: '{action}', treating as No", file=sys.stderr)
                            print(f"Witch chose not to save {self._killed_list[0]}", file=sys.stderr)

                    self._print_infos()
                    # 延迟死亡执行：如果女巫没有救人，将死亡玩家加入夜晚死亡列表，等到白天再执行死亡
                    if len(self._killed_list) > 0:
                        # 不立即执行死亡，而是加入夜晚死亡列表，等到白天再执行
                        if self._killed_list[0] not in self._night_kill_list:
                            self._night_kill_list.append(self._killed_list[0])
                        print(f"Added to night_kill_list: {self._killed_list[0]} (will die at daybreak)", file=sys.stderr)
                        self._killed_list = []
                    self._print_infos()
                    
                    # 进入女巫毒药阶段
                    self._moderator_speak(f"You {self.get_next_player()} have a bottle of poison. Who are you going to kill tonight? "
                                          f"CRITICAL: You MUST respond with EXACTLY 'I choose Player X' or 'I choose pass'. "
                                          f"Do NOT add your player name prefix (like 'Player X:'). Do NOT add any explanation or other text. "
                                          f"Just state your choice clearly. Valid options: [{', '.join(self._alive_list)}]. "
                                          f"Example CORRECT: 'I choose Player 3'. Example WRONG: 'Player 5: I choose Player 3'.",
                                          visible_to=[self.player_names[i] for i in self._identity_mapping["witch"]],
                                          importance=6 if self._judge_is_alive(self.get_next_player()) else 0)
                    self._number_of_rounds = 1
                elif self._number_of_rounds == 1:
                    # 女巫毒药阶段：决定是否毒死某个玩家
                    if self._judge_is_alive(player_name):
                        vote = self._text2vote(action, player_name=player_name)
                        print(f"vote result: {vote}", file=sys.stderr)
                        if self._witch_poison:
                            self._witch_poison_to = vote if self._judge_is_alive(vote) else "pass"
                            self._witch_poison = False if self._judge_is_alive(vote) else True
                            print(f"_witch_poison_to: {self._witch_poison_to}", file=sys.stderr)
                            print(f"_witch_poison: {self._witch_poison}", file=sys.stderr)
                        else:
                            self._moderator_speak("Failed, your poison has run out!",
                                                  visible_to=[self.player_names[i] for i in self._identity_mapping["witch"]],
                                                  importance=3)
                            self._witch_poison_to = None
                        # 如果女巫选择毒死某个玩家，将其加入死亡列表
                        if self._witch_poison_to is not None and self._witch_poison_to != "pass":
                            if self._witch_poison_to not in self._killed_list:
                                self._killed_list.append(self._witch_poison_to)
                                print(f"Witch poisoned {self._witch_poison_to}", file=sys.stderr)
                    self._print_infos()
                    # 延迟死亡执行：包括狼人杀人和女巫毒药，等到白天再执行死亡
                    if len(self._killed_list) > 0:
                        # 不立即执行死亡，而是加入夜晚死亡列表，等到白天再执行
                        if self._killed_list[0] not in self._night_kill_list:
                            self._night_kill_list.append(self._killed_list[0])
                        print(f"Added to night_kill_list: {self._killed_list[0]} (will die at daybreak)", file=sys.stderr)
                        self._killed_list = []
                    self._print_infos()
                    self._next_player_idx += 1
                    self._number_of_rounds = 0

                    print(f"_next_player_idx: {self._next_player_idx}", file=sys.stderr)
                    self._moderator_speak(f"You seer, {self.get_next_player()}, please open your eyes! "
                                          f"Who are you going to verify its identity tonight? "
                                          f"Choose only one from the following living options: [{', '.join(self._alive_list)}]. ",
                                          # f"For example: I choose to verify Player...",
                                          visible_to=[self.player_names[i] for i in self._identity_mapping["seer"]],
                                          importance=1 if self._judge_is_alive(self.get_next_player()) else 0)

                terminal = False
            elif self._next_player_idx == self.werewolf_numbers + 2:
                print(f"action: {action}", file=sys.stderr)
                message = Message(agent_name=player_name, content=action, turn=self._current_turn,
                                  visible_to=[self.player_names[i] for i in self._identity_mapping["seer"]],
                                  importance=1 if self._judge_is_alive(player_name) else 0)
                self.message_pool.append_message(message)
                if self._judge_is_alive(player_name):
                    vote = self._text2vote(action, player_name)
                    print(f"vote result: {vote}", file=sys.stderr)
                    idx = 0 if vote == "pass" else self.player_names.index(vote)
                    print(f"idx: {idx}", file=sys.stderr)
                    # 检查被查验的玩家是否在夜晚死亡列表中
                    target_player = self.player_names[idx] if idx > 0 else None
                    if target_player and target_player in self._night_kill_list:
                        # 如果被查验的玩家在夜晚死亡，预言家不应该在夜晚就知道结果，等到白天
                        print(f"Player {target_player} will die tonight, seer result will be revealed at daybreak", file=sys.stderr)
                        # 暂时不告诉预言家结果，等到白天再告诉
                    else:
                        # 正常情况：告诉预言家查验结果
                        if self._characters[idx] == "werewolf":
                            self._moderator_speak(f"{self.player_names[idx]} is a werewolf!",
                                                  visible_to=[self.player_names[i] for i in self._identity_mapping["seer"]],
                                                  importance=5)
                        else:
                            self._moderator_speak(f"{self.player_names[idx]} is not a werewolf!",
                                                  visible_to=[self.player_names[i] for i in self._identity_mapping["seer"]],
                                                  importance=5)

                self._next_player_idx += 1
                print(f"_next_player_idx: {self._next_player_idx}", file=sys.stderr)
                terminal = False

                # 白天开始：执行夜晚的死亡
                if len(self._night_kill_list) > 0:
                    # 执行死亡：将夜晚死亡列表中的玩家标记为死亡
                    self._kill_by_name(self._night_kill_list)
                    print(f"Executed deaths at daybreak: {self._night_kill_list}", file=sys.stderr)
                
                self._moderator_speak("The sun rose. Everyone woke up except those who had been killed.")
                print(f"killed list: {self._night_kill_list}", file=sys.stderr)
                if len(self._night_kill_list) != 0:
                    self._moderator_speak(f"{','.join(self._night_kill_list)} died last night!", importance=6)
                    # 删除：不应该宣布死亡玩家的身份
                else:
                    self._moderator_speak("It was a peaceful night and no one died!", importance=4)

                self._players_votes = {name: 0 for name in self.player_names}
                self._players_votes["pass"] = 0
                self._guard_to = None
                self._witch_antidote_to = None
                self._witch_poison_to = None
                self._killed_list = []
                self._night_kill_list = []
                if self._check_game_over():
                    return TimeStep(observation=self.get_observation(), reward=rewards, terminal=True)
                self._next_player_idx = self._current_first_alive = self._get_next_alive(-1)
                print(f"_next_player_idx: {self._next_player_idx}", file=sys.stderr)
                self._current_phase = "daytime"
                
                self._moderator_speak(f"Now freely talk about roles of other players with each other based on your observation and "
                                      f"reflection with few sentences. Decide whether to reveal your identity based on your reflection. "
                                      f"IMPORTANT: This is the DISCUSSION phase. Do NOT vote now. Only discuss and analyze. You will vote later when the moderator asks you to vote. "
                                      f"CRITICAL: Do NOT explicitly reveal your role in your speech. Use normal, natural language. "
                                      f"Never say phrases like 'I am a werewolf', 'as a werewolf', 'being a werewolf', 'I need to blend in', or any phrase that reveals your role or that you are hiding something. "
                                      f"Speak naturally as a player trying to find werewolves or defend yourself. "
                                      f"REMINDER: If you have information that can help eliminate players from suspicion (e.g., you successfully protected someone, used an antidote to save someone, or verified someone as good), sharing this information helps the village narrow down suspects and make better voting decisions. "
                                      f"The first {self.get_next_player()}, you please.\n"
                                      # f"For example: I observed that... I think that..."
                                      )

                self._number_of_nights += 1
                
            else:
                raise ValueError(f"Unknown player_idx: {self._next_player_idx}")

            timestep = TimeStep(observation=self.get_observation(), reward=rewards, terminal=terminal)
        else:
            raise ValueError(f"Unknown phase: {self._current_phase}")

        # Check if the player signals the end of the conversation
        if self.is_terminal():
            timestep.terminal = True

        return timestep
