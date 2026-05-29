#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Werewolf Evaluator - 增强版狼人杀评估器

提供详细的游戏分析和评估指标，包括：
- 投票准确性
- 发言分析
- 角色表现
- 游戏流程记录
"""

import json
import os
import re
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class GameEvent:
    """游戏事件"""
    timestamp: float
    phase: str  # "day" or "night"
    day_night: int
    event_type: str
    player: str
    target: Optional[str]
    content: str
    details: Dict[str, Any] = field(default_factory=dict)


class EnhancedWerewolfEvaluator:
    """增强版狼人杀评估器"""
    
    def __init__(self, system_name: str = "full_system", model_name: str = "default", seed: int = 0):
        self.system_name = system_name
        self.model_name = model_name
        self.seed = seed
        
        # 游戏基本信息
        self.players: List[str] = []
        self.roles: Dict[str, str] = {}
        self.winner: Optional[str] = None
        self.current_turn: int = 0
        self.current_phase: str = "day"
        self.alive_players: List[str] = []
        
        # 事件记录
        self.events: List[GameEvent] = []
        
        # 统计数据
        self.speeches: List[Dict[str, Any]] = []
        self.votes: List[Dict[str, Any]] = []
        self.lynches: List[Dict[str, Any]] = []
        self.seer_peeks: List[Dict[str, Any]] = []
        self.witch_actions: List[Dict[str, Any]] = []
        
        # 玩家统计
        self.player_stats: Dict[str, Dict[str, Any]] = {}
        
    def start_game(self, players: List[str], roles: Dict[str, str]):
        """开始游戏"""
        self.players = list(players)
        self.roles = dict(roles)
        self.alive_players = list(players)
        self.current_turn = 1
        self.current_phase = "day"
        
        # 初始化玩家统计
        for player in players:
            self.player_stats[player] = {
                "role": roles.get(player, "unknown"),
                "survival_rounds": 0,
                "total_actions": 0,
                "speeches": 0,
                "votes": 0,
                "night_actions": 0,
                "is_alive": True
            }
        
        # 记录游戏开始事件
        self._add_event(
            event_type="game_start",
            player="System",
            target=None,
            content="Game started",
            details={
                "players": self.players,
                "roles": self.roles
            }
        )
    
    def on_speech(self, speaker: str, content: str, current_phase: str, current_turn: int):
        """记录发言"""
        self.current_phase = current_phase
        self.current_turn = current_turn
        
        # 更新玩家统计
        if speaker in self.player_stats:
            self.player_stats[speaker]["speeches"] += 1
            self.player_stats[speaker]["total_actions"] += 1
        
        # 分析发言内容
        analysis = self._analyze_speech(content)
        
        # 记录发言
        speech_data = {
            "speaker": speaker,
            "content": content,
            "phase": current_phase,
            "day_night": current_turn,
            "analysis": analysis
        }
        self.speeches.append(speech_data)
        
        # 记录事件
        self._add_event(
            event_type="speech",
            player=speaker,
            target=None,
            content=content,
            details={
                "phase": current_phase,
                "day_night": current_turn,
                "analysis": analysis
            }
        )
    
    def on_vote(self, speaker: str, target: str, current_phase: str, current_turn: int):
        """记录投票"""
        self.current_phase = current_phase
        self.current_turn = current_turn
        
        # 更新玩家统计
        if speaker in self.player_stats:
            self.player_stats[speaker]["votes"] += 1
            self.player_stats[speaker]["total_actions"] += 1
        
        # 判断投票是否正确（目标是否为狼人）
        target_role = self.roles.get(target, "")
        is_correct = (target_role == "werewolf" or target_role == "wolf")
        
        # 记录投票
        vote_data = {
            "voter": speaker,
            "target": target,
            "phase": current_phase,
            "day_night": current_turn,
            "is_correct": is_correct,
            "target_role": target_role
        }
        self.votes.append(vote_data)
        
        # 记录事件
        self._add_event(
            event_type="vote",
            player=speaker,
            target=target,
            content=f"Voted for {target}",
            details={
                "phase": current_phase,
                "day_night": current_turn,
                "is_correct": is_correct,
                "target_role": target_role
            }
        )
    
    def on_lynch(self, lynched: str, current_turn: int):
        """记录处决"""
        self.current_turn = current_turn
        
        # 更新存活列表
        if lynched in self.alive_players:
            self.alive_players.remove(lynched)
        
        # 更新玩家统计
        if lynched in self.player_stats:
            self.player_stats[lynched]["is_alive"] = False
            self.player_stats[lynched]["survival_rounds"] = current_turn
        
        # 获取被处决玩家的角色
        lynched_role = self.roles.get(lynched, "")
        is_wolf = (lynched_role == "werewolf" or lynched_role == "wolf")
        
        # 记录处决
        lynch_data = {
            "day": current_turn,
            "lynched": lynched,
            "is_wolf": is_wolf,
            "lynched_role": lynched_role
        }
        self.lynches.append(lynch_data)
        
        # 记录事件
        self._add_event(
            event_type="lynch",
            player="System",
            target=lynched,
            content=f"Lynched {lynched}",
            details={
                "day": current_turn,
                "is_wolf": is_wolf,
                "lynched_role": lynched_role
            }
        )
    
    def on_night_begin(self, current_turn: int):
        """记录夜晚开始"""
        self.current_turn = current_turn
        self.current_phase = "night"
        
        # 更新所有存活玩家的生存轮数
        for player in self.alive_players:
            if player in self.player_stats:
                self.player_stats[player]["survival_rounds"] = max(
                    self.player_stats[player]["survival_rounds"], 
                    current_turn
                )
        
        # 记录事件
        self._add_event(
            event_type="night_begin",
            player="System",
            target=None,
            content=f"Night {current_turn} begins",
            details={
                "night": current_turn,
                "alive_players": list(self.alive_players)
            }
        )
    
    def on_seer_peek(self, seer: str, target: str, is_wolf: bool):
        """记录预言家查验"""
        # 更新玩家统计
        if seer in self.player_stats:
            self.player_stats[seer]["night_actions"] += 1
            self.player_stats[seer]["total_actions"] += 1
        
        target_role = self.roles.get(target, "")
        
        # 记录查验
        peek_data = {
            "night": self.current_turn,
            "seer": seer,
            "target": target,
            "is_wolf": is_wolf,
            "target_role": target_role
        }
        self.seer_peeks.append(peek_data)
        
        # 记录事件
        self._add_event(
            event_type="seer_peek",
            player=seer,
            target=target,
            content=f"Peeked {target}",
            details={
                "is_wolf": is_wolf,
                "night": self.current_turn,
                "target_role": target_role
            }
        )
    
    def on_witch_action(self, witch: str, action_type: str, target: str, success: bool):
        """记录女巫行动"""
        # 更新玩家统计
        if witch in self.player_stats:
            self.player_stats[witch]["night_actions"] += 1
            self.player_stats[witch]["total_actions"] += 1
        
        target_role = self.roles.get(target, "")
        
        # 记录行动
        action_data = {
            "night": self.current_turn,
            "witch": witch,
            "action_type": action_type,
            "target": target,
            "success": success,
            "target_role": target_role
        }
        self.witch_actions.append(action_data)
        
        # 记录事件
        self._add_event(
            event_type="witch_action",
            player=witch,
            target=target,
            content=f"{action_type} {target}",
            details={
                "action_type": action_type,
                "success": success,
                "night": self.current_turn,
                "target_role": target_role
            }
        )
    
    def on_game_end(self, winner: str):
        """记录游戏结束"""
        self.winner = winner
        
        # 更新所有存活玩家的最终生存轮数
        for player in self.alive_players:
            if player in self.player_stats:
                self.player_stats[player]["survival_rounds"] = self.current_turn
        
        # 记录事件
        self._add_event(
            event_type="game_end",
            player="System",
            target=None,
            content=f"Game ended, {winner} won",
            details={
                "winner": winner,
                "final_alive": list(self.alive_players),
                "total_rounds": self.current_turn
            }
        )
    
    def _analyze_speech(self, content: str) -> Dict[str, Any]:
        """分析发言内容"""
        analysis = {
            "length": len(content),
            "mentions_players": [],
            "mentions_roles": [],
            "sentiment": "neutral",
            "strategy_indicators": []
        }
        
        # 提取提到的玩家
        player_pattern = r"Player\s+\d+"
        mentioned_players = re.findall(player_pattern, content, re.IGNORECASE)
        analysis["mentions_players"] = list(set(mentioned_players))
        
        # 提取提到的角色
        role_keywords = ["werewolf", "wolf", "villager", "seer", "witch", "guard"]
        mentioned_roles = []
        content_lower = content.lower()
        for role in role_keywords:
            if role in content_lower:
                mentioned_roles.append(role)
        analysis["mentions_roles"] = mentioned_roles
        
        # 简单的策略识别
        strategy_keywords = {
            "accusation": ["accuse", "suspicious", "guilty", "liar", "lying"],
            "defense": ["innocent", "defend", "not guilty", "trust me"],
            "information": ["verify", "check", "found", "discovered", "reveal"],
            "cooperation": ["trust", "work together", "team", "ally"],
            "distraction": ["maybe", "perhaps", "could be", "might"]
        }
        
        content_lower = content.lower()
        for strategy, keywords in strategy_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                analysis["strategy_indicators"].append(strategy)
        
        return analysis
    
    def _add_event(self, event_type: str, player: str, target: Optional[str], 
                   content: str, details: Dict[str, Any]):
        """添加游戏事件"""
        event = GameEvent(
            timestamp=time.time(),
            phase=self.current_phase,
            day_night=self.current_turn,
            event_type=event_type,
            player=player,
            target=target,
            content=content,
            details=details
        )
        self.events.append(event)
    
    def _compute_vote_accuracy(self) -> Dict[str, float]:
        """计算投票准确性"""
        vote_accuracy = {}
        for player in self.players:
            player_votes = [v for v in self.votes if v["voter"] == player]
            if player_votes:
                correct_votes = sum(1 for v in player_votes if v["is_correct"])
                vote_accuracy[player] = correct_votes / len(player_votes)
            else:
                vote_accuracy[player] = 0.0
        return vote_accuracy
    
    def _compute_lynch_accuracy(self) -> List[Dict[str, Any]]:
        """计算处决准确性"""
        return [
            {
                "day": l["day"],
                "lynched": l["lynched"],
                "is_wolf": l["is_wolf"],
                "lynched_role": l["lynched_role"]
            }
            for l in self.lynches
        ]
    
    def _compute_speech_analysis(self) -> Dict[str, Any]:
        """计算发言分析"""
        speeches_by_player = {}
        speeches_by_day = {}
        strategy_distribution = {}
        total_length = 0
        
        for speech in self.speeches:
            speaker = speech["speaker"]
            day = str(speech["day_night"])
            analysis = speech.get("analysis", {})
            
            # 按玩家统计
            speeches_by_player[speaker] = speeches_by_player.get(speaker, 0) + 1
            
            # 按天数统计
            speeches_by_day[day] = speeches_by_day.get(day, 0) + 1
            
            # 策略分布
            strategies = analysis.get("strategy_indicators", [])
            for strategy in strategies:
                strategy_distribution[strategy] = strategy_distribution.get(strategy, 0) + 1
            
            # 总长度
            total_length += analysis.get("length", 0)
        
        # 确保所有玩家都在统计中
        for player in self.players:
            if player not in speeches_by_player:
                speeches_by_player[player] = 0
        
        return {
            "total_speeches": len(self.speeches),
            "speeches_by_player": speeches_by_player,
            "speeches_by_day": speeches_by_day,
            "strategy_distribution": strategy_distribution,
            "average_length": total_length / len(self.speeches) if self.speeches else 0.0
        }
    
    def export_detailed_results(self) -> str:
        """导出详细结果到JSON文件"""
        # 计算指标
        vote_accuracy = self._compute_vote_accuracy()
        lynch_accuracy = self._compute_lynch_accuracy()
        speech_analysis = self._compute_speech_analysis()
        
        # 构建结果数据
        result = {
            "meta": {
                "game_id": f"{self.system_name}_{self.model_name}_{int(time.time())}",
                "system": self.system_name,
                "model": self.model_name,
                "seed": self.seed,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
            },
            "game_info": {
                "total_rounds": self.current_turn,
                "winner": self.winner,
                "roles": self.roles,
                "initial_players": self.players,
                "final_alive_players": list(self.alive_players)
            },
            "metrics": {
                "vote_accuracy": vote_accuracy,
                "lynch_accuracy": lynch_accuracy,
                "speech_analysis": speech_analysis,
                "role_performance": self.player_stats,
                "errors": []
            },
            "game_flow": [asdict(event) for event in self.events],
            "events": [asdict(event) for event in self.events]
        }
        
        # 保存到文件
        os.makedirs("experiment_results/detailed", exist_ok=True)
        filename = f"{result['meta']['game_id']}_detailed.json"
        filepath = os.path.join("experiment_results/detailed", filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return filepath

