"""
完整的共情分析和情绪建模模块
包含：共情提取、情绪建模、策略选择
"""
import json
import logging
from typing import Dict, List, Tuple, Optional
import re

logger = logging.getLogger("EmpathyModule")


class EmotionModel:
    """PAD情绪模型 (Pleasure-Arousal-Dominance)"""
    
    def __init__(self):
        self.pleasure = 0.5  # 愉悦度 (0-1)
        self.arousal = 0.5   # 激活度 (0-1)
        self.dominance = 0.5 # 支配度 (0-1)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "pleasure": self.pleasure,
            "arousal": self.arousal,
            "dominance": self.dominance
        }
    
    @staticmethod
    def from_dict(data: Dict[str, float]) -> 'EmotionModel':
        model = EmotionModel()
        model.pleasure = data.get("pleasure", 0.5)
        model.arousal = data.get("arousal", 0.5)
        model.dominance = data.get("dominance", 0.5)
        return model


class PlayerEmpathyProfile:
    """单个玩家的共情档案"""
    
    def __init__(self, player_name: str):
        self.player_name = player_name
        self.emotion = EmotionModel()
        self.stance_to_me = 0.0  # 对我的态度 (-1到1)
        self.trust_score = 0.5   # 信任度 (0-1)
        self.speech_acts = []    # 最近的发言行为
        self.politeness = 0.5    # 礼貌度
        self.consistency = 0.7   # 一致性
        self.influence = 0.6     # 影响力
        self.role_probability = {  # 角色概率
            "werewolf": 0.2,
            "villager": 0.3,
            "seer": 0.15,
            "witch": 0.15,
            "guard": 0.2
        }
        self.susceptibility = {  # 易受影响的方面
            "logic": 0.5,
            "emotion": 0.5,
            "authority": 0.5,
            "consensus": 0.5,
            "reciprocity": 0.5,
            "scarcity": 0.5,
            "commitment": 0.5
        }
    
    def to_dict(self) -> Dict:
        return {
            "player_name": self.player_name,
            "emotion": self.emotion.to_dict(),
            "stance_to_me": self.stance_to_me,
            "trust_score": self.trust_score,
            "speech_acts": self.speech_acts[-5:],  # 最近5个发言
            "politeness": self.politeness,
            "consistency": self.consistency,
            "influence": self.influence,
            "role_probability": self.role_probability,
            "susceptibility": self.susceptibility
        }


class EmpathyAnalyzer:
    """共情分析器 - 通过LLM分析其他玩家"""
    
    def __init__(self, backend=None):
        self.backend = backend
    
    def analyze_all_players(
        self,
        agent_name: str,
        alive_players: List[str],
        game_history: str,
        current_observation: str,
        my_role: str
    ) -> Dict[str, PlayerEmpathyProfile]:
        """
        分析所有其他玩家的共情档案
        
        Args:
            agent_name: 当前玩家名称
            alive_players: 存活玩家列表
            game_history: 游戏历史
            current_observation: 当前观察
            my_role: 我的角色
        
        Returns:
            Dict[玩家名 -> 共情档案]
        """
        empathy_profiles = {}
        
        for player in alive_players:
            if player == agent_name or player == "pass":
                continue
            
            profile = self._analyze_single_player(
                agent_name=agent_name,
                target_player=player,
                game_history=game_history,
                current_observation=current_observation,
                my_role=my_role,
                alive_players=alive_players
            )
            empathy_profiles[player] = profile
        
        return empathy_profiles
    
    def _analyze_single_player(
        self,
        agent_name: str,
        target_player: str,
        game_history: str,
        current_observation: str,
        my_role: str,
        alive_players: List[str]
    ) -> PlayerEmpathyProfile:
        """分析单个玩家"""
        profile = PlayerEmpathyProfile(target_player)
        
        if not self.backend:
            logger.warning(f"No backend for empathy analysis of {target_player}")
            return profile
        
        # 构建分析提示
        prompt = self._build_analysis_prompt(
            agent_name=agent_name,
            target_player=target_player,
            game_history=game_history,
            current_observation=current_observation,
            my_role=my_role,
            alive_players=alive_players
        )
        
        try:
            # 调用LLM进行分析
            response = self.backend.query(None, prompt)
            
            # 解析响应
            analysis = self._parse_empathy_response(response)
            
            # 更新档案
            if analysis:
                profile.emotion.pleasure = analysis.get("emotion", {}).get("pleasure", 0.5)
                profile.emotion.arousal = analysis.get("emotion", {}).get("arousal", 0.5)
                profile.emotion.dominance = analysis.get("emotion", {}).get("dominance", 0.5)
                profile.stance_to_me = analysis.get("stance_to_me", 0.0)
                profile.trust_score = analysis.get("trust_score", 0.5)
                profile.role_probability = analysis.get("role_probability", profile.role_probability)
                profile.susceptibility = analysis.get("susceptibility", profile.susceptibility)
        
        except Exception as e:
            logger.warning(f"Empathy analysis failed for {target_player}: {e}")
        
        return profile
    
    def _build_analysis_prompt(
        self,
        agent_name: str,
        target_player: str,
        game_history: str,
        current_observation: str,
        my_role: str,
        alive_players: List[str]
    ) -> str:
        """构建共情分析提示"""
        return f"""你是一个心理分析专家。分析玩家 {target_player} 的情绪状态和意图。

当前玩家: {agent_name}
我的角色: {my_role}
存活玩家: {', '.join(alive_players)}

游戏历史:
{game_history}

当前观察:
{current_observation}

请分析 {target_player} 的：
1. 情绪状态 (PAD模型):
   - pleasure (愉悦度, 0-1): 
   - arousal (激活度, 0-1):
   - dominance (支配度, 0-1):

2. 对我的态度 (stance_to_me, -1到1):

3. 信任度 (trust_score, 0-1):

4. 角色概率 (role_probability):
   - werewolf:
   - villager:
   - seer:
   - witch:
   - guard:

5. 易受影响的方面 (susceptibility, 0-1):
   - logic (逻辑):
   - emotion (情感):
   - authority (权威):
   - consensus (共识):
   - reciprocity (互惠):
   - scarcity (稀缺):
   - commitment (承诺):

返回JSON格式的分析结果。"""
    
    def _parse_empathy_response(self, response: str) -> Optional[Dict]:
        """解析LLM的共情分析响应"""
        try:
            # 尝试直接解析JSON
            if isinstance(response, dict):
                return response
            
            # 从字符串中提取JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            
            return None
        except Exception as e:
            logger.warning(f"Failed to parse empathy response: {e}")
            return None


class StrategySelector:
    """基于共情的策略选择器"""
    
    @staticmethod
    def select_strategy(
        my_role: str,
        empathy_profiles: Dict[str, PlayerEmpathyProfile],
        game_state: Dict
    ) -> Tuple[str, str, str]:
        """
        基于共情档案选择策略
        
        Returns:
            (action_type, target_player, speech_style)
        """
        if not empathy_profiles:
            return ("pass", "pass", "neutral")
        
        # 根据角色选择策略
        if my_role == "werewolf":
            return StrategySelector._select_werewolf_strategy(empathy_profiles)
        elif my_role == "villager":
            return StrategySelector._select_villager_strategy(empathy_profiles)
        elif my_role == "seer":
            return StrategySelector._select_seer_strategy(empathy_profiles)
        elif my_role == "witch":
            return StrategySelector._select_witch_strategy(empathy_profiles)
        elif my_role == "guard":
            return StrategySelector._select_guard_strategy(empathy_profiles)
        else:
            return ("pass", "pass", "neutral")
    
    @staticmethod
    def _select_werewolf_strategy(
        empathy_profiles: Dict[str, PlayerEmpathyProfile]
    ) -> Tuple[str, str, str]:
        """狼人策略：找最容易迷惑的玩家"""
        # 找信任度最高的玩家（最容易迷惑）
        best_target = max(
            empathy_profiles.items(),
            key=lambda x: x[1].trust_score
        )
        
        # 根据目标的易受影响程度选择发言风格
        susceptibility = best_target[1].susceptibility
        if susceptibility["emotion"] > 0.6:
            speech_style = "emotional"
        elif susceptibility["logic"] > 0.6:
            speech_style = "logical"
        else:
            speech_style = "neutral"
        
        return ("vote", best_target[0], speech_style)
    
    @staticmethod
    def _select_villager_strategy(
        empathy_profiles: Dict[str, PlayerEmpathyProfile]
    ) -> Tuple[str, str, str]:
        """村民策略：投票给最可能是狼人的玩家"""
        # 找狼人概率最高的玩家
        best_target = max(
            empathy_profiles.items(),
            key=lambda x: x[1].role_probability.get("werewolf", 0.2)
        )
        
        return ("vote", best_target[0], "logical")
    
    @staticmethod
    def _select_seer_strategy(
        empathy_profiles: Dict[str, PlayerEmpathyProfile]
    ) -> Tuple[str, str, str]:
        """预言家策略：检查最可疑的玩家"""
        # 找最可疑的玩家（狼人概率高 + 态度不友好）
        best_target = max(
            empathy_profiles.items(),
            key=lambda x: x[1].role_probability.get("werewolf", 0.2) - x[1].stance_to_me
        )
        
        return ("check", best_target[0], "neutral")
    
    @staticmethod
    def _select_witch_strategy(
        empathy_profiles: Dict[str, PlayerEmpathyProfile]
    ) -> Tuple[str, str, str]:
        """女巫策略：保护最有价值的玩家"""
        # 找最有影响力且态度友好的玩家
        best_target = max(
            empathy_profiles.items(),
            key=lambda x: x[1].influence * (1 + x[1].stance_to_me)
        )
        
        return ("protect", best_target[0], "supportive")
    
    @staticmethod
    def _select_guard_strategy(
        empathy_profiles: Dict[str, PlayerEmpathyProfile]
    ) -> Tuple[str, str, str]:
        """卫兵策略：保护最可能被攻击的玩家"""
        # 找最有影响力的玩家
        best_target = max(
            empathy_profiles.items(),
            key=lambda x: x[1].influence
        )
        
        return ("protect", best_target[0], "protective")


class SpeechGenerator:
    """基于共情和情绪的发言生成器"""
    
    @staticmethod
    def generate_speech(
        agent_name: str,
        my_role: str,
        action: Tuple[str, str, str],
        empathy_profiles: Dict[str, PlayerEmpathyProfile],
        backend=None
    ) -> str:
        """
        生成策略性发言
        
        Args:
            agent_name: 玩家名称
            my_role: 玩家角色
            action: (action_type, target_player, speech_style)
            empathy_profiles: 共情档案
            backend: LLM后端
        
        Returns:
            生成的发言
        """
        action_type, target_player, speech_style = action
        
        if not backend:
            return SpeechGenerator._generate_default_speech(action_type, target_player)
        
        # 构建发言提示
        prompt = SpeechGenerator._build_speech_prompt(
            agent_name=agent_name,
            my_role=my_role,
            action=action,
            empathy_profiles=empathy_profiles
        )
        
        try:
            speech = backend.query(None, prompt)
            return speech if isinstance(speech, str) else str(speech)
        except Exception as e:
            logger.warning(f"Speech generation failed: {e}")
            return SpeechGenerator._generate_default_speech(action_type, target_player)
    
    @staticmethod
    def _build_speech_prompt(
        agent_name: str,
        my_role: str,
        action: Tuple[str, str, str],
        empathy_profiles: Dict[str, PlayerEmpathyProfile]
    ) -> str:
        """构建发言生成提示"""
        action_type, target_player, speech_style = action
        
        # 构建目标玩家的共情信息
        target_info = ""
        if target_player in empathy_profiles:
            profile = empathy_profiles[target_player]
            target_info = f"""
目标玩家 {target_player} 的特征：
- 情绪: 愉悦度={profile.emotion.pleasure:.2f}, 激活度={profile.emotion.arousal:.2f}, 支配度={profile.emotion.dominance:.2f}
- 对我的态度: {profile.stance_to_me:.2f}
- 信任度: {profile.trust_score:.2f}
- 易受影响: 逻辑={profile.susceptibility['logic']:.2f}, 情感={profile.susceptibility['emotion']:.2f}
"""
        
        return f"""你是一个策略性的狼人杀游戏玩家。生成一句发言。

玩家: {agent_name}
角色: {my_role}
行动: {action_type} {target_player}
发言风格: {speech_style}
{target_info}

生成一句自然、有说服力的发言（1-2句话）。"""
    
    @staticmethod
    def _generate_default_speech(action_type: str, target_player: str) -> str:
        """生成默认发言"""
        if action_type == "vote":
            return f"I vote to eliminate {target_player}."
        elif action_type == "check":
            return f"I want to check {target_player}."
        elif action_type == "protect":
            return f"I will protect {target_player}."
        else:
            return "I choose to pass."
