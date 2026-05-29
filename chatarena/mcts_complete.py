"""
完整的MCTS + 共情流程实现
包含：节点选择、模拟、反向传播、共情集成
"""
import logging
import math
import random
from typing import Dict, List, Tuple, Optional
from .empathy_module import (
    EmpathyAnalyzer,
    StrategySelector,
    SpeechGenerator,
    PlayerEmpathyProfile
)

logger = logging.getLogger("MCTS_Complete")


class MCTSNode:
    """MCTS节点"""
    
    def __init__(
        self,
        state: Dict,
        player: str,
        parent: Optional['MCTSNode'] = None,
        action: Optional[Tuple] = None
    ):
        self.state = state
        self.player = player
        self.parent = parent
        self.action = action or ("pass", player, "neutral")
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.value = 0.0
        self.empathy_data = {}  # 该节点的共情数据
    
    def ucb_score(self, c: float = 1.414) -> float:
        """计算UCB分数"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        exploration = 0
        if self.parent and self.parent.visits > 0:
            exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        
        return exploitation + exploration
    
    def best_child(self, c: float = 1.414) -> Optional['MCTSNode']:
        """选择最佳子节点"""
        if not self.children:
            return None
        return max(self.children, key=lambda child: child.ucb_score(c))
    
    def add_child(self, action: Tuple, state: Dict) -> 'MCTSNode':
        """添加子节点"""
        child = MCTSNode(state, self.player, parent=self, action=action)
        self.children.append(child)
        return child
    
    def update(self, value: float):
        """更新节点统计"""
        self.visits += 1
        self.value += value


class MCTSWithEmpathy:
    """集成共情分析的MCTS"""
    
    def __init__(
        self,
        backend=None,
        n_iterations: int = 20,
        use_empathy: bool = True
    ):
        self.backend = backend
        self.n_iterations = n_iterations
        self.use_empathy = use_empathy
        self.empathy_analyzer = EmpathyAnalyzer(backend)
    
    def search(
        self,
        root_state: Dict,
        agent_name: str,
        my_role: str,
        alive_players: List[str],
        game_history: str,
        current_observation: str
    ) -> Tuple[MCTSNode, Tuple[str, str, str]]:
        """
        执行MCTS搜索
        
        Returns:
            (best_node, action)
        """
        # 1. 共情分析
        empathy_profiles = {}
        if self.use_empathy:
            logger.info(f"[MCTS] 开始共情分析: agent={agent_name}, role={my_role}")
            empathy_profiles = self.empathy_analyzer.analyze_all_players(
                agent_name=agent_name,
                alive_players=alive_players,
                game_history=game_history,
                current_observation=current_observation,
                my_role=my_role
            )
            logger.info(f"[MCTS] 共情分析完成: 获取 {len(empathy_profiles)} 个玩家档案")
        
        # 2. 策略选择
        logger.info(f"[MCTS] 开始策略选择")
        action = StrategySelector.select_strategy(
            my_role=my_role,
            empathy_profiles=empathy_profiles,
            game_state=root_state
        )
        logger.info(f"[MCTS] 策略选择完成: action={action}")
        
        # 3. MCTS搜索
        root = MCTSNode(root_state, agent_name)
        root.empathy_data = empathy_profiles
        
        logger.info(f"[MCTS] 开始MCTS搜索: iterations={self.n_iterations}")
        for i in range(self.n_iterations):
            node = self._tree_policy(root, empathy_profiles)
            reward = self._default_policy(node, empathy_profiles)
            self._backup(node, reward)
            
            if (i + 1) % 5 == 0:
                logger.debug(f"[MCTS] 完成 {i + 1}/{self.n_iterations} 次迭代")
        
        # 4. 选择最佳节点
        best_node = root.best_child(c=0)  # 利用阶段，不探索
        if not best_node:
            best_node = root
        
        logger.info(f"[MCTS] MCTS搜索完成: best_action={best_node.action}, visits={best_node.visits}, value={best_node.value:.2f}")
        
        return best_node, action
    
    def _tree_policy(
        self,
        node: MCTSNode,
        empathy_profiles: Dict[str, PlayerEmpathyProfile]
    ) -> MCTSNode:
        """树策略：选择或扩展节点"""
        while not self._is_terminal(node):
            if not node.children:
                # 扩展节点
                return self._expand(node, empathy_profiles)
            else:
                # 选择最佳子节点
                child = node.best_child()
                if child is None:
                    return node
                node = child
        
        return node
    
    def _expand(
        self,
        node: MCTSNode,
        empathy_profiles: Dict[str, PlayerEmpathyProfile]
    ) -> MCTSNode:
        """扩展节点：生成可能的动作"""
        # 生成可能的动作
        possible_actions = self._generate_actions(node, empathy_profiles)
        
        if not possible_actions:
            return node
        
        # 随机选择一个未探索的动作
        action = random.choice(possible_actions)
        
        # 创建新的子节点
        new_state = self._simulate_action(node.state, action, node.player)
        child = node.add_child(action, new_state)
        child.empathy_data = empathy_profiles
        
        return child
    
    def _default_policy(
        self,
        node: MCTSNode,
        empathy_profiles: Dict[str, PlayerEmpathyProfile]
    ) -> float:
        """默认策略：随机模拟到游戏结束"""
        state = node.state.copy()
        depth = 0
        max_depth = 10
        
        while not self._is_terminal(state) and depth < max_depth:
            # 随机选择动作
            actions = self._generate_actions_from_state(state)
            if not actions:
                break
            
            action = random.choice(actions)
            state = self._simulate_action(state, action, node.player)
            depth += 1
        
        # 评估最终状态
        reward = self._evaluate_state(state, node.player, empathy_profiles)
        return reward
    
    def _backup(self, node: MCTSNode, reward: float):
        """反向传播：更新节点统计"""
        while node is not None:
            node.update(reward)
            node = node.parent
    
    def _generate_actions(
        self,
        node: MCTSNode,
        empathy_profiles: Dict[str, PlayerEmpathyProfile]
    ) -> List[Tuple]:
        """生成可能的动作"""
        actions = []
        
        # 基于共情档案生成动作
        for player_name, profile in empathy_profiles.items():
            # 投票动作
            actions.append(("vote", player_name, "logical"))
            actions.append(("vote", player_name, "emotional"))
            
            # 其他动作
            actions.append(("check", player_name, "neutral"))
            actions.append(("protect", player_name, "supportive"))
        
        # 通过动作
        actions.append(("pass", "pass", "neutral"))
        
        return actions
    
    def _generate_actions_from_state(self, state: Dict) -> List[Tuple]:
        """从状态生成动作"""
        alive = state.get("alive_players", [])
        actions = []
        
        for player in alive:
            if player != "pass":
                actions.append(("vote", player, "neutral"))
        
        actions.append(("pass", "pass", "neutral"))
        return actions
    
    def _simulate_action(
        self,
        state: Dict,
        action: Tuple,
        player: str
    ) -> Dict:
        """模拟动作的结果"""
        new_state = state.copy()
        action_type, target, style = action
        
        # 简单的状态转移
        if action_type == "vote" and target in new_state.get("alive_players", []):
            # 模拟投票的影响
            new_state["last_vote"] = (player, target)
        
        return new_state
    
    def _evaluate_state(
        self,
        state: Dict,
        player: str,
        empathy_profiles: Dict[str, PlayerEmpathyProfile]
    ) -> float:
        """评估状态的价值"""
        reward = 0.5  # 基础奖励
        
        # 如果玩家还活着，增加奖励
        if player in state.get("alive_players", []):
            reward += 0.3
        
        # 基于共情档案调整奖励
        for profile in empathy_profiles.values():
            # 如果有友好的玩家活着，增加奖励
            if profile.stance_to_me > 0.5:
                reward += 0.1
        
        return min(reward, 1.0)
    
    def _is_terminal(self, node: MCTSNode) -> bool:
        """检查节点是否是终端状态"""
        # 简化：假设所有节点都不是终端
        return False


def run_mcts_with_empathy(
    agent_name: str,
    my_role: str,
    alive_players: List[str],
    game_history: str,
    current_observation: str,
    backend=None,
    n_iterations: int = 20
) -> Tuple[str, str]:
    """
    运行完整的MCTS + 共情流程
    
    Returns:
        (selected_action, generated_speech)
    """
    logger.info(f"[MCTS_Complete] 开始MCTS+共情流程: agent={agent_name}, role={my_role}")
    
    # 构建游戏状态
    game_state = {
        "agent": agent_name,
        "role": my_role,
        "alive_players": alive_players,
        "history": game_history
    }
    
    # 创建MCTS实例
    mcts = MCTSWithEmpathy(backend=backend, n_iterations=n_iterations, use_empathy=True)
    
    # 执行搜索
    best_node, action = mcts.search(
        root_state=game_state,
        agent_name=agent_name,
        my_role=my_role,
        alive_players=alive_players,
        game_history=game_history,
        current_observation=current_observation
    )
    
    # 获取共情档案用于发言生成
    empathy_profiles = best_node.empathy_data
    
    # 生成发言
    logger.info(f"[MCTS_Complete] 开始生成发言: action={action}")
    speech = SpeechGenerator.generate_speech(
        agent_name=agent_name,
        my_role=my_role,
        action=action,
        empathy_profiles=empathy_profiles,
        backend=backend
    )
    logger.info(f"[MCTS_Complete] 发言生成完成: {speech}")
    
    return action, speech
