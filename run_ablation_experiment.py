"""
消融实验脚本：对比同一LLM在有/无MCTS+共情模块下的表现

实验设计：
- 使用同一个LLM模型
- 按阵营分配：一个阵营使用完整版本（MCTS+共情，is_mcts=1）
- 另一个阵营使用基础版本（只有基础提示词，is_mcts=0）
- 记录并对比两个阵营的性能指标
"""

import argparse
import os
import json
import sys
from chatarena.arena import Arena, TooManyInvalidActions
from chatarena.config import ArenaConfig
from chatarena.message import Message
from experiments.eval_logger import EvaluationSession
from enhanced_evaluator import EnhancedWerewolfEvaluator
import re
import random

# 放在程序最开始的位置
log_file = open('ablation_debug_output.txt', 'w', encoding='utf-8')
sys.stderr = log_file
sys.stdout = log_file

import builtins
import inspect

old_print = print

def debug_print(*args, **kwargs):
    frame = inspect.currentframe().f_back
    info = inspect.getframeinfo(frame)
    prefix = f"[{info.filename}:{info.lineno} {info.function}]"
    old_print(prefix, *args, **kwargs)

builtins.print = debug_print

VOTE_PATTERN = re.compile(r"vote\s+(Player\s*\d+)", re.IGNORECASE)

def parse_vote_from_text(text):
    if not text:
        return None
    m = VOTE_PATTERN.search(text)
    return m.group(1) if m else None


class AblationArgs:
    """消融实验参数类，按阵营设置不同的is_mcts值"""
    def __init__(self, base_args, role_mcts_config, roles_dict):
        """
        base_args: 基础参数对象
        role_mcts_config: dict，格式为 {role: is_mcts_value}
        例如: {"werewolf": 1, "villager": 0, "seer": 0, ...}
        roles_dict: dict，格式为 {player_name: role}
        例如: {"Player 1": "werewolf", "Player 2": "villager", ...}
        """
        # 复制所有基础属性
        for attr in dir(base_args):
            if not attr.startswith('_'):
                setattr(self, attr, getattr(base_args, attr))
        
        self.role_mcts_config = role_mcts_config
        self.roles_dict = roles_dict
        self._original_is_mcts = getattr(base_args, 'is_mcts', 1)
    
    def get_is_mcts_for_player(self, player_name):
        """为特定玩家获取is_mcts值（根据其角色）"""
        role = self.roles_dict.get(player_name, "")
        # 检查是否是狼人阵营
        if role in ("werewolf", "wolf"):
            return self.role_mcts_config.get("werewolf", self._original_is_mcts)
        else:
            # 好人阵营
            return self.role_mcts_config.get("good", self._original_is_mcts)
    
    @property
    def is_mcts(self):
        """默认返回1，但实际使用时应该通过get_is_mcts_for_player获取"""
        return self._original_is_mcts


def create_ablation_args_by_faction(base_args, roles_dict, mcts_faction="werewolf"):
    """
    创建消融实验参数（按阵营分配）
    
    Args:
        base_args: 基础参数对象
        roles_dict: dict，格式为 {player_name: role}
        mcts_faction: 使用MCTS+共情的阵营，"werewolf" 或 "good"
    """
    role_mcts_config = {}
    if mcts_faction == "werewolf":
        role_mcts_config["werewolf"] = 1  # 狼人阵营使用MCTS+共情
        role_mcts_config["good"] = 0  # 好人阵营不使用
    else:
        role_mcts_config["werewolf"] = 0  # 狼人阵营不使用
        role_mcts_config["good"] = 1  # 好人阵营使用MCTS+共情
    
    return AblationArgs(base_args, role_mcts_config, roles_dict)


def main():
    parser = argparse.ArgumentParser(description="消融实验：对比MCTS+共情模块的效果")
    
    # 基础参数（从run_werewolf.py继承）
    parser.add_argument("--current-game-number", type=int, default=0, help="当前游戏编号")
    parser.add_argument("--message-window", type=int, default=10, help="用于推理的最新消息数量")
    parser.add_argument("--answer-topk", type=int, default=5, help="检索答案数量")
    parser.add_argument("--exps-retrieval-threshold", type=float, default=0.6, help="经验检索阈值")
    parser.add_argument("--similar-exps-threshold", type=float, default=0.1, help="相似经验阈值")
    parser.add_argument("--max-tokens", type=int, default=100, help="每次生成的最大token数")
    parser.add_argument("--retri-question-number", type=int, default=5, help="问题历史数量")
    parser.add_argument("--temperature", type=float, default=0.2, help="生成温度")
    parser.add_argument("--use-api-server", type=int, default=0, help="使用自建API服务器")
    
    # 消融实验特定参数
    parser.add_argument("--model", type=str, default="qwen2-72b-instruct", 
                       help="用于消融实验的LLM模型（所有玩家使用同一模型）")
    parser.add_argument("--mcts-faction", type=str, default="werewolf",
                       choices=["werewolf", "good"],
                       help="使用MCTS+共情的阵营：werewolf=狼人阵营, good=好人阵营（默认：werewolf）")
    
    parser.add_argument("--save-exps-incremental", action="store_true", default=False, help="增量保存经验")
    parser.add_argument("--use-crossgame-exps", action="store_true", default=False, help="使用跨游戏经验")
    parser.add_argument("--use-crossgame-ques", action="store_true", default=False, help="使用跨游戏问题")
    parser.add_argument("--human-in-combat", action="store_true", default=False, help="Player 1使用人类")
    
    parser.add_argument("--environment-config", type=str, default="./examples/werewolf.json", 
                       help="环境配置文件")
    parser.add_argument("--role-config", type=str, default="./config/1.json", help="角色配置文件")
    parser.add_argument("--exps-path-to", type=str, help="经验保存路径")
    parser.add_argument("--ques-path-to", type=str, help="问题保存路径")
    parser.add_argument("--logs-path-to", type=str, default="./logs", help="日志保存路径")
    parser.add_argument("--load-exps-from", type=str, help="经验加载路径")
    parser.add_argument("--load-ques-from", type=str, help="问题加载路径")
    parser.add_argument("--who-use-exps", nargs='+', help="使用经验的角色列表")
    parser.add_argument("--who-use-ques", nargs='+", help="使用问题的角色列表")
    
    args = parser.parse_args()
    
    # 创建基础args对象（用于兼容原有代码）
    base_args = args
    
    # 读取环境配置
    with open(args.environment_config, "r") as f:
        config = json.load(f)
    
    # 创建目录
    if args.exps_path_to:
        os.makedirs(args.exps_path_to, exist_ok=True)
    if args.ques_path_to:
        os.makedirs(args.ques_path_to, exist_ok=True)
    os.makedirs(args.logs_path_to, exist_ok=True)
    
    # 保存实验配置（在获取角色信息后）
    ablation_log_path = os.path.join(args.logs_path_to, f"ablation_{args.current_game_number}.md")
    with open(ablation_log_path, "w", encoding="utf-8") as f:
        f.write(f"# 消融实验配置\n\n")
        f.write(f"## 实验参数\n")
        for arg in vars(args):
            f.write(f"- {arg}: {getattr(args, arg)}\n")
        f.write(f"\n## 角色分配\n")
        for player, role in roles.items():
            f.write(f"- {player}: {role}\n")
        f.write(f"\n## MCTS分配（按阵营）\n")
        f.write(f"- 使用MCTS+共情的阵营: {mcts_faction}\n")
        f.write(f"- MCTS+共情玩家: {', '.join(mcts_players)}\n")
        f.write(f"- 基础版本玩家: {', '.join(no_mcts_players)}\n")
        f.write(f"- 使用模型: {args.model}\n")
        f.write(f"\n")
    
    # 配置moderator
    moderator_config = {
        "role_desc": "",
        "global_prompt": config["global_prompt"],
        "terminal_condition": "",
        "backend": {
            "backend_type": "openai-chat",
            "temperature": 0.2,
            "max_tokens": 100
        }
    }
    
    env_config = {
        "env_type": "werewolf",
        "parallel": False,
        "moderator": moderator_config,
        "moderator_visibility": "all",
        "moderator_period": "turn"
    }
    
    # 配置玩家（所有玩家使用相同的模型和配置）
    player_configs = []
    for i in range(len(config["players"])):
        player_name = f"Player {i + 1}"
        role_desc = config["players"][i]["role_desc"]
        backend_type = config["players"][i]["backend"]["backend_type"]
        temperature = config["players"][i]["backend"]["temperature"]
        max_tokens = config["players"][i]["backend"]["max_tokens"]
        
        player_config = {
            "name": player_name,
            "role_desc": role_desc,
            "global_prompt": config["global_prompt"],
            "backend": {
                "backend_type": backend_type,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        }
        player_configs.append(player_config)
    
    # 创建Arena（先创建临时args，稍后会替换为ablation_args）
    temp_args = base_args
    arena = Arena.from_config(ArenaConfig(players=player_configs, environment=env_config), temp_args)
    
    # 获取玩家和角色信息
    players = arena.environment.player_names
    roles = dict(zip(players, arena.environment._characters))
    
    # 根据角色分配MCTS
    mcts_faction = args.mcts_faction
    werewolf_players = [p for p, r in roles.items() if r in ("werewolf", "wolf")]
    good_players = [p for p, r in roles.items() if r not in ("werewolf", "wolf")]
    
    if mcts_faction == "werewolf":
        mcts_players = werewolf_players
        no_mcts_players = good_players
        print(f"[ABLATION] 狼人阵营使用MCTS+共情: {mcts_players}", file=sys.stderr)
        print(f"[ABLATION] 好人阵营使用基础版本: {no_mcts_players}", file=sys.stderr)
    else:
        mcts_players = good_players
        no_mcts_players = werewolf_players
        print(f"[ABLATION] 好人阵营使用MCTS+共情: {mcts_players}", file=sys.stderr)
        print(f"[ABLATION] 狼人阵营使用基础版本: {no_mcts_players}", file=sys.stderr)
    
    # 创建消融实验参数（按阵营分配）
    ablation_args = create_ablation_args_by_faction(base_args, roles, mcts_faction)
    
    # 配置所有玩家使用相同的模型
    try:
        from chatarena.backends import OpenAIChat, QwenChat, GeminiChat
        
        model_name = args.model
        temperature = getattr(ablation_args, "temperature", 0.2)
        max_tokens = getattr(ablation_args, "max_tokens", 100)
        
        def _create_backend(model_name: str):
            lower = (model_name or "").lower()
            BackendCls = OpenAIChat
            if lower.startswith("gemini"):
                BackendCls = GeminiChat
            elif lower.startswith("qwen"):
                BackendCls = QwenChat
            return BackendCls(ablation_args, temperature=temperature, max_tokens=max_tokens, model=model_name)
        
        # 所有玩家使用相同的模型
        for p in arena.players:
            p.backend = _create_backend(model_name)
            # 为每个玩家设置is_mcts值（通过修改backend的args）
            # 注意：需要在backend.query中动态获取
            if not hasattr(p.backend, '_ablation_args'):
                p.backend._ablation_args = ablation_args
                p.backend._player_name = p.name
        
        print(f"[ABLATION] 所有玩家使用模型: {model_name}", file=sys.stderr)
        print(f"[ABLATION] MCTS玩家: {mcts_players}", file=sys.stderr)
        print(f"[ABLATION] 无MCTS玩家: {no_mcts_players}", file=sys.stderr)
        
    except Exception as e:
        print(f"[ABLATION ERROR] 配置backend失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    
    # 为每个backend添加动态is_mcts支持
    # 方法：在backend实例上存储玩家名称和ablation_args，然后在query方法中动态获取is_mcts值
    for p in arena.players:
        # 在backend实例上存储玩家信息
        p.backend._ablation_player_name = p.name
        p.backend._ablation_args = ablation_args
        
        # 包装query方法
        original_query = p.backend.query
        
        def make_wrapped_query(backend_instance, player_name):
            def wrapped_query(arg, *args, **kwargs):
                # 获取该玩家的is_mcts值
                is_mcts_value = ablation_args.get_is_mcts_for_player(player_name)
                
                # 创建临时args对象，设置正确的is_mcts值
                class TempArgs:
                    def __init__(self, base_args, is_mcts_val):
                        # 复制所有属性
                        for attr in dir(base_args):
                            if not attr.startswith('_'):
                                try:
                                    setattr(self, attr, getattr(base_args, attr))
                                except:
                                    pass
                        self.is_mcts = is_mcts_val
                
                temp_args = TempArgs(arg if arg is not None else ablation_args, is_mcts_value)
                
                # 调用原始query方法，传入修改后的args
                return original_query(temp_args, *args, **kwargs)
            
            return wrapped_query
        
        p.backend.query = make_wrapped_query(p.backend, p.name)
    
    # 初始化评估器
    es = EvaluationSession(
        system_name="ablation_experiment",
        model_name=args.model,
        seed=args.current_game_number,
        game_id=str(arena.uuid),
    )
    es.on_game_start(players=players, roles=roles)
    
    enhanced_evaluator = EnhancedWerewolfEvaluator(
        system_name="ablation_experiment",
        model_name=args.model,
        seed=args.current_game_number
    )
    enhanced_evaluator.start_game(players, roles)
    
    # 记录玩家分组信息到评估器
    enhanced_evaluator.mcts_players = mcts_players
    enhanced_evaluator.no_mcts_players = no_mcts_players
    
    # 用于检测每日处刑
    prev_alive = list(arena.environment.alive_list)
    
    # 游戏主循环
    while True:
        try:
            timestep = arena.step(ablation_args)
            
            # 每步测评打点
            current_phase = arena.environment.current_phase
            current_turn = arena.environment.current_turn
            alive = list(arena.environment.alive_list)
            
            msg_pool = arena.environment.message_pool
            
            # 检查是否处于投票阶段
            is_voting_phase = False
            if hasattr(arena.environment, "_number_of_rounds"):
                is_voting_phase = (arena.environment._number_of_rounds == 1)
            else:
                if hasattr(msg_pool, "_messages") and msg_pool._messages:
                    for msg in reversed(msg_pool._messages[-5:]):
                        msg_content = msg.content
                        if isinstance(msg_content, list):
                            if msg_content and isinstance(msg_content[0], str):
                                msg_content = msg_content[0]
                            else:
                                continue
                        if msg.agent_name == "Moderator" and isinstance(msg_content, str):
                            if "are asked to choose" in msg_content.lower() or "vote to kill" in msg_content.lower():
                                is_voting_phase = True
                                break
            
            if hasattr(msg_pool, "_messages") and msg_pool._messages:
                last_msg = msg_pool._messages[-1]
                speaker = getattr(last_msg, "agent_name", None)
                content_raw = getattr(last_msg, "content", "")
                
                if isinstance(content_raw, list):
                    content = str(content_raw[0]) if content_raw and isinstance(content_raw[0], str) else ""
                else:
                    content = str(content_raw) if content_raw else ""
                
                if speaker and speaker != "Moderator" and content:
                    es.on_speech(speaker=speaker, target_player="Unknown", speech_style="ambiguous")
                    enhanced_evaluator.on_speech(speaker, content, current_phase, current_turn)
                    
                    # 记录发言者的MCTS状态
                    is_mcts_player = speaker in mcts_players
                    if hasattr(enhanced_evaluator, 'record_speech_mcts_status'):
                        enhanced_evaluator.record_speech_mcts_status(speaker, is_mcts_player)
                    
                    if current_phase == "day" and is_voting_phase:
                        strict_vote_pattern = re.compile(r"(?:I\s+)?(?:vote\s+to\s+kill|vote\s+for)\s+(Player\s+\d+)", re.IGNORECASE)
                        match = strict_vote_pattern.search(content)
                        if match:
                            vt = match.group(1)
                            es.on_vote(voter=speaker, target=vt)
                            enhanced_evaluator.on_vote(speaker, vt, current_phase, current_turn)
                            
                            # 记录投票者的MCTS状态
                            is_mcts_player = speaker in mcts_players
                            if hasattr(enhanced_evaluator, 'record_vote_mcts_status'):
                                enhanced_evaluator.record_vote_mcts_status(speaker, is_mcts_player)
            
            # 检测白天处刑
            if current_phase == "day":
                removed = [p for p in prev_alive if p not in alive]
                if removed:
                    es.on_day_end_lynch(lynched=removed[0])
                    enhanced_evaluator.on_lynch(removed[0], current_turn)
            
            # 检测夜晚行动
            if current_phase == "night":
                enhanced_evaluator.on_night_begin(current_turn)
                
                # 检测预言家查验
                if hasattr(msg_pool, "_messages") and msg_pool._messages:
                    for msg in msg_pool._messages[-5:]:
                        msg_content = msg.content
                        if isinstance(msg_content, list):
                            if msg_content and isinstance(msg_content[0], str):
                                msg_content = msg_content[0]
                            else:
                                continue
                        elif not isinstance(msg_content, str):
                            continue
                        
                        msg_content_lower = msg_content.lower()
                        if "seer" in msg_content_lower and "verify" in msg_content_lower:
                            seer_match = re.search(r'Player \d+', msg_content)
                            verify_pos = msg_content.find("verify")
                            if verify_pos >= 0:
                                target_match = re.search(r'Player \d+', msg_content[verify_pos:])
                            else:
                                target_match = None
                            if seer_match and target_match:
                                seer = seer_match.group(0)
                                target = target_match.group(0)
                                is_wolf = roles.get(target) == "werewolf"
                                enhanced_evaluator.on_seer_peek(seer, target, is_wolf)
                                break
                
                # 检测女巫行动
                if hasattr(msg_pool, "_messages") and msg_pool._messages:
                    for msg in msg_pool._messages[-5:]:
                        msg_content = msg.content
                        if isinstance(msg_content, list):
                            if msg_content and isinstance(msg_content[0], str):
                                msg_content = msg_content[0]
                            else:
                                continue
                        elif not isinstance(msg_content, str):
                            continue
                        
                        msg_content_lower = msg_content.lower()
                        if "witch" in msg_content_lower:
                            if "antidote" in msg_content_lower or "save" in msg_content_lower:
                                witch_match = re.search(r'Player \d+', msg_content)
                                save_pos = msg_content.find("save")
                                if save_pos >= 0:
                                    target_match = re.search(r'Player \d+', msg_content[save_pos:])
                                else:
                                    target_match = None
                                if witch_match and target_match:
                                    witch = witch_match.group(0)
                                    target = target_match.group(0)
                                    enhanced_evaluator.on_witch_action(witch, "antidote", target, True)
                            elif "poison" in msg_content_lower or "kill" in msg_content_lower:
                                witch_match = re.search(r'Player \d+', msg_content)
                                kill_pos = msg_content.find("kill")
                                if kill_pos >= 0:
                                    target_match = re.search(r'Player \d+', msg_content[kill_pos:])
                                else:
                                    target_match = None
                                if witch_match and target_match:
                                    witch = witch_match.group(0)
                                    target = target_match.group(0)
                                    enhanced_evaluator.on_witch_action(witch, "poison", target, True)
            
            prev_alive = alive
            
        except TooManyInvalidActions as e:
            timestep = arena.current_timestep
            timestep.observation.append(
                Message("System", "Too many invalid actions. Game over.", turn=-1, visible_to="all"))
            timestep.terminal = True
        
        if timestep.terminal == True:
            # 终局胜者判定
            wolves_alive = [p for p in arena.environment.alive_list if roles.get(p) in ("wolf", "werewolf")]
            non_wolves_alive = [p for p in arena.environment.alive_list if roles.get(p) not in ("wolf", "werewolf")]
            if len(wolves_alive) == 0:
                winner = "villagers"
            elif len(wolves_alive) >= len(non_wolves_alive):
                winner = "werewolves"
            else:
                winner = "villagers"
            
            es.on_game_end(winner=winner)
            enhanced_evaluator.on_game_end(winner)
            
            # 导出结果
            out_path = es.finalize_and_export(out_dir=os.path.join("experiment_results", "raw"), fmt="json")
            print(f"[ABLATION] 原始评测数据已保存：{out_path}", file=sys.stderr)
            
            enhanced_path = enhanced_evaluator.export_detailed_results()
            print(f"[ABLATION] 详细评测数据已保存：{enhanced_path}", file=sys.stderr)
            
            # 导出消融实验对比结果
            ablation_results_path = os.path.join("experiment_results", "ablation", 
                                                 f"ablation_{args.current_game_number}.json")
            os.makedirs(os.path.dirname(ablation_results_path), exist_ok=True)
            
            # 统计MCTS vs 无MCTS的表现（按阵营）
            final_alive = list(arena.environment.alive_list)
            mcts_alive = [p for p in final_alive if p in mcts_players]
            no_mcts_alive = [p for p in final_alive if p in no_mcts_players]
            
            # 判断哪个阵营获胜
            mcts_faction_won = False
            if mcts_faction == "werewolf":
                # 如果MCTS阵营是狼人，检查狼人是否获胜
                mcts_faction_won = (winner == "werewolves")
            else:
                # 如果MCTS阵营是好人，检查好人是否获胜
                mcts_faction_won = (winner == "villagers")
            
            ablation_stats = {
                "experiment_id": args.current_game_number,
                "model": args.model,
                "mcts_faction": mcts_faction,
                "winner": winner,
                "mcts_faction_won": mcts_faction_won,
                "roles": roles,
                "werewolf_players": werewolf_players,
                "good_players": good_players,
                "mcts_players": mcts_players,
                "no_mcts_players": no_mcts_players,
                "final_alive": final_alive,
                "mcts_alive": mcts_alive,
                "no_mcts_alive": no_mcts_alive,
                "mcts_survival_rate": len(mcts_alive) / len(mcts_players) if mcts_players else 0.0,
                "no_mcts_survival_rate": len(no_mcts_alive) / len(no_mcts_players) if no_mcts_players else 0.0,
            }
            
            with open(ablation_results_path, "w", encoding="utf-8") as f:
                json.dump(ablation_stats, f, indent=2, ensure_ascii=False)
            
            print(f"[ABLATION] 消融实验结果已保存：{ablation_results_path}", file=sys.stderr)
            
            break
    
    if args.exps_path_to:
        arena.environment.message_pool.save_exps_to(args.save_exps_incremental)
    if args.ques_path_to:
        arena.environment.question_pool.save_ques_to(args.save_exps_incremental)


if __name__ == "__main__":
    main()

# 程序结尾
log_file.close()

