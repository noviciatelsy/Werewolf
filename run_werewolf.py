"""
    Automatic Running Script of Werewolf Game
    Developed by Yuzhuang Xu, Tsinghua University
    v1.0 2023-05-04
    Base Platform: ChatArena(https://github.com/chatarena/chatarena)
"""

import argparse
import os
import json
from chatarena.arena import Arena, TooManyInvalidActions
from chatarena.config import ArenaConfig
from chatarena.message import Message
import sys
from experiments.eval_logger import EvaluationSession
from enhanced_evaluator import EnhancedWerewolfEvaluator
import os, re, json

# ===== 消融实验控制变量（手动设置） =====
# 设置为 True 表示该阵营使用 MCTS+共情模块，False 表示只使用基础提示词
Is_Wolf_MCTS = True   # 狼人阵营是否使用MCTS+共情（True=使用, False=不使用）
Is_Goods_MCTS = True # 好人阵营是否使用MCTS+共情（True=使用, False=不使用）

# 消融实验模式：如果设置了阵营MCTS变量，将自动启用消融实验
# 消融实验要求所有玩家使用同一模型
ABLATION_MODEL = "gpt-4o"  # 消融实验使用的模型（所有玩家使用此模型）
# ==========================================

# 放在程序最开始的位置
log_file = open('debug_output.txt', 'w', encoding='utf-8')
sys.stderr = log_file
sys.stdout = log_file  # 如果你还希望普通 print 也写入同一文件

import builtins
import inspect

old_print = print

def debug_print(*args, **kwargs):
    # 找到调用print的地方
    frame = inspect.currentframe().f_back
    info = inspect.getframeinfo(frame)
    prefix = f"[{info.filename}:{info.lineno} {info.function}]"
    old_print(prefix, *args, **kwargs)

# 替换全局print
builtins.print = debug_print

VOTE_PATTERN = re.compile(r"vote\\s+(Player\\s*\\d+)", re.IGNORECASE)

def parse_vote_from_text(text):
    if not text:
        return None
    m = VOTE_PATTERN.search(text)
    return m.group(1) if m else None

def main():
    parser = argparse.ArgumentParser(description="The command-line parameter of werewolf game.")
    
    parser.add_argument("--current-game-number", type=int, default=0, help="this is the serial number of current game, must a integer")
    parser.add_argument("--message-window", type=int, default=10, help="number of the newest message for driving game reasoning")
    parser.add_argument("--answer-topk", type=int, default=5, help="number of the retrieval answers for choosing")
    parser.add_argument("--exps-retrieval-threshold", type=float, default=0.6, help="experiences whose reflexion similarity larger than it will be recalled")
    parser.add_argument("--similar-exps-threshold", type=float, default=0.1, help="experiences whose similarity difference is less than it will be omited")
    parser.add_argument("--max-tokens", type=int, default=100, help="maximum tokens of each generation")
    parser.add_argument("--retri-question-number", type=int, default=5, help="number of questions from question history")
    parser.add_argument("--temperature", type=float, default=0.2, help="temperature hyper-parameter of generation model")
    parser.add_argument("--use-api-server", type=int, default=0, help="use the self-developed api server for anytime calling")
    parser.add_argument("--is-mcts", type=int, default=1, help="1 to enable full MCTS+empathy pipeline; 0 to disable and use minimal single-step LLM")

    parser.add_argument("--save-exps-incremental", action="store_true", default=False, help="save all experiences defore this piece of game in a file")
    parser.add_argument("--use-crossgame-exps", action="store_true", default=False, help="use the cross-trajectory experiences of different games")
    parser.add_argument("--use-crossgame-ques", action="store_true", default=False, help="use the cross-trajectory questions of different games")
    parser.add_argument("--human-in-combat", action="store_true", default=False, help="enable Player 1 with human")
    
    parser.add_argument("--environment-config", type=str, default="./examples/werewolf.json", help="json file that define the rule and players")
    parser.add_argument("--role-config", type=str, default="./config/1.json", help="json file that define the number of roles")
    parser.add_argument("--exps-path-to", type=str, help="path of saving binary files of experiences")
    parser.add_argument("--ques-path-to", type=str, help="path of saving binary files of questions")
    parser.add_argument("--logs-path-to", type=str, default="./logs", help="path of saving log files of competitive talking")
    parser.add_argument("--load-exps-from", type=str, help="path of experience files for loading")
    parser.add_argument("--load-ques-from", type=str, help="path of question files for loading")
    parser.add_argument("--who-use-exps", nargs='+', help="a list of roles that will use experiences")
    parser.add_argument("--who-use-ques", nargs='+', help="a list of roles that will use question histories")
    parser.add_argument("--wolves-model", type=str, default="gpt-4o", help="Model for werewolves (default: gemini-1.5-flash). Use gemini-1.5-flash or gemini-2.0-flash for Gemini models.")
    parser.add_argument("--goods-model", type=str, default="gpt-4o", help="Model for good players (default: gemini-2.0-flash). Use gemini-1.5-flash or gemini-2.0-flash for Gemini models.")
    
    # 消融实验参数
    parser.add_argument("--ablation-mode", type=str, default=None,
                       choices=["werewolf", "good"],
                       help="消融实验模式：werewolf=狼人阵营使用MCTS+共情, good=好人阵营使用MCTS+共情。设置此参数后，所有玩家使用同一模型（--ablation-model）")
    parser.add_argument("--ablation-model", type=str, default="qwen2-72b-instruct",
                       help="消融实验使用的模型（当--ablation-mode设置时，所有玩家使用此模型）")
    
    args = parser.parse_args()
    
    # 测试关键导入
    print("[STARTUP] Testing critical imports...", file=sys.stderr)
    try:
        from chatarena.arena import Arena, TooManyInvalidActions
        from chatarena.config import ArenaConfig
        from chatarena.message import Message
        print("[STARTUP] ✅ Core modules imported successfully", file=sys.stderr)
    except ImportError as e:
        print(f"[STARTUP] ❌ Failed to import core modules: {e}", file=sys.stderr)
        sys.exit(1)
    
    try:
        from chatarena.MCTS import GameState, MCTS, llm_empathy_extract
        print("[STARTUP] ✅ MCTS modules imported successfully", file=sys.stderr)
    except ImportError as e:
        print(f"[STARTUP] ⚠️  MCTS modules import failed: {e}", file=sys.stderr)
        print("[STARTUP] Game will run in fallback mode without MCTS", file=sys.stderr)
    except Exception as e:
        print(f"[STARTUP] ⚠️  MCTS modules error: {e}", file=sys.stderr)
        print("[STARTUP] Game will run in fallback mode without MCTS", file=sys.stderr)
    
    if args.exps_path_to:
        os.makedirs(args.exps_path_to, exist_ok=True)
    if args.ques_path_to:
        os.makedirs(args.ques_path_to, exist_ok=True)
    os.makedirs(args.logs_path_to, exist_ok=True)
    with open(os.path.join(args.logs_path_to, str(args.current_game_number) + ".md"), "w") as f:
        for arg in vars(args):
            f.write(f"{arg} : {getattr(args, arg)}  " + "\n")
        f.write("\n")

    # 读取配置文件
    print(f"[STARTUP] Loading config from {args.environment_config}", file=sys.stderr)
    
    # 尝试多个可能的路径
    config_paths = [
        args.environment_config,  # 用户指定的路径
        os.path.join(os.path.dirname(__file__), args.environment_config.lstrip('./')),  # 相对于脚本文件的路径
        os.path.join(os.path.dirname(__file__), "examples", "werewolf.json"),  # 相对于脚本文件的examples目录
    ]
    
    config = None
    config_path_used = None
    for config_path in config_paths:
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                config_path_used = config_path
                print(f"[STARTUP] ✅ Config loaded successfully from {config_path_used}", file=sys.stderr)
                break
            except Exception as e:
                print(f"[STARTUP] ⚠️  Failed to load config from {config_path}: {e}", file=sys.stderr)
                continue
    
    if config is None:
        print(f"[STARTUP] ❌ Failed to load config from any of the following paths:", file=sys.stderr)
        for path in config_paths:
            print(f"  - {path}", file=sys.stderr)
        sys.exit(1)
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
    player_configs = []
    for i in range(len(config["players"])):
        player_name = f"Player {i + 1}"
        role_desc, backend_type, temperature, max_tokens = config["players"][i]["role_desc"], \
            config["players"][i]["backend"]["backend_type"], config["players"][i]["backend"]["temperature"], \
            config["players"][i]["backend"]["max_tokens"]
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
        
    print(f"[STARTUP] Creating arena with {len(player_configs)} players", file=sys.stderr)
    try:
        arena = Arena.from_config(ArenaConfig(players=player_configs, environment=env_config), args)
        print("[STARTUP] ✅ Arena created successfully", file=sys.stderr)
    except Exception as e:
        print(f"[STARTUP] ❌ Failed to create arena: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    # ==== 评测会话初始化 ====
    players = arena.environment.player_names
    roles = dict(zip(players, arena.environment._characters))  # 真实角色

    # ==== 消融实验模式：按阵营分配MCTS ====
    # 检查是否通过变量设置了消融实验
    is_ablation_mode_by_var = (Is_Wolf_MCTS != Is_Goods_MCTS)  # 两个阵营MCTS设置不同时启用消融实验
    is_ablation_mode_by_arg = args.ablation_mode is not None
    is_ablation_mode = is_ablation_mode_by_var or is_ablation_mode_by_arg
    
    # 如果两个阵营都使用MCTS，直接设置args.is_mcts = 1
    if Is_Wolf_MCTS and Is_Goods_MCTS:
        print(f"[MCTS MODE] 所有玩家都使用 MCTS+共情模块", file=sys.stderr)
        args.is_mcts = 1
        is_ablation_mode = False  # 不是消融实验，是完整MCTS模式
    elif not Is_Wolf_MCTS and not Is_Goods_MCTS:
        print(f"[BASELINE MODE] 所有玩家都使用基础提示词（不使用MCTS）", file=sys.stderr)
        args.is_mcts = 0
        is_ablation_mode = False  # 不是消融实验，是完整基础模式
    
    if is_ablation_mode:
        # 如果通过变量设置，使用变量值；否则使用命令行参数
        if is_ablation_mode_by_var:
            # 根据变量确定哪个阵营使用MCTS
            if Is_Wolf_MCTS and not Is_Goods_MCTS:
                ablation_faction = "werewolf"
            elif Is_Goods_MCTS and not Is_Wolf_MCTS:
                ablation_faction = "good"
            else:
                # 如果两个都True或都False，使用默认（都True时默认狼人用MCTS）
                ablation_faction = "werewolf" if Is_Wolf_MCTS else "good"
            ablation_model = ABLATION_MODEL
            print(f"[ABLATION MODE] 通过变量设置启用消融实验", file=sys.stderr)
            print(f"[ABLATION MODE] Is_Wolf_MCTS={Is_Wolf_MCTS}, Is_Goods_MCTS={Is_Goods_MCTS}", file=sys.stderr)
        else:
            # 使用命令行参数
            ablation_faction = args.ablation_mode
            ablation_model = args.ablation_model
            print(f"[ABLATION MODE] 通过命令行参数启用消融实验", file=sys.stderr)
        
        print(f"[ABLATION MODE] 消融实验已启用", file=sys.stderr)
        print(f"[ABLATION MODE] 所有玩家使用模型: {ablation_model}", file=sys.stderr)
        print(f"[ABLATION MODE] MCTS+共情阵营: {ablation_faction}", file=sys.stderr)
        
        # 创建消融实验参数类（支持更灵活的配置）
        class AblationArgs:
            def __init__(self, base_args, roles_dict, wolf_mcts, good_mcts):
                for attr in dir(base_args):
                    if not attr.startswith('_'):
                        try:
                            setattr(self, attr, getattr(base_args, attr))
                        except:
                            pass
                self.roles_dict = roles_dict
                self.wolf_mcts = wolf_mcts
                self.good_mcts = good_mcts
                self._original_is_mcts = getattr(base_args, 'is_mcts', 1)
            
            def get_is_mcts_for_player(self, player_name):
                role = self.roles_dict.get(player_name, "")
                is_werewolf = role in ("werewolf", "wolf")
                
                if is_werewolf:
                    return 1 if self.wolf_mcts else 0
                else:
                    return 1 if self.good_mcts else 0
            
            @property
            def is_mcts(self):
                return self._original_is_mcts
        
        # 使用变量值创建AblationArgs
        ablation_args = AblationArgs(args, roles, Is_Wolf_MCTS, Is_Goods_MCTS)
        
        # 根据变量设置确定MCTS玩家
        werewolf_players = [p for p, r in roles.items() if r in ("werewolf", "wolf")]
        good_players = [p for p, r in roles.items() if r not in ("werewolf", "wolf")]
        
        mcts_players = []
        no_mcts_players = []
        if Is_Wolf_MCTS:
            mcts_players.extend(werewolf_players)
        else:
            no_mcts_players.extend(werewolf_players)
        if Is_Goods_MCTS:
            mcts_players.extend(good_players)
        else:
            no_mcts_players.extend(good_players)
        
        print(f"[ABLATION MODE] MCTS+共情玩家: {mcts_players}", file=sys.stderr)
        print(f"[ABLATION MODE] 基础版本玩家: {no_mcts_players}", file=sys.stderr)
        
        # 使用消融实验参数替换原始args
        args = ablation_args
    
    # ==== 将阵营绑定到不同OpenAI模型（gpt-3.5-turbo / gpt-4o）====
    # 实现：运行时替换各玩家 backend，使用不同的OpenAI模型
    # 两个阵营都使用OpenAIChat backend，但使用不同的模型，共享同一个API key
    try:
        from chatarena.backends import OpenAIChat, QwenChat, GeminiChat
        import random
        
        temperature = getattr(args, "temperature", 0.2)
        max_tokens = getattr(args, "max_tokens", 100)

        def _create_backend(model_name: str):
            lower = (model_name or "").lower()
            BackendCls = OpenAIChat
            if lower.startswith("gemini"):
                BackendCls = GeminiChat
            elif lower.startswith("qwen"):
                BackendCls = QwenChat
            return BackendCls(args, temperature=temperature, max_tokens=max_tokens, model=model_name)

        if is_ablation_mode:
            # 消融实验：所有玩家使用同一模型
            for p in arena.players:
                p.backend = _create_backend(ablation_model)
            
            # 为每个backend包装query方法以支持动态is_mcts
            for p in arena.players:
                original_query = p.backend.query
                
                def make_wrapped_query(backend_instance, player_name, ablation_args_obj):
                    def wrapped_query(arg, *query_args, **query_kwargs):
                        # 获取该玩家的is_mcts值
                        is_mcts_value = ablation_args_obj.get_is_mcts_for_player(player_name)
                        
                        # 创建临时args对象
                        class TempArgs:
                            def __init__(self, base_args, is_mcts_val):
                                for attr in dir(base_args):
                                    if not attr.startswith('_'):
                                        try:
                                            setattr(self, attr, getattr(base_args, attr))
                                        except:
                                            pass
                                self.is_mcts = is_mcts_val
                        
                        temp_args = TempArgs(arg if arg is not None else ablation_args_obj, is_mcts_value)
                        return original_query(temp_args, *query_args, **query_kwargs)
                    
                    return wrapped_query
                
                p.backend.query = make_wrapped_query(p.backend, p.name, ablation_args)
            
            api_type = "Gemini API" if ablation_model.lower().startswith("gemini") else "OpenAI API"
            print(f"[ABLATION MODE] 所有玩家使用模型: {ablation_model} (using {api_type})", file=sys.stderr)
            
            mapping_text = f"Announcement: [ABLATION EXPERIMENT] All players use {ablation_model.upper()}. {ablation_faction.upper()} faction uses MCTS+empathy, the other uses basic prompts."
        else:
            # 正常模式：不同阵营使用不同模型
            wolves_model = args.wolves_model
            goods_model = args.goods_model
            
            # 根据角色替换后端，按模型前缀挑选具体 backend
            for p in arena.players:
                role = roles.get(p.name, "")
                is_wolf = role in ("wolf", "werewolf")
                # 根据角色选择对应的模型
                model_to_use = wolves_model if is_wolf else goods_model
                p.backend = _create_backend(model_to_use)

            # 判断使用的 API 类型
            api_type = "Gemini API" if (wolves_model.lower().startswith("gemini") or goods_model.lower().startswith("gemini")) else "OpenAI API"
            print(f"[LLM ROUTING] wolves use: {wolves_model}, goods use: {goods_model} (using {api_type})", file=sys.stderr)

            mapping_text = f"Announcement: In this game, werewolves are powered by {wolves_model.upper()}, and villagers/special roles are powered by {goods_model.upper()}. Both use {api_type}."

        # 向所有玩家公告
        try:
            from chatarena.message import Message
            announce_msg = Message(agent_name="Moderator", content=mapping_text, turn=-1, visible_to="all", importance=5)
            arena.environment.message_pool.append_message_at_index(announce_msg, 1)
        except Exception as _e2:
            print(f"[LLM ROUTING ANNOUNCE WARN] {_e2}", file=sys.stderr)

        # 写入模型日志，便于离线校验
        try:
            with open("model_reply.log", "a", encoding="utf-8") as f:
                f.write(f"[SYSTEM] {mapping_text}\n")
        except Exception:
            pass
    except Exception as _e:
        print(f"[LLM ROUTING WARN] fallback to config backends, reason: {_e}", file=sys.stderr)
    
    # 原始评估器（保持兼容性）
    es = EvaluationSession(
        system_name=getattr(args, "system_name", "full_system"),
        model_name=getattr(args, "model_name", "default"),
        seed=getattr(args, "current_game_number", 0),
        game_id=str(arena.uuid),
    )
    es.on_game_start(players=players, roles=roles)
    
    # 增强评估器（详细指标）
    enhanced_evaluator = EnhancedWerewolfEvaluator(
        system_name=getattr(args, "system_name", "full_system"),
        model_name=getattr(args, "model_name", "default"),
        seed=getattr(args, "current_game_number", 0)
    )
    enhanced_evaluator.start_game(players, roles)
    
    # 如果是消融实验，初始化相关变量（用于结果导出）
    if is_ablation_mode:
        werewolf_players = [p for p, r in roles.items() if r in ("werewolf", "wolf")]
        good_players = [p for p, r in roles.items() if r not in ("werewolf", "wolf")]
        
        # 根据变量设置确定MCTS玩家
        mcts_players = []
        no_mcts_players = []
        if Is_Wolf_MCTS:
            mcts_players.extend(werewolf_players)
        else:
            no_mcts_players.extend(werewolf_players)
        if Is_Goods_MCTS:
            mcts_players.extend(good_players)
        else:
            no_mcts_players.extend(good_players)
        
        # 确定ablation_faction（用于结果统计）
        if Is_Wolf_MCTS and not Is_Goods_MCTS:
            ablation_faction = "werewolf"
        elif Is_Goods_MCTS and not Is_Wolf_MCTS:
            ablation_faction = "good"
        else:
            ablation_faction = "mixed"  # 两个都True或都False的情况
    else:
        mcts_players = []
        no_mcts_players = []
        ablation_faction = None
        ablation_model = None

    # 如果你有开局的共情结果（例如从 MCTS.llm_empathy_extract 获取），可在此记录一次快照：
    # es.on_empathy_snapshot(round_no=1, player_reports=empathy_json.get("player_reports", {}))

    # 用于检测每日处刑：保存上一时刻的存活列表
    prev_alive = list(arena.environment.alive_list)

    while True:
        try:
            print(f"[GAME] Executing step {arena.environment.current_turn if hasattr(arena.environment, 'current_turn') else 'unknown'}", file=sys.stderr)
            timestep = arena.step(args)

            # ==== 每步测评打点 ====
            current_phase = arena.environment.current_phase  # "day" 或 "night"
            current_turn = arena.environment.current_turn  # 第几天（白天）
            alive = list(arena.environment.alive_list)

            # 记录发言与可能的投票（从最后一条消息粗解析）
            msg_pool = arena.environment.message_pool
            
            # 检查是否处于投票阶段：通过检查环境状态或最近的主持人消息
            is_voting_phase = False
            if hasattr(arena.environment, "_number_of_rounds"):
                # _number_of_rounds == 1 表示投票阶段
                is_voting_phase = (arena.environment._number_of_rounds == 1)
            else:
                # 如果没有该属性，通过检查最近的主持人消息判断
                if hasattr(msg_pool, "_messages") and msg_pool._messages:
                    # 检查最近5条消息中是否有主持人要求投票的消息
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
                
                # 处理 content 可能是字符串或列表的情况
                if isinstance(content_raw, list):
                    # 如果是列表，取第一个元素（通常是文本内容）
                    content = str(content_raw[0]) if content_raw and isinstance(content_raw[0], str) else ""
                else:
                    content = str(content_raw) if content_raw else ""

                # 发言（没有结构化 talk_target/speech_style 时，先用占位）
                if speaker and speaker != "Moderator" and content:
                    # 原始评估器
                    es.on_speech(speaker=speaker, target_player="Unknown", speech_style="ambiguous")
                    
                    # 增强评估器
                    enhanced_evaluator.on_speech(speaker, content, current_phase, current_turn)

                    # 仅在投票阶段才检测投票，避免将讨论中的"投票"误判为实际投票
                    # 使用更严格的投票模式：只匹配明确的投票语句（如 "I vote to kill Player X"）
                    if current_phase == "day" and is_voting_phase:
                        # 更严格的投票模式：必须包含 "vote to kill" 或 "I vote" 等明确投票语句
                        strict_vote_pattern = re.compile(r"(?:I\s+)?(?:vote\s+to\s+kill|vote\s+for)\s+(Player\s+\d+)", re.IGNORECASE)
                        match = strict_vote_pattern.search(content)
                        if match:
                            vt = match.group(1)
                            # 原始评估器
                            es.on_vote(voter=speaker, target=vt)
                            # 增强评估器
                            enhanced_evaluator.on_vote(speaker, vt, current_phase, current_turn)

            # 检测白天处刑：上一时刻和当前存活列表对比
            if current_phase == "day":
                removed = [p for p in prev_alive if p not in alive]
                if removed:
                    # 原始评估器
                    es.on_day_end_lynch(lynched=removed[0])
                    # 增强评估器
                    enhanced_evaluator.on_lynch(removed[0], current_turn)

            # 检测夜晚行动
            if current_phase == "night":
                # 记录夜晚开始
                enhanced_evaluator.on_night_begin(current_turn)
                
                # 检测预言家查验（从消息中解析）
                if hasattr(msg_pool, "_messages") and msg_pool._messages:
                    for msg in msg_pool._messages[-5:]:  # 检查最近5条消息
                        # 处理 content 可能是字符串或列表的情况
                        msg_content = msg.content
                        if isinstance(msg_content, list):
                            # 如果是列表，取第一个元素（通常是文本内容）
                            if msg_content and isinstance(msg_content[0], str):
                                msg_content = msg_content[0]
                            else:
                                continue
                        elif not isinstance(msg_content, str):
                            continue
                        
                        msg_content_lower = msg_content.lower()
                        if "seer" in msg_content_lower and "verify" in msg_content_lower:
                            # 尝试解析预言家查验
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
                
                # 检测女巫行动（从消息中解析）
                if hasattr(msg_pool, "_messages") and msg_pool._messages:
                    for msg in msg_pool._messages[-5:]:  # 检查最近5条消息
                        # 处理 content 可能是字符串或列表的情况
                        msg_content = msg.content
                        if isinstance(msg_content, list):
                            # 如果是列表，取第一个元素（通常是文本内容）
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

            # 可选：夜晚开始（若你要记录预言家夜查，可以在环境夜查发生处调用 on_seer_peek）
            # if current_phase == "night":
            #     es.on_night_begin(night=current_turn)
            #     es.on_seer_peek(seer=seer_name, target=peek_target, is_wolf=(roles.get(peek_target) in ["wolf","werewolf"]))

            prev_alive = alive

        except TooManyInvalidActions as e:
            print(f"[GAME] ❌ Too many invalid actions: {e}", file=sys.stderr)
            timestep = arena.current_timestep
            timestep.observation.append(
                Message("System", "Too many invalid actions. Game over.", turn=-1, visible_to="all"))
            timestep.terminal = True  
        except Exception as e:
            print(f"[GAME] ❌ Unexpected error during game step: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            # Try to continue or terminate gracefully
            try:
                timestep = arena.current_timestep
                timestep.observation.append(
                    Message("System", f"Game error: {str(e)}. Game over.", turn=-1, visible_to="all"))
                timestep.terminal = True
            except:
                print("[GAME] ❌ Cannot recover from error, exiting", file=sys.stderr)
                break
            
        if timestep.terminal == True:

            # ==== 终局胜者判定 ====
            wolves_alive = [p for p in arena.environment.alive_list if roles.get(p) in ("wolf", "werewolf")]
            non_wolves_alive = [p for p in arena.environment.alive_list if roles.get(p) not in ("wolf", "werewolf")]
            if len(wolves_alive) == 0:
                winner = "villagers"
            elif len(wolves_alive) >= len(non_wolves_alive):
                winner = "werewolves"
            else:
                winner = "villagers"  # 兜底

            # 原始评估器
            es.on_game_end(winner=winner)
            
            # 增强评估器
            enhanced_evaluator.on_game_end(winner)

            # ==== 导出测评数据（JSON，包含你列的所有指标；不含性能）====
            # 原始评估器结果
            out_path = es.finalize_and_export(out_dir=os.path.join("experiment_results", "raw"), fmt="json")
            print("原始评测数据已保存：", out_path)
            
            # 增强评估器结果
            enhanced_path = enhanced_evaluator.export_detailed_results()
            print("详细评测数据已保存：", enhanced_path)
            
            # 如果是消融实验，导出消融对比结果
            if is_ablation_mode:
                ablation_results_path = os.path.join("experiment_results", "ablation", 
                                                     f"ablation_{args.current_game_number}.json")
                os.makedirs(os.path.dirname(ablation_results_path), exist_ok=True)
                
                final_alive = list(arena.environment.alive_list)
                mcts_alive = [p for p in final_alive if p in mcts_players]
                no_mcts_alive = [p for p in final_alive if p in no_mcts_players]
                
                # 判断哪个阵营获胜
                mcts_faction_won = False
                if ablation_faction == "werewolf":
                    mcts_faction_won = (winner == "werewolves")
                else:
                    mcts_faction_won = (winner == "villagers")
                
                ablation_stats = {
                    "experiment_id": args.current_game_number,
                    "model": ablation_model,
                    "mcts_faction": ablation_faction,
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

# 程序结尾或 finally 中
log_file.close()