"""
MCTS + Empathy Module for Werewolf Game

Design:
  - MCTS / empathy scoring: rule-based only, NO LLM calls
  - Final natural-language reply: generated once in openai.py via _get_response
"""
from typing import List, Dict, Tuple, Optional, Any, Union
import random
import logging
import math
import re
import json

logger = logging.getLogger("MCTS")

SPEECH_STYLES = (
    "soothe", "evidence", "align", "redirect", "counter",
    "reveal", "demand_info", "bargain", "humor", "ambiguous", "neutral",
)


def _default_player_report() -> Dict[str, Any]:
    return {
        "stance_to_me": 0.0,
        "emotion": {"pleasure": 0.5, "arousal": 0.5, "dominance": 0.5},
        "speech_acts_recent": ["neutral"],
        "politeness": 0.5,
        "consistency": 0.7,
        "influence": 0.6,
        "trust_score": 0.5,
        "hard_wolf_prob": 0.0,
        "soft_wolf_prob": 0.3,
        "evidence_type": "none",
        "accusation_count": 0,
        "bandwagon_risk": False,
        "information_gain": 0.0,
        "susceptibility": {
            "logic": 0.5, "emotion": 0.5, "authority": 0.5,
            "consensus": 0.5, "reciprocity": 0.5, "scarcity": 0.5, "commitment": 0.5,
        },
        "role_probability": {
            "werewolf": 0.3, "villager": 0.4,
            "seer": 0.1, "witch": 0.1, "guard": 0.1,
        },
        "recommended_action": "observe",
        "votes_received": 0,
        "votes_cast_targets": [],
        "vote_bandwagon_rate": 0.0,
        "speech_vote_consistency": 1.0,
        "eliminated_by_vote": False,
        "died_at_night": False,
        "post_death_last_words": "",
        "current_round_vote_pressure": 0.0,
        "posterior_wolf_prob": 0.3,
        "private_verified_good": False,
        "private_verified_wolf": False,
        "support_targets": [],
        "supported_by": [],
        "relation_index": [],
        "semantic_memory": "",
        "uncertainty_notes": "",
        "reflection_sketch": {
            "what_i_know": "",
            "what_might_be_fake": "",
            "what_conflicts_exist": "",
            "what_to_do_next": "",
        },
        "claim_type": "",
        "claim_target": "",
        "claim_strength": 0.0,
        "misdirection_risk": 0.0,
        "support_uncertainty": 0.5,
        "clear_uncertainty": 0.5,
        "support_summary": "",
    }


def _is_player_report_key(key: str) -> bool:
    return bool(re.match(r"^Player\s+\d+$", str(key).strip(), re.I))


def _ensure_report_schema(report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Guarantee nested dict shape (LLM / meta keys must not break role_probability)."""
    base = _default_player_report()
    if not isinstance(report, dict):
        return base
    for k, v in report.items():
        if k == "role_probability":
            if isinstance(v, dict):
                base["role_probability"] = {**_default_player_report()["role_probability"], **v}
            elif isinstance(v, (int, float)):
                wp = float(v)
                base["role_probability"] = {
                    "werewolf": wp,
                    "villager": max(0.0, 1.0 - wp - 0.2),
                    "seer": 0.1,
                    "witch": 0.1,
                    "guard": 0.1,
                }
            continue
        if k == "emotion" and isinstance(v, dict):
            base["emotion"].update(v)
            continue
        if k == "susceptibility" and isinstance(v, dict):
            base["susceptibility"].update(v)
            continue
        if k == "reflection_sketch" and isinstance(v, dict):
            base["reflection_sketch"].update(v)
            continue
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k].update(v)
        else:
            base[k] = v
    if "werewolf_prob" in report and "werewolf" not in base["role_probability"]:
        wp = float(report["werewolf_prob"])
        base["role_probability"]["werewolf"] = wp
    if "posterior_wolf_prob" in report:
        wp = float(report["posterior_wolf_prob"])
        base["role_probability"]["werewolf"] = wp
        base["posterior_wolf_prob"] = wp
    if not isinstance(base.get("role_probability"), dict):
        base["role_probability"] = dict(_default_player_report()["role_probability"])
    return base


def _player_reports_only(reports: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        k: _ensure_report_schema(v)
        for k, v in (reports or {}).items()
        if _is_player_report_key(k)
    }


def _split_player_and_meta_reports(
    reports: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    players: Dict[str, Dict[str, Any]] = {}
    meta: Dict[str, Any] = {}
    for k, v in (reports or {}).items():
        if _is_player_report_key(k):
            players[k] = _ensure_report_schema(v)
        elif str(k).startswith("_"):
            meta[k] = v
    return players, meta


def _posterior_wolf_prob(report: Dict[str, Any]) -> float:
    """Single belief scalar for MCTS — prefer BeliefState-synced posterior when present."""
    if not report:
        return 0.3
    if "posterior_wolf_prob" in report:
        return float(report["posterior_wolf_prob"])
    return float(report.get("role_probability", {}).get("werewolf", 0.3))


def get_unified_prior_decision(game_state, agent_name: str) -> Optional[Dict[str, Any]]:
    """Discussion commitment for vote-phase MCTS (BeliefState first, then cache)."""
    try:
        from .belief_state import get_belief_state
    except ImportError:
        from chatarena.belief_state import get_belief_state
    belief = get_belief_state(game_state)
    if belief and belief.viewer == agent_name:
        return belief.get_prior_decision()
    return None


# ============ Integrated game analytics (speech + votes + deaths) ============

def _norm_player_id(raw: str) -> str:
    m = re.search(r"player\s*(\d+)", str(raw).lower())
    if m:
        return f"Player {m.group(1)}"
    s = str(raw).strip()
    if s.startswith("Player") and len(s.split()) >= 2:
        return f"Player {s.split()[1]}"
    return s


def _empty_player_profile() -> Dict[str, Any]:
    return {
        "speeches": 0,
        "trust_mentioned": [],
        "accuse_mentioned": [],
        "votes_cast": [],
        "votes_received_history": [],
        "bandwagon_votes": 0,
        "total_day_votes": 0,
        "eliminated_by_day_vote": False,
        "night_death": False,
        "last_words": "",
    }


def extract_game_analytics(history: List[Tuple[str, str]]) -> Dict[str, Any]:
    """
    Unified public timeline: discussion, sequential votes, eliminations, night deaths.
    Shared by all roles — MCTS/empathy read the same vote psychology model.
    """
    analytics: Dict[str, Any] = {
        "vote_rounds": [],
        "current_round_votes": {},
        "in_voting_phase": False,
        "deaths": [],
        "player_profiles": {},
        "round_idx": 0,
    }

    def _prof(name: str) -> Dict[str, Any]:
        if name not in analytics["player_profiles"]:
            analytics["player_profiles"][name] = _empty_player_profile()
        return analytics["player_profiles"][name]

    def _flush_vote_round(eliminated: str = ""):
        if not analytics["current_round_votes"]:
            return
        tally: Dict[str, int] = {}
        for _, tgt in analytics["current_round_votes"].items():
            if tgt and tgt != "pass":
                tally[tgt] = tally.get(tgt, 0) + 1
        majority_target = max(tally, key=tally.get) if tally else ""
        majority_count = tally.get(majority_target, 0) if majority_target else 0
        for voter, tgt in analytics["current_round_votes"].items():
            p = _prof(voter)
            p["votes_cast"].append({"round": analytics["round_idx"], "target": tgt})
            p["total_day_votes"] += 1
            if majority_target and tgt == majority_target and majority_count >= 2:
                p["bandwagon_votes"] += 1
        for tgt, cnt in tally.items():
            pr = _prof(tgt)
            pr["votes_received_history"].append(
                {"round": analytics["round_idx"], "count": cnt, "voters": [
                    v for v, t in analytics["current_round_votes"].items() if t == tgt
                ]},
            )
        analytics["vote_rounds"].append({
            "round": analytics["round_idx"],
            "votes": dict(analytics["current_round_votes"]),
            "tally": tally,
            "majority_target": majority_target,
            "majority_count": majority_count,
            "eliminated": eliminated,
        })
        analytics["current_round_votes"] = {}

    in_discussion = False
    for speaker, content in history or []:
        cl = str(content).lower()
        sp = speaker if speaker.startswith("Player") else ""

        if "discussion phase" in cl or "freely talk" in cl:
            in_discussion = True
            analytics["in_voting_phase"] = False
            continue
        if "voting phase" in cl or ("must vote" in cl and "vote to kill" in cl):
            in_discussion = False
            if not analytics["in_voting_phase"]:
                analytics["current_round_votes"] = {}
            analytics["in_voting_phase"] = True
            continue
        if "will be killed" in cl and ("moderator" in speaker.lower() or speaker == "Moderator"):
            m = re.search(r"(player \d+) will be killed", cl)
            if m:
                eliminated = _norm_player_id(m.group(1))
                _flush_vote_round(eliminated)
                _prof(eliminated)["eliminated_by_day_vote"] = True
                analytics["deaths"].append({
                    "player": eliminated, "cause": "vote", "round": analytics["round_idx"],
                })
                analytics["round_idx"] += 1
                analytics["in_voting_phase"] = False
            continue
        if "died last night" in cl:
            m = re.search(r"(player \d+) died", cl)
            if m:
                dead = _norm_player_id(m.group(1))
                _prof(dead)["night_death"] = True
                analytics["deaths"].append({
                    "player": dead, "cause": "night", "round": analytics["round_idx"],
                })
                analytics["round_idx"] += 1
            continue
        if "peaceful night" in cl and "no one died" in cl:
            analytics["round_idx"] += 1
            continue

        if sp:
            if in_discussion or not analytics["in_voting_phase"]:
                pr = _prof(sp)
                pr["speeches"] += 1
                for pat in (
                    r"trust(?:s|ing)?\s+(player \d+)",
                    r"agree with (player \d+)",
                    r"(player \d+).{0,30}(?:solid|good|not suspicious)",
                ):
                    for m in re.finditer(pat, cl):
                        tgt = _norm_player_id(m.group(1))
                        if tgt.startswith("Player") and tgt not in pr["trust_mentioned"]:
                            pr["trust_mentioned"].append(tgt)
                for acc in _extract_accusations_from_message(sp, content):
                    tgt = acc["target"]
                    if tgt not in pr["accuse_mentioned"]:
                        pr["accuse_mentioned"].append(tgt)

            if "i vote to kill" in cl:
                m = re.search(r"i vote to kill (player \d+)", cl)
                if m:
                    tgt = _norm_player_id(m.group(1))
                    analytics["current_round_votes"][sp] = tgt
                    analytics["in_voting_phase"] = True

            if "last statement" in cl or len(content) > 30:
                if any(kw in cl for kw in ("werewolf", "guard", "seer", "witch", "verified", "not a werewolf")):
                    _prof(sp)["last_words"] = str(content)[:200]

    if analytics["current_round_votes"]:
        tally = {}
        for _, tgt in analytics["current_round_votes"].items():
            if tgt and tgt != "pass":
                tally[tgt] = tally.get(tgt, 0) + 1
        analytics["current_round_tally"] = tally
        if tally:
            lead = max(tally, key=tally.get)
            analytics["current_round_leader"] = lead
            analytics["current_round_leader_count"] = tally[lead]
        else:
            analytics["current_round_leader"] = ""
            analytics["current_round_leader_count"] = 0
    else:
        analytics["current_round_tally"] = {}
        analytics["current_round_leader"] = ""
        analytics["current_round_leader_count"] = 0

    return analytics


def _speech_vote_consistency(player: str, profile: Dict[str, Any]) -> float:
    """1.0 = speech aligns with votes; lower = praised then voted against, etc."""
    trust_set = set(profile.get("trust_mentioned", []))
    votes_cast = profile.get("votes_cast", [])
    if not votes_cast:
        return 1.0
    conflicts = 0
    for vc in votes_cast:
        tgt = vc.get("target", "")
        if tgt in trust_set:
            conflicts += 1
    if not votes_cast:
        return 1.0
    return max(0.0, 1.0 - conflicts / max(1, len(votes_cast)))


def apply_game_analytics_to_reports(
    reports: Dict[str, Dict[str, Any]],
    analytics: Dict[str, Any],
    agent_name: str,
    game_phase: str,
    my_role: str,
) -> Dict[str, Dict[str, Any]]:
    """Merge vote/death/speech timeline into per-player empathy reports (all roles)."""
    profiles = analytics.get("player_profiles", {})
    current_votes = analytics.get("current_round_votes", {})
    current_tally = analytics.get("current_round_tally", {})
    leader = analytics.get("current_round_leader", "")
    leader_count = int(analytics.get("current_round_leader_count", 0))

    for player, report in reports.items():
        if not _is_player_report_key(player) or not isinstance(report, dict):
            continue
        report = _ensure_report_schema(report)
        prof = profiles.get(player, _empty_player_profile())
        total_cast = max(1, int(prof.get("total_day_votes", 0)))
        bandwagon_rate = prof.get("bandwagon_votes", 0) / total_cast
        recv_hist = prof.get("votes_received_history", [])
        votes_recv = sum(h.get("count", 0) for h in recv_hist)
        if analytics.get("in_voting_phase") and player in current_tally:
            votes_recv = current_tally.get(player, votes_recv)

        report["votes_received"] = votes_recv
        report["votes_cast_targets"] = [v.get("target") for v in prof.get("votes_cast", [])]
        report["vote_bandwagon_rate"] = round(bandwagon_rate, 3)
        report["speech_vote_consistency"] = round(_speech_vote_consistency(player, prof), 3)
        report["eliminated_by_vote"] = prof.get("eliminated_by_day_vote", False)
        report["died_at_night"] = prof.get("night_death", False)
        report["post_death_last_words"] = prof.get("last_words", "")
        report["supported_by"] = list(dict.fromkeys(report.get("supported_by", [])))
        report["support_targets"] = list(dict.fromkeys(report.get("support_targets", [])))
        report.setdefault("relation_index", [])
        report.setdefault("semantic_memory", "")
        report.setdefault("uncertainty_notes", "")
        report.setdefault("reflection_sketch", {
            "what_i_know": "",
            "what_might_be_fake": "",
            "what_conflicts_exist": "",
            "what_to_do_next": "",
        })
        report.setdefault("support_uncertainty", 0.5)
        report.setdefault("clear_uncertainty", 0.5)
        report.setdefault("support_summary", "")

        pressure = 0.0
        if analytics.get("in_voting_phase") and player in current_tally:
            pressure = current_tally[player] / max(1, len(current_votes) + 1)
        elif leader == player and leader_count >= 2:
            pressure = leader_count / 7.0
        report["current_round_vote_pressure"] = round(pressure, 3)

        hard = float(report.get("hard_wolf_prob", 0.0))
        soft = float(report.get("soft_wolf_prob", 0.3))

        if bandwagon_rate >= 0.67 and votes_recv < 2:
            soft = min(0.85, soft + 0.06)
            report["bandwagon_risk"] = True

        consistency = report["speech_vote_consistency"]
        if consistency < 0.5:
            soft = min(0.9, soft + 0.12)
            hard = min(0.85, hard + 0.08)

        # Use support relations to reduce suspicion on supported targets,
        # but keep the supporter's misdirection risk if the support is a self-claim.
        support_targets = report.get("support_targets", []) or []
        if support_targets:
            for st in support_targets:
                if st in reports:
                    reports[st]["role_probability"]["werewolf"] = max(
                        0.02, float(reports[st]["role_probability"].get("werewolf", 0.3)) - 0.12
                    )
                    reports[st]["trust_score"] = round(1.0 - reports[st]["role_probability"]["werewolf"], 3)
                    reports[st]["supported_by"] = list(dict.fromkeys(reports[st].get("supported_by", []) + [player]))
                    reports[st]["relation_index"] = list(dict.fromkeys(reports[st].get("relation_index", []) + ["supported_by_claim"]))
                    if st == player:
                        report["support_uncertainty"] = min(1.0, float(report.get("support_uncertainty", 0.5)) + 0.25)

        if prof.get("trust_mentioned") and prof.get("votes_cast"):
            last_votes = [v.get("target") for v in prof["votes_cast"][-2:]]
            for trusted in prof["trust_mentioned"]:
                if trusted in last_votes:
                    soft = min(0.88, soft + 0.1)

        if votes_recv >= 3 and game_phase == "voting":
            soft = min(0.92, soft + 0.14)
            if not any(
                a.get("evidence_type") == "hard"
                for rnd in analytics.get("vote_rounds", [])
                for a in []  # day vote pile-on without hard claim in discussion
            ):
                report["bandwagon_pile_on"] = True

        if prof.get("eliminated_by_day_vote") and prof.get("last_words"):
            lw = prof["last_words"].lower()
            if any(k in lw for k in ("seer", "witch", "guard", "verified", "not a werewolf")):
                for other, oreport in reports.items():
                    if other == player or other == "_game":
                        continue
                    oprof = profiles.get(other, _empty_player_profile())
                    cast = [v.get("target") for v in oprof.get("votes_cast", [])]
                    if player in cast:
                        oreport["soft_wolf_prob"] = min(
                            0.92, float(oreport.get("soft_wolf_prob", 0.3)) + 0.15
                        )

        combined = min(0.95, max(0.05, hard * 0.68 + soft * 0.32))
        report["hard_wolf_prob"] = round(hard, 3)
        report["soft_wolf_prob"] = round(soft, 3)
        report["role_probability"]["werewolf"] = round(combined, 3)
        report["trust_score"] = round(1.0 - combined, 3)

    agent_prof = profiles.get(agent_name, _empty_player_profile())
    reports.setdefault("_game", {})
    reports["_game"] = {
        "vote_rounds_count": len(analytics.get("vote_rounds", [])),
        "current_tally": current_tally,
        "current_leader": leader,
        "current_leader_count": leader_count,
        "my_trust_mentioned": list(agent_prof.get("trust_mentioned", [])),
        "my_votes_cast": [v.get("target") for v in agent_prof.get("votes_cast", [])],
        "in_voting_phase": analytics.get("in_voting_phase", False),
    }
    return reports


def format_game_analytics_brief(
    analytics: Dict[str, Any],
    agent_name: str,
    empathy_data: Optional[Dict[str, Any]] = None,
) -> str:
    """Concise vote/death summary for LLM empathy + speech (shared strategy)."""
    lines = []
    rounds = analytics.get("vote_rounds", [])
    if rounds:
        lines.append("## Vote history (algorithm)")
        for vr in rounds[-3:]:
            tally = vr.get("tally", {})
            tally_s = ", ".join(f"{k}:{v}" for k, v in sorted(tally.items(), key=lambda x: -x[1]))
            elim = vr.get("eliminated", "")
            lines.append(f"- Round {vr.get('round')}: {tally_s or 'no tally'}" + (f" → eliminated {elim}" if elim else ""))

    tally = analytics.get("current_round_tally") or analytics.get("current_round_votes", {})
    if analytics.get("in_voting_phase") and tally:
        if isinstance(tally, dict) and tally and "Player" in str(next(iter(tally.keys()), "")):
            ts = ", ".join(f"{k}:{v}" for k, v in sorted(tally.items(), key=lambda x: -x[1]))
        else:
            ts = str(tally)
        lines.append(f"## Current vote tally (before/at {agent_name}'s vote): {ts}")
        leader = analytics.get("current_round_leader", "")
        if leader:
            lines.append(f"- Leading elimination target: {leader} ({analytics.get('current_round_leader_count', 0)} votes)")

    deaths = analytics.get("deaths", [])
    if deaths:
        lines.append("## Deaths")
        for d in deaths[-4:]:
            lines.append(f"- {d.get('player')}: {d.get('cause')} (round {d.get('round')})")

    game_meta = (empathy_data or {}).get("_game", {})
    priv = (empathy_data or {}).get("_private") or {}
    if priv.get("verified_good"):
        lines.append(f"- Your private checks (good): {', '.join(priv['verified_good'])}")
    if priv.get("verified_wolf"):
        lines.append(f"- Your private checks (wolf): {', '.join(priv['verified_wolf'])}")
    if game_meta.get("my_trust_mentioned"):
        lines.append(f"- You publicly leaned trust toward: {', '.join(game_meta['my_trust_mentioned'])}")
    if game_meta.get("my_votes_cast"):
        lines.append(f"- Your past day votes: {', '.join(game_meta['my_votes_cast'])}")

    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def get_game_analytics(game_state, force_refresh: bool = False) -> Dict[str, Any]:
    if not force_refresh:
        cached = _get_state_field(game_state, "game_analytics", None)
        if isinstance(cached, dict) and cached.get("player_profiles") is not None:
            return cached
    history = _get_state_field(game_state, "history", []) or []
    analytics = extract_game_analytics(history)
    analytics["history_len"] = len(history)
    analytics["refreshed_at_phase"] = _get_state_field(game_state, "game_phase", "")
    if isinstance(game_state, dict):
        game_state["game_analytics"] = analytics
    else:
        try:
            game_state["game_analytics"] = analytics
        except (TypeError, KeyError):
            pass
    return analytics


def history_from_messages(history_messages, task_content: str = "") -> List[Tuple[str, str]]:
    """Build full observable timeline from message pool (incl. Moderator + partial votes)."""
    history: List[Tuple[str, str]] = []
    if history_messages:
        for msg in history_messages:
            if isinstance(msg, tuple):
                speaker, content = msg[0], msg[1] or ""
            else:
                speaker = getattr(msg, "agent_name", "unknown")
                content = getattr(msg, "content", "") or ""
            history.append((speaker, content))
    return append_live_moderator_task(history, task_content)


def append_live_moderator_task(
    history: List[Tuple[str, str]], task_content: str
) -> List[Tuple[str, str]]:
    """Append current moderator prompt so voting-phase / latest discussion cues are visible."""
    tc = (task_content or "").strip()
    if not tc:
        return history
    lower = tc.lower()
    if not any(k in lower for k in ("voting phase", "discussion phase", "must vote", "freely talk")):
        return history
    if history and history[-1][1].strip() == tc:
        return history
    if "voting phase" in lower or "must vote" in lower:
        for i in range(len(history) - 1, -1, -1):
            sp, ct = history[i]
            if sp.startswith("Player") and "i vote to kill" in (ct or "").lower():
                return history
            if "voting phase" in (ct or "").lower() and "must vote" in (ct or "").lower():
                break
    return history + [("Moderator", tc)]


def extract_private_beliefs(
    agent_name: str,
    my_role: str,
    history: List[Tuple[str, str]],
) -> Dict[str, Any]:
    """
    Hard-update facts visible only to this agent (moderator whispers).
    Not broadcast — used to adjust posterior, then reason about public contradictions.
    """
    beliefs: Dict[str, Any] = {
        "verified_good": [],
        "verified_wolf": [],
        "saved_target": "",
        "poisoned_target": "",
        "protected_target": "",
        "wolf_teammates": [],
        "private_facts": [],
    }
    my_speeches = " ".join(c for s, c in history if s == agent_name).lower()

    for speaker, content in history or []:
        cl = str(content).lower()
        is_mod = speaker == "Moderator" or "moderator" in str(speaker).lower()

        if my_role == "seer" and is_mod:
            m = re.search(r"(player \d+) is not a werewolf", content, re.I)
            if m:
                p = _norm_player_id(m.group(1))
                if p not in beliefs["verified_good"]:
                    beliefs["verified_good"].append(p)
                    beliefs["private_facts"].append(f"verified_good:{p}")
            m = re.search(r"(player \d+) is a werewolf", content, re.I)
            if m:
                p = _norm_player_id(m.group(1))
                if p not in beliefs["verified_wolf"]:
                    beliefs["verified_wolf"].append(p)
                    beliefs["private_facts"].append(f"verified_wolf:{p}")

        if my_role == "witch" and is_mod:
            # 只在主持人明确要求“是否使用解药”时，才把这轮记为解药已用/待用
            asks_save = ("do you want to save" in cl) or ("antidote" in cl) or ("will die tonight" in cl and "yes" in cl)
            asks_poison = ("who are you going to kill" in cl) or ("who are you going to poison" in cl)
            if asks_save:
                m = re.search(r"(player \d+) was attacked", content, re.I)
                if m and "will die tonight" in cl:
                    beliefs["saved_target"] = _norm_player_id(m.group(1))
                    beliefs["private_facts"].append(f"attack_target:{beliefs['saved_target']}")
                if "yes" in my_speeches and beliefs["saved_target"]:
                    beliefs["private_facts"].append(f"used_antidote_on:{beliefs['saved_target']}")
            # 只有真正的毒药询问才允许记录 poison 相关信息
            if asks_poison:
                m = re.search(r"(player \d+)", content, re.I)
                if m:
                    beliefs["poisoned_target"] = _norm_player_id(m.group(1))
                    beliefs["private_facts"].append(f"poison_prompt:{beliefs['poisoned_target']}")

        if my_role == "guard" and speaker == agent_name and "protect" in cl:
            m = re.search(r"protect (player \d+)", cl)
            if m:
                beliefs["protected_target"] = _norm_player_id(m.group(1))
                beliefs["private_facts"].append(f"protected:{beliefs['protected_target']}")

        if my_role in ("werewolf", "wolf") and is_mod and "werewolves" in cl:
            for m in re.finditer(r"player \d+", content, re.I):
                p = _norm_player_id(m.group(0))
                if p != agent_name and p not in beliefs["wolf_teammates"]:
                    beliefs["wolf_teammates"].append(p)

    return beliefs


def project_private_public_contradictions(
    reports: Dict[str, Dict[str, Any]],
    private_beliefs: Dict[str, Any],
    analytics: Dict[str, Any],
    agent_name: str,
) -> Dict[str, Dict[str, Any]]:
    """
    If I privately know P is good but the village is piling votes on P,
    raise wolf posterior on voters (public reasoning), not on P.
    """
    verified_good = set(private_beliefs.get("verified_good", []))
    if not verified_good:
        return reports

    current_votes = analytics.get("current_round_votes", {}) or {}
    tally = analytics.get("current_round_tally", {}) or {}

    for good_p in verified_good:
        if good_p in reports:
            reports[good_p]["private_verified_good"] = True
            reports[good_p]["posterior_wolf_prob"] = min(
                float(reports[good_p].get("posterior_wolf_prob", 0.3)), 0.1
            )
            reports[good_p]["role_probability"]["werewolf"] = reports[good_p]["posterior_wolf_prob"]
            reports[good_p]["trust_score"] = round(1.0 - reports[good_p]["posterior_wolf_prob"], 3)

        vote_count = tally.get(good_p, 0)
        if vote_count < 2 and good_p not in tally:
            continue

        for voter, tgt in current_votes.items():
            if tgt != good_p or voter == agent_name or voter == "_game":
                continue
            if voter not in reports:
                continue
            r = reports[voter]
            post = float(r.get("posterior_wolf_prob", r.get("role_probability", {}).get("werewolf", 0.3)))
            post = min(0.92, post + 0.22 + 0.05 * vote_count)
            r["posterior_wolf_prob"] = round(post, 3)
            r["role_probability"]["werewolf"] = round(post, 3)
            r["soft_wolf_prob"] = round(max(float(r.get("soft_wolf_prob", 0.3)), post * 0.85), 3)
            r["trust_score"] = round(1.0 - post, 3)
            r["public_contradiction"] = f"voted_verified_good:{good_p}"

    for wolf_p in private_beliefs.get("verified_wolf", []):
        if wolf_p in reports:
            reports[wolf_p]["private_verified_wolf"] = True
            reports[wolf_p]["posterior_wolf_prob"] = max(
                float(reports[wolf_p].get("posterior_wolf_prob", 0.3)), 0.9
            )
            reports[wolf_p]["role_probability"]["werewolf"] = reports[wolf_p]["posterior_wolf_prob"]
            reports[wolf_p]["trust_score"] = round(1.0 - reports[wolf_p]["posterior_wolf_prob"], 3)

    return reports


def fuse_posterior_beliefs(
    reports: Dict[str, Dict[str, Any]],
    game_state,
    agent_name: str,
) -> Dict[str, Dict[str, Any]]:
    """Delegate to unified BeliefState (single posterior for empathy + MCTS)."""
    try:
        from .belief_state import build_and_attach_belief_state
    except ImportError:
        from chatarena.belief_state import build_and_attach_belief_state
    _, synced = build_and_attach_belief_state(game_state, agent_name, reports)
    return synced


def refresh_live_observation(
    game_state,
    history_messages=None,
    agent_name: str = "",
    my_role: str = "",
    task_content: str = "",
    sync_public_field=None,
) -> Dict[str, Any]:
    """Lightweight live refresh: only rebuild when message history changed."""
    if history_messages is not None:
        hist = history_from_messages(history_messages, task_content)
        prev_len = len(_get_state_field(game_state, "history", []) or [])
        new_len = len(hist)
        if isinstance(game_state, dict):
            game_state["history"] = hist
            game_state["history_cache_len"] = new_len
        else:
            game_state["history"] = hist
            game_state["history_cache_len"] = new_len
    elif task_content:
        hist = append_live_moderator_task(
            _get_state_field(game_state, "history", []) or [], task_content
        )
        new_len = len(hist)
        if isinstance(game_state, dict):
            game_state["history"] = hist
            game_state["history_cache_len"] = new_len
        else:
            game_state["history"] = hist
            game_state["history_cache_len"] = new_len
    else:
        hist = _get_state_field(game_state, "history", []) or []
        new_len = len(hist)

    if agent_name and my_role:
        private = extract_private_beliefs(
            agent_name, my_role, _get_state_field(game_state, "history", []) or []
        )
        if isinstance(game_state, dict):
            game_state["private_beliefs"] = private
        else:
            game_state["private_beliefs"] = private

    if sync_public_field is not None:
        try:
            from .empathy_field import sync_field_from_history
        except ImportError:
            from chatarena.empathy_field import sync_field_from_history
        hist = _get_state_field(game_state, "history", []) or []
        sync_public_field.round_no = _get_state_field(game_state, "round_no", 1)
        sync_field_from_history(sync_public_field, hist)
        if isinstance(game_state, dict):
            game_state["public_empathy_field"] = sync_public_field.to_dict()

    cached = _get_state_field(game_state, "game_analytics", None)
    cached_len = _get_state_field(game_state, "analytics_cache_len", None)
    if isinstance(cached, dict) and cached.get("player_profiles") is not None and cached_len == new_len:
        analytics = cached
    else:
        analytics = get_game_analytics(game_state, force_refresh=True)
        if isinstance(game_state, dict):
            game_state["analytics_cache_len"] = new_len
        else:
            game_state["analytics_cache_len"] = new_len
    logger.info(
        f"[LiveObs] {agent_name} phase={_get_state_field(game_state, 'game_phase')} "
        f"in_vote={analytics.get('in_voting_phase')} "
        f"partial_votes={len(analytics.get('current_round_votes', {}))} "
        f"tally={analytics.get('current_round_tally', {})}",
    )
    return analytics


def refresh_empathy_reports_live(
    game_state,
    agent_name: str,
    empathy_data: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Re-run rule enrich + BeliefState fuse without LLM (after live observation refresh)."""
    reports = dict(empathy_data or {})
    reports = enrich_empathy_reports(game_state, agent_name, reports, skip_belief_fuse=True)
    try:
        from .belief_state import build_and_attach_belief_state
    except ImportError:
        from chatarena.belief_state import build_and_attach_belief_state
    analytics = get_game_analytics(game_state, force_refresh=True)
    _, reports = build_and_attach_belief_state(
        game_state, agent_name, reports, analytics=analytics
    )
    return reports


# ============ Action / Node ============

class ActionResult:
    def __init__(self, action_type: str, target: str, speech: str = "", style: str = "neutral"):
        self.action_type = action_type
        self.target = target
        self.speech = speech
        self.style = style

    def to_tuple(self) -> Tuple[str, str, str]:
        return (self.action_type, self.target, self.style)

    def __repr__(self):
        preview = (self.speech or "")[:30]
        return f"ActionResult(type={self.action_type}, target={self.target}, speech={preview}...)"


class MCTSNode:
    def __init__(self, state, player: str, parent: Optional["MCTSNode"] = None, action=None):
        self.state = state
        self.player = player
        self.parent = parent
        self.action = action if action else ("pass", "pass", "ambiguous")
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.value = 0.0
        self.empathy_context: Dict[str, Any] = {}
        self.intent_meta: Dict[str, Any] = {}

    def ucb_score(self, c: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        exploitation = self.value / self.visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits) if self.parent else 0
        return exploitation + exploration

    def best_child(self) -> Optional["MCTSNode"]:
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.ucb_score())


# ============ GameState ============

class GameState(dict):
    """Game state container used by MCTS and empathy modules."""

    _ATTR_KEYS = (
        "alive_players", "player_roles", "day_night", "night_kill_list",
        "player_order", "history", "votes", "round_no", "game_phase",
        "my_role", "current_player", "observation",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setdefault("alive_players", [])
        self.setdefault("player_roles", {})
        self.setdefault("day_night", "daytime")
        self.setdefault("night_kill_list", [])
        self.setdefault("player_order", [])
        self.setdefault("history", [])
        self.setdefault("votes", [])
        self.setdefault("round_no", 1)
        self.setdefault("game_phase", "discussion")
        self.setdefault("my_role", "villager")
        self.setdefault("current_player", "")
        self.setdefault("observation", "")

        for key, value in kwargs.items():
            self[key] = value

    @property
    def alive_players(self) -> List[str]:
        return self.get("alive_players", [])

    @alive_players.setter
    def alive_players(self, value: List[str]):
        self["alive_players"] = value

    @property
    def player_roles(self) -> Dict[str, str]:
        return self.get("player_roles", {})

    @player_roles.setter
    def player_roles(self, value: Dict[str, str]):
        self["player_roles"] = value

    @property
    def day_night(self) -> str:
        return self.get("day_night", "daytime")

    @day_night.setter
    def day_night(self, value: str):
        self["day_night"] = value

    @property
    def my_role(self) -> str:
        return self.get("my_role", "villager")

    @my_role.setter
    def my_role(self, value: str):
        self["my_role"] = value

    @property
    def round_no(self) -> int:
        return self.get("round_no", 1)

    @round_no.setter
    def round_no(self, value: int):
        self["round_no"] = value

    @property
    def history(self) -> List[Tuple[str, str]]:
        return self.get("history", [])

    @history.setter
    def history(self, value: List[Tuple[str, str]]):
        self["history"] = value

    def kill_player(self, player_name: str):
        if player_name in self.alive_players:
            alive = list(self.alive_players)
            alive.remove(player_name)
            self.alive_players = alive
            kills = list(self.get("night_kill_list", []))
            kills.append(player_name)
            self["night_kill_list"] = kills

    def apply_action(self, player: str, action: Tuple[str, str, str]) -> "GameState":
        new_state = GameState(dict(self))
        target = action[0]
        if target and target != "pass" and target in new_state.alive_players and target != player:
            new_state["votes"] = list(new_state.get("votes", [])) + [(player, target)]
        return new_state

    def __str__(self):
        return (
            f"GameState(alive={self.alive_players}, role={self.my_role}, "
            f"phase={self.get('game_phase', 'discussion')}, round={self.round_no})"
        )


def detect_game_phase(alive_players: List[str]) -> str:
    count = len(alive_players)
    if count <= 3:
        return "endgame"
    if count <= 5:
        return "midgame"
    return "earlygame"


# ============ Empathy (rule-based, no LLM) ============

def _get_state_field(state, name, default=None):
    if isinstance(state, dict):
        return state.get(name, default)
    return getattr(state, name, default)


_HARD_CLAIM_PATTERNS = (
    r"i am the seer", r"i am seer", r"i verified", r"verified .+ werewolf",
    r"is a werewolf", r"is not a werewolf", r"i saved", r"used my antidote",
    r"i protected", r"i am the witch", r"i am witch", r"i am the guard",
)
_SOFT_READ_PATTERNS = (
    r"suspicious", r"defensive", r"quiet", r"hasn'?t spoken", r"has not spoken",
    r"reacted", r"deflect", r"strange", r"odd", r"interesting that",
)
_ACCUSE_PATTERNS = (
    r"suspicious of (player \d+)", r"doubt (player \d+)", r"targeting (player \d+)",
    r"vote (?:to kill|against) (player \d+)", r"focus(?:ing)? on (player \d+)",
    r"(player \d+) (?:is|seems|looks) suspicious",
)


def _classify_evidence_in_text(text: str) -> str:
    lower = str(text).lower()
    if any(re.search(p, lower) for p in _HARD_CLAIM_PATTERNS):
        return "hard"
    if any(re.search(p, lower) for p in _SOFT_READ_PATTERNS):
        return "soft"
    return "none"


def _extract_accusations_from_message(speaker: str, content: str) -> List[Dict[str, str]]:
    out = []
    lower = str(content).lower()
    ev_type = _classify_evidence_in_text(content)
    for pat in _ACCUSE_PATTERNS:
        for m in re.finditer(pat, lower):
            target = m.group(1)
            if target.startswith("player"):
                target = "Player " + target.split()[-1] if " " not in target.replace("player", "").strip() else target.title()
            if not target.startswith("Player"):
                continue
            parts = target.split()
            if len(parts) >= 2:
                target = f"Player {parts[-1]}"
            out.append({"accuser": speaker, "target": target, "evidence_type": ev_type, "snippet": content[:80]})
    if not out and ev_type == "soft":
        m = re.search(r"(player \d+)", lower)
        if m and any(kw in lower for kw in ("suspicious", "doubt", "suspect", "watch")):
            target = f"Player {m.group(1).split()[-1]}"
            out.append({"accuser": speaker, "target": target, "evidence_type": "soft", "snippet": content[:80]})
    return out


def extract_round_discussion_signals(history: List[Tuple[str, str]]) -> Dict[str, Any]:
    """Parse discussion speeches/votes into structured signals for scoring."""
    signals: Dict[str, Any] = {
        "accusations": {},
        "claims": [],
        "speakers_order": [],
        "topic_counts": {},
    }
    in_discussion = False
    for speaker, content in history or []:
        cl = str(content).lower()
        if "discussion phase" in cl or "freely talk" in cl:
            in_discussion = True
            continue
        if "voting phase" in cl or ("vote to kill" in cl and "must vote" in cl):
            in_discussion = False
        if not speaker.startswith("Player"):
            continue
        if in_discussion:
            if speaker not in signals["speakers_order"]:
                signals["speakers_order"].append(speaker)
            for topic_key, kws in (
                ("peaceful_night", ("peaceful night", "no one died", "no one dead")),
                ("demand_claims", ("seer", "witch", "guard", "who did you check", "who did you protect")),
                ("defensive_label", ("defensive", "reacted", "deflect")),
            ):
                if any(kw in cl for kw in kws):
                    signals["topic_counts"][topic_key] = signals["topic_counts"].get(topic_key, 0) + 1
            for acc in _extract_accusations_from_message(speaker, content):
                tgt = acc["target"]
                signals["accusations"].setdefault(tgt, []).append(acc)
            if _classify_evidence_in_text(content) == "hard":
                signals["claims"].append({"player": speaker, "content": content[:120]})
        if "i vote to kill" in cl:
            m = re.search(r"i vote to kill (player \d+)", cl)
            if m:
                target = f"Player {m.group(1).split()[-1]}"
                signals["accusations"].setdefault(target, []).append(
                    {"accuser": speaker, "target": target, "evidence_type": "hard", "snippet": "vote"}
                )
    return signals


def _has_unshared_private_info(state, my_role: str, agent_name: str) -> bool:
    history = _get_state_field(state, "history", []) or []
    my_speeches = " ".join(c for s, c in history if s == agent_name).lower()
    for speaker, content in history:
        cl = str(content).lower()
        if my_role == "seer":
            if speaker == "Moderator" or "moderator" in speaker.lower():
                if re.search(r"player \d+ is (?:a werewolf|not a werewolf)", cl):
                    if not any(kw in my_speeches for kw in ("i am the seer", "i verified", "i checked")):
                        return True
        if my_role == "witch":
            if "was attacked" in cl and "will die tonight" in cl:
                if not any(kw in my_speeches for kw in ("i saved", "witch", "antidote")):
                    return True
        if my_role == "guard":
            if speaker == agent_name and "protect" in cl:
                return False
            if "peaceful night" in " ".join(c for _, c in history).lower():
                if not any(kw in my_speeches for kw in ("i protected", "guard")):
                    return True
    return my_role in ("seer", "witch", "guard") and _get_state_field(state, "round_no", 1) >= 2


def _compute_bandwagon_penalty(target: str, report: Dict[str, Any], round_signals: Dict[str, Any]) -> float:
    accs = round_signals.get("accusations", {}).get(target, [])
    if not accs:
        return 0.0
    accusers = {a["accuser"] for a in accs}
    hard_accs = [a for a in accs if a.get("evidence_type") == "hard"]
    if len(accusers) >= 2 and len(hard_accs) == 0:
        return 0.18 + 0.04 * min(len(accusers), 4)
    if report.get("bandwagon_risk"):
        return 0.12
    return 0.0


def _compute_information_gain(
    state,
    my_role: str,
    agent_name: str,
    target: str,
    report: Dict[str, Any],
    game_phase: str,
    round_signals: Dict[str, Any],
) -> float:
    gain = 0.0
    hard = float(report.get("hard_wolf_prob", 0.0))
    soft = float(report.get("soft_wolf_prob", 0.3))
    topic_counts = round_signals.get("topic_counts", {})

    if game_phase == "discussion":
        if _has_unshared_private_info(state, my_role, agent_name):
            gain += 0.42
            if my_role == "witch" and topic_counts.get("peaceful_night", 0) >= 2:
                gain += 0.12
        if target and target != "pass":
            if hard >= 0.45:
                gain += 0.28 + hard * 0.2
            elif soft >= 0.5 and hard < 0.2:
                gain += 0.08
            if topic_counts.get("defensive_label", 0) >= 2 and hard < 0.25:
                gain -= 0.15
        else:
            gain += 0.1
    elif game_phase == "voting":
        if hard >= 0.5:
            gain += 0.35
        elif hard >= 0.3:
            gain += 0.15
        gain += float(report.get("information_gain", 0.0)) * 0.2
        analytics = get_game_analytics(state) if state else {}
        if analytics.get("in_voting_phase"):
            gain += 0.2
        vr = int(report.get("votes_received", 0))
        if vr >= 2:
            gain += 0.1

    return max(0.0, min(1.0, gain))


def enrich_empathy_reports(
    game_state,
    agent_name: str,
    reports: Dict[str, Dict[str, Any]],
    skip_belief_fuse: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Add evidence grading, bandwagon risk, and information-gain estimates."""
    history = _get_state_field(game_state, "history", []) or []
    round_signals = extract_round_discussion_signals(history)
    yet_to_speak = []
    alive = _get_state_field(game_state, "alive_players", []) or []
    spoken = set(round_signals.get("speakers_order", []))
    for p in alive:
        if p not in spoken and p != agent_name and p != "pass":
            yet_to_speak.append(p)

    game_phase = _get_state_field(game_state, "game_phase", "discussion")
    my_role = _get_state_field(game_state, "my_role", "villager")

    player_reports, meta = _split_player_and_meta_reports(reports)
    for player, report in list(player_reports.items()):
        msgs = [content for speaker, content in history if speaker == player]
        hard = float(report.get("hard_wolf_prob", 0.0))
        soft = float(report.get("soft_wolf_prob", report.get("role_probability", {}).get("werewolf", 0.3)))
        semantic_chunks = []
        relation_index = []
        uncertainty_notes = []
        reflection = {
            "what_i_know": "",
            "what_might_be_fake": "",
            "what_conflicts_exist": "",
            "what_to_do_next": "",
        }

        for msg in msgs:
            ev = _classify_evidence_in_text(msg)
            lower = str(msg).lower()
            semantic_chunks.append(str(msg)[:300])

            has_hedge = any(kw in lower for kw in ("maybe", "probably", "perhaps", "i think", "could be", "seems", "might", "not sure", "maybe not"))
            has_probe = any(kw in lower for kw in ("what if", "let's see", "test", "probe", "bait", "challenge", "look closely", "hear from"))
            has_irony = any(kw in lower for kw in ("sure", "yeah right", "as if", "obviously", "interesting", "hmm")) and any(kw in lower for kw in ("not", "but", "yet", "though", "however"))
            has_soft_support = any(kw in lower for kw in ("lean", "tend", "seems good", "looks okay", "could be good", "not suspicious", "for now"))
            has_soft_accuse = any(kw in lower for kw in ("lean wolf", "seems suspicious", "kinda suspicious", "feels off", "not convinced", "unlikely good"))
            has_strong_support = any(kw in lower for kw in ("trust", "good", "verified", "not a werewolf", "safe", "confirmed"))
            has_strong_accuse = any(kw in lower for kw in ("wolf", "werewolf", "vote", "kill", "eliminate", "push"))

            if ev == "hard":
                if has_strong_support:
                    hard = max(0.0, hard - 0.22)
                    relation_index.append("hard_clear")
                elif has_strong_accuse:
                    hard = min(0.95, hard + 0.18)
                    relation_index.append("hard_accuse")
                else:
                    hard = max(0.0, hard - 0.10)
                    uncertainty_notes.append("hard claim without direct verification")
            elif ev == "soft":
                if has_soft_support:
                    soft = max(0.05, soft - 0.05)
                    relation_index.append("soft_support")
                elif has_soft_accuse:
                    soft = min(0.90, soft + 0.08)
                    relation_index.append("soft_accuse")
                else:
                    soft = min(0.85, soft + 0.04)
                    uncertainty_notes.append("soft suspicion")

            if has_hedge:
                relation_index.append("hedge")
                uncertainty_notes.append("hedged language")
                soft = min(0.90, soft + 0.03)
            if has_probe:
                relation_index.append("probe")
                uncertainty_notes.append("probing / baiting style")
            if has_irony:
                relation_index.append("irony_or_tension")
                uncertainty_notes.append("possible irony or tension")

            if has_strong_support:
                relation_index.append("clear_or_soft_support")
            if any(kw in lower for kw in ("i am the seer", "i am seer", "i verified", "i protected", "i saved")):
                relation_index.append("role_or_action_claim")
                hard = max(hard - 0.06, 0.0)
            if has_soft_support:
                relation_index.append("tentative_support")
            if has_soft_accuse:
                relation_index.append("tentative_accuse")
            if has_soft_support and has_soft_accuse:
                relation_index.append("mixed_signal")
                uncertainty_notes.append("mixed support and caution")

        for acc in round_signals.get("accusations", {}).get(player, []):
            if acc.get("evidence_type") == "hard":
                hard = min(0.95, hard + 0.12)
                relation_index.append("hard_accusation_received")
            else:
                soft = min(0.85, soft + 0.06)
                relation_index.append("soft_accusation_received")

        if player in yet_to_speak:
            soft = max(0.05, soft - 0.12)
            uncertainty_notes.append("yet_to_speak_in_round")

        accs = round_signals.get("accusations", {}).get(player, [])
        accusers = {a["accuser"] for a in accs}
        hard_accs = [a for a in accs if a.get("evidence_type") == "hard"]
        bandwagon = len(accusers) >= 2 and len(hard_accs) == 0
        if bandwagon:
            soft *= 0.55
            uncertainty_notes.append("bandwagon_without_hard_evidence")

        combined = min(0.95, max(0.05, hard * 0.72 + soft * 0.28))
        report["hard_wolf_prob"] = round(hard, 3)
        report["soft_wolf_prob"] = round(soft, 3)
        report["role_probability"]["werewolf"] = round(combined, 3)
        report["accusation_count"] = len(accs)
        report["bandwagon_risk"] = bandwagon
        report["evidence_type"] = "hard" if hard >= 0.4 else ("soft" if soft >= 0.45 else "none")
        report["trust_score"] = round(1.0 - combined, 3)
        report["information_gain"] = round(
            _compute_information_gain(game_state, my_role, agent_name, player, report, game_phase, round_signals),
            3,
        )
        if hard >= 0.45:
            report["recommended_action"] = "accuse"
        elif bandwagon and hard < 0.25:
            report["recommended_action"] = "question"
        elif combined < 0.25:
            report["recommended_action"] = "support"
        elif _has_unshared_private_info(game_state, my_role, agent_name) and player == agent_name:
            report["recommended_action"] = "reveal"
        else:
            report["recommended_action"] = report.get("recommended_action", "observe")

        if report.get("private_verified_good"):
            report["hard_wolf_prob"] = 0.0
            report["soft_wolf_prob"] = min(float(report.get("soft_wolf_prob", 0.3)), 0.03)
            report["role_probability"]["werewolf"] = 0.01
            report["trust_score"] = 0.99
            report["evidence_type"] = "hard"
        if report.get("private_verified_wolf"):
            report["hard_wolf_prob"] = 0.99
            report["soft_wolf_prob"] = 0.98
            report["role_probability"]["werewolf"] = 0.99
            report["trust_score"] = 0.01
            report["evidence_type"] = "hard"
        if report.get("claim_type") and not report.get("private_verified_good") and not report.get("private_verified_wolf"):
            report["misdirection_risk"] = max(float(report.get("misdirection_risk", 0.0)), 0.22)

        report["semantic_memory"] = " | ".join(semantic_chunks[-3:])
        report["relation_index"] = list(dict.fromkeys(report.get("relation_index", []) + relation_index))
        report["uncertainty_notes"] = " ; ".join(uncertainty_notes[-4:])
        report["reflection_sketch"] = {
            "what_i_know": f"{player} current combined wolf risk {combined:.2f}",
            "what_might_be_fake": "; ".join([n for n in uncertainty_notes if any(x in n for x in ("fake", "bandwagon", "hedged", "irony", "probing", "mixed"))])[:220],
            "what_conflicts_exist": "; ".join([n for n in uncertainty_notes if any(x in n for x in ("conflict", "hard claim", "mixed", "irony", "probing"))])[:220],
            "what_to_do_next": report.get("recommended_action", "observe"),
        }

        player_reports[player] = report

    reports = {**player_reports, **meta}
    try:
        from .empathy_field import apply_field_to_viewer_reports, get_public_field_from_state
    except ImportError:
        from chatarena.empathy_field import apply_field_to_viewer_reports, get_public_field_from_state

    public_field = get_public_field_from_state(game_state)
    if public_field:
        patched, meta2 = _split_player_and_meta_reports(reports)
        patched = apply_field_to_viewer_reports(public_field, agent_name, patched)
        reports = {**patched, **meta, **meta2}

    analytics = get_game_analytics(game_state)
    patched, meta3 = _split_player_and_meta_reports(reports)
    patched = apply_game_analytics_to_reports(
        patched, analytics, agent_name, game_phase, my_role
    )
    reports = {**patched, **meta3}
    if not skip_belief_fuse:
        try:
            from .belief_state import build_and_attach_belief_state
        except ImportError:
            from chatarena.belief_state import build_and_attach_belief_state
        pr, meta4 = _split_player_and_meta_reports(reports)
        _, synced = build_and_attach_belief_state(
            game_state, agent_name, pr, analytics=analytics
        )
        reports = {**synced, **meta4}

    return reports


def resolve_agent_intent(
    action: Tuple[str, str, str],
    target_report: Dict[str, Any],
    my_role: str,
    game_state,
    agent_name: str,
) -> Dict[str, Any]:
    """Role-conditioned intent resolver: phase-safe and adversarial by role."""
    vote_target, target_player, speech_style = action
    report = target_report or {}
    hard = float(report.get("hard_wolf_prob", 0.0))
    soft = float(report.get("soft_wolf_prob", 0.3))
    combined = float(report.get("role_probability", {}).get("werewolf", 0.3))
    phase = _get_state_field(game_state, "game_phase", "discussion")
    day_night = _get_state_field(game_state, "day_night", "daytime")
    round_no = int(_get_state_field(game_state, "round_no", 1) or 1)

    if day_night == "night":
        if my_role == "witch":
            stance = "protect" if round_no <= 1 or vote_target == "pass" else "poison"
            vote_lean = "pass" if round_no <= 1 else (vote_target if vote_target not in ("pass", agent_name, "") else "pass")
            return {
                "stance": stance,
                "focus": target_player,
                "vote_lean": vote_lean,
                "confidence": 0.99 if round_no <= 1 else 0.9,
                "hard_wolf_prob": hard,
                "soft_wolf_prob": soft,
                "phase": "night",
            }
        if my_role in ("werewolf", "wolf"):
            return {
                "stance": "kill",
                "focus": target_player,
                "vote_lean": vote_target if vote_target not in ("pass", agent_name) else "pass",
                "confidence": 0.9,
                "hard_wolf_prob": hard,
                "soft_wolf_prob": soft,
                "phase": "night",
            }
        if my_role in ("guard", "seer"):
            return {
                "stance": "night_action",
                "focus": target_player,
                "vote_lean": vote_target if vote_target not in ("pass", agent_name) else "pass",
                "confidence": 0.9,
                "hard_wolf_prob": hard,
                "soft_wolf_prob": soft,
                "phase": "night",
            }

    if speech_style == "reveal" and phase == "discussion":
        return {
            "stance": "reveal",
            "focus": "share_private_info",
            "vote_lean": vote_target if vote_target not in ("pass", agent_name, "") else "pass",
            "confidence": 0.75,
            "hard_wolf_prob": hard,
            "soft_wolf_prob": soft,
        }

    if vote_target == "pass" and phase == "discussion" and _has_unshared_private_info(game_state, my_role, agent_name):
        return {
            "stance": "reveal",
            "focus": "share_private_info",
            "vote_lean": "pass",
            "confidence": 0.7,
            "hard_wolf_prob": hard,
            "soft_wolf_prob": soft,
        }

    if my_role in ("werewolf", "wolf"):
        if phase == "discussion":
            stance = "mislead" if speech_style in ("align", "soothe", "ambiguous") else "pressure"
        else:
            stance = "kill"
    elif my_role == "witch":
        stance = "caution" if hard < 0.4 else "support"
    elif speech_style in ("align", "soothe") or combined < 0.28:
        stance = "support"
    elif speech_style in ("evidence", "counter") or hard >= 0.4 or (combined >= 0.52 and hard >= 0.15):
        stance = "suspect"
    elif soft >= 0.5 and hard < 0.2:
        stance = "caution"
    else:
        stance = "neutral"

    vote_lean = vote_target if vote_target not in ("pass", agent_name) else "pass"
    if my_role in ("werewolf", "wolf") and phase == "discussion":
        vote_lean = target_player if target_player not in ("pass", agent_name) else "pass"
    if stance == "support" and vote_lean == target_player:
        vote_lean = "pass"
    if stance in ("suspect", "caution") and vote_lean == "pass" and target_player not in ("pass", agent_name):
        vote_lean = target_player

    return {
        "stance": stance,
        "focus": target_player,
        "vote_lean": vote_lean,
        "confidence": min(0.95, 0.35 + hard * 0.45 + (0.15 if stance == "neutral" else 0.25)),
        "hard_wolf_prob": hard,
        "soft_wolf_prob": soft,
        "phase": phase,
    }


def build_decision_brief(
    intent: Dict[str, Any],
    action: Tuple[str, str, str],
    report: Dict[str, Any],
    my_role: str,
) -> str:
    """Role-conditioned decision summary for LLM (guides without forcing)."""
    _, target_player, speech_style = action
    stance = intent.get("stance", "neutral")
    hard = float(intent.get("hard_wolf_prob", report.get("hard_wolf_prob", 0.0)))
    soft = float(intent.get("soft_wolf_prob", report.get("soft_wolf_prob", 0.3)))
    phase = intent.get("phase", _get_state_field(report.get("_state"), "game_phase", "discussion") if isinstance(report.get("_state"), dict) else "discussion")
    parts = []

    if my_role in ("werewolf", "wolf"):
        parts.append(
            "You are operating in adversarial mode: do not talk like a villager seeking truth. "
            "Favor plausible suspicion, social disruption, and identity concealment."
        )
        if phase == "discussion":
            parts.append("Discussion should redirect attention away from wolves and create disagreement among good players.")
        else:
            parts.append("Night actions should prioritize removing the most dangerous good-information source.")
    elif my_role == "witch":
        if phase == "night":
            parts.append("Night witch logic is resource-preserving: default to pass unless evidence is overwhelming.")
            if stance == "protect":
                parts.append("You are deciding whether to save; prioritize survival and verified urgency.")
            else:
                parts.append("Do not waste poison on weak reads or first-night hunches.")
        else:
            parts.append("As witch, speak cautiously and do not overclaim certainty unless you have strong evidence.")
    elif stance == "reveal":
        parts.append(
            "Your internal read favors contributing private, verifiable information rather than "
            "extending speculative reads."
        )
    elif stance == "suspect" and target_player not in ("pass", ""):
        if hard >= 0.4:
            parts.append(
                f"Hard evidence weights {target_player} as a leading concern (hard={hard:.2f}); "
                f"speech tone: {speech_style}."
            )
        else:
            parts.append(
                f"Behavioral signals raise caution on {target_player}, but proof is limited "
                f"(soft={soft:.2f}, hard={hard:.2f}); keep reasoning proportional."
            )
    elif stance == "support" and target_player not in ("pass", ""):
        parts.append(f"Relative trust is higher toward {target_player}; avoid overstating suspicion.")
    elif stance == "caution":
        parts.append(
            f"Soft reads on {target_player} merit watchful curiosity rather than a strong push "
            f"(bandwagon risk noted)."
        )
    else:
        parts.append("The board still lacks decisive claims; probing for verifiable info may help most.")

    vote_lean = intent.get("vote_lean", "pass")
    if vote_lean and vote_lean not in ("pass", ""):
        if stance == "support" and vote_lean == target_player:
            pass
        else:
            parts.append(f"Your current integrated view inclines toward {vote_lean} if the village votes today.")

    return " ".join(parts)


def build_empathy_reports(
    game_state,
    agent_name: str,
    empathy_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build / refine per-player empathy reports from state + optional prior data."""
    alive = _get_state_field(game_state, "alive_players", []) or []
    history = _get_state_field(game_state, "history", []) or []
    reports: Dict[str, Dict[str, Any]] = {}

    # map private claims about protection/heal/verify to support relations
    support_relations = []
    for speaker, content in history:
        cl = str(content).lower()
        if not speaker.startswith("Player"):
            continue
        m = re.search(r"(?:i saved|i protected|i verified|i checked|i saw|i am the witch|i am the guard|i am the seer).{0,40}?(player\s*\d+)", cl)
        if m:
            tgt = _norm_player_id(m.group(1))
            support_relations.append((speaker, tgt, cl))
        m2 = re.search(r"(?:player\s*\d+).{0,40}?(?:is not a werewolf|is good|seems good|safe)", cl)
        if m2:
            # if speaker references a player as good, treat that as soft support of target
            tgt = _norm_player_id(m2.group(0))
            support_relations.append((speaker, tgt, cl))

    for player in alive:
        if player == agent_name or player == "pass":
            continue
        base = _default_player_report()
        if empathy_data and player in empathy_data:
            incoming = empathy_data[player]
            if isinstance(incoming, dict):
                for k, v in incoming.items():
                    if isinstance(v, dict) and isinstance(base.get(k), dict):
                        base[k].update(v)
                    else:
                        base[k] = v

        msgs = [content for speaker, content in history if speaker == player]
        suspicion = base["role_probability"].get("werewolf", 0.3)
        stance = base["stance_to_me"]
        arousal = base["emotion"].get("arousal", 0.5)
        support_targets = []
        claim_type = ""
        claim_target = ""
        claim_strength = 0.0
        misdirection_risk = 0.0
        info_gain = float(base.get("information_gain", 0.0))
        public_trust = float(base.get("public_trust", 0.0))
        vote_pressure = float(base.get("current_round_vote_pressure", 0.0))
        speech_vote_align = float(base.get("speech_vote_consistency", 1.0))

        for msg in msgs:
            lower = str(msg).lower()
            ev = _classify_evidence_in_text(msg)
            has_question = "?" in lower
            has_claim = any(kw in lower for kw in ("i am", "i verified", "i checked", "i protected", "i saved", "i suspect", "i think"))
            has_action = any(kw in lower for kw in ("vote", "kill", "protect", "verify", "save", "poison", "check"))
            has_hard_info = any(kw in lower for kw in ("verified", "not a werewolf", "is a werewolf", "saved", "protected", "poisoned"))
            has_soft_info = any(kw in lower for kw in ("maybe", "probably", "seems", "looks", "might", "could"))
            has_pressure = any(kw in lower for kw in ("must", "should", "need", "have to", "let's", "please"))
            has_support = any(kw in lower for kw in ("trust", "good", "safe", "not suspicious", "support", "agree"))
            has_accuse = any(kw in lower for kw in ("suspicious", "wolf", "werewolf", "doubt", "vote against", "eliminate"))

            if ev == "hard" and has_hard_info:
                suspicion += 0.22
                info_gain += 0.18
                claim_type = claim_type or "hard_claim"
                claim_strength = max(claim_strength, 0.7)
                public_trust += 0.1
            elif ev == "soft" and has_soft_info:
                suspicion += 0.06
                info_gain += 0.05
            elif has_action:
                info_gain += 0.04

            if has_support:
                suspicion -= 0.08
                stance += 0.08
                public_trust += 0.08
                info_gain += 0.03
            if has_accuse:
                suspicion += 0.05
                stance -= 0.06
                info_gain += 0.03
            if has_pressure:
                arousal += 0.04
                vote_pressure += 0.03
            if has_question:
                info_gain += 0.02
            if has_claim:
                claim_type = claim_type or "special_role_claim"
                claim_strength = max(claim_strength, 0.45)
                info_gain += 0.06
            if any(kw in lower for kw in ("i am the seer", "i am seer", "i verified", "i protected", "i saved")):
                suspicion -= 0.1
                claim_type = "special_role_claim"
                claim_strength = max(claim_strength, 0.55)
                if any(kw in lower for kw in ("protected", "saved", "verified", "checked")):
                    m = re.search(r"(player\s*\d+)", lower)
                    if m:
                        claim_target = _norm_player_id(m.group(1))
            if agent_name.lower() in lower and any(kw in lower for kw in ("agree", "support", "trust")):
                stance += 0.1
            if agent_name.lower() in lower and any(kw in lower for kw in ("suspicious", "vote", "kill")):
                stance -= 0.08
            if len(msg) > 180:
                arousal += 0.02
                info_gain += 0.01
            if len(msg) > 40:
                info_gain += 0.01

        if msgs:
            diversity = len({w for m in msgs for w in re.findall(r"[a-z']+", str(m).lower())})
            if diversity >= 25:
                info_gain += 0.05
            if any("?" in str(m) for m in msgs):
                info_gain += 0.02
            if any(kw in " ".join(msgs).lower() for kw in ("because", "therefore", "so", "since")):
                info_gain += 0.04
            if any(kw in " ".join(msgs).lower() for kw in ("first", "last night", "yesterday", "tonight")):
                info_gain += 0.03

        # if player claims to have saved/protected someone, lower suspicion on target,
        # but keep claimant's misdirection risk alive
        for speaker, tgt, cl in support_relations:
            if speaker == player and tgt in alive:
                support_targets.append(tgt)
                if tgt == player:
                    continue
                # target becomes less suspicious, claimant retains risk if claim lacks external confirmation
                reports.setdefault(tgt, _default_player_report())
                reports[tgt]["role_probability"]["werewolf"] = max(
                    0.02, reports[tgt]["role_probability"].get("werewolf", 0.3) - 0.18
                )
                reports[tgt]["trust_score"] = round(1.0 - reports[tgt]["role_probability"]["werewolf"], 3)
                reports[tgt]["support_targets"] = list(dict.fromkeys(reports[tgt].get("support_targets", []) + [speaker]))
                reports[tgt]["relation_index"] = list(dict.fromkeys(reports[tgt].get("relation_index", []) + ["supported_by_claim"]))
                if speaker == player:
                    base.setdefault("relation_index", [])
                    base["relation_index"] = list(dict.fromkeys(base.get("relation_index", []) + ["self_claim_support", "claim_may_be_deceptive"]))
                    base.setdefault("support_uncertainty", 0.5)
                    base["support_uncertainty"] = min(1.0, float(base.get("support_uncertainty", 0.5)) + 0.25)
                    base.setdefault("uncertainty_notes", "")
                    extra = "self-claim support may be deceptive"
                    base["uncertainty_notes"] = (base["uncertainty_notes"] + "; " + extra).strip("; ") if base["uncertainty_notes"] else extra
                # claimant may be bluffing; reduce only slightly and add misdirection risk
                claim_strength = max(claim_strength, 0.55)
                misdirection_risk = max(misdirection_risk, 0.25)

        if support_targets:
            public_trust += 0.04 * len(support_targets)
            info_gain += 0.02 * len(support_targets)
        if vote_pressure >= 0.3:
            info_gain += 0.05
        if speech_vote_align < 0.65:
            misdirection_risk += 0.08
            suspicion += 0.04
        if public_trust > 0.5:
            suspicion -= 0.08
        if info_gain < 0.08 and msgs:
            info_gain = 0.08 + 0.01 * min(len(msgs), 5)

        base["role_probability"]["werewolf"] = max(0.05, min(0.95, suspicion))
        base["stance_to_me"] = max(-1.0, min(1.0, stance))
        base["emotion"]["arousal"] = max(0.0, min(1.0, arousal))
        base["trust_score"] = 1.0 - base["role_probability"]["werewolf"]
        base["support_targets"] = list(dict.fromkeys(base.get("support_targets", []) + support_targets))
        base["claim_type"] = claim_type
        base["claim_target"] = claim_target
        base["claim_strength"] = round(claim_strength, 3)
        base["misdirection_risk"] = round(misdirection_risk, 3)
        base["public_trust"] = round(max(0.0, min(1.0, public_trust)), 3)
        base["information_gain"] = round(max(0.0, min(1.0, info_gain)), 3)
        reports[player] = base

    enriched = enrich_empathy_reports(game_state, agent_name, reports)
    # Add a compact board-level snapshot so downstream prompts can reason over the whole table,
    # not only a single target player.
    try:
        player_view = {
            p: enriched[p] for p in enriched.keys() if _is_player_report_key(p)
        }
        top_suspects = sorted(
            player_view.items(),
            key=lambda kv: (
                -float(kv[1].get("hard_wolf_prob", 0.0)),
                -float(kv[1].get("information_gain", 0.0)),
                bool(kv[1].get("bandwagon_risk", False)),
            ),
        )[:3]
        top_support = sorted(
            player_view.items(),
            key=lambda kv: (
                -float(kv[1].get("public_trust", 0.0)),
                -float(kv[1].get("trust_score", 0.0)),
            ),
        )[:3]
        enriched["_game"] = {
            "agent_name": agent_name,
            "round_no": int(_get_state_field(game_state, "round_no", 1) or 1),
            "game_phase": _get_state_field(game_state, "game_phase", "discussion"),
            "day_night": _get_state_field(game_state, "day_night", "daytime"),
            "top_suspects": [
                {
                    "player": p,
                    "hard_wolf_prob": round(float(r.get("hard_wolf_prob", 0.0)), 3),
                    "soft_wolf_prob": round(float(r.get("soft_wolf_prob", 0.0)), 3),
                    "information_gain": round(float(r.get("information_gain", 0.0)), 3),
                    "bandwagon_risk": bool(r.get("bandwagon_risk", False)),
                    "recommended_action": r.get("recommended_action", "observe"),
                }
                for p, r in top_suspects
            ],
            "top_trust": [
                {
                    "player": p,
                    "public_trust": round(float(r.get("public_trust", 0.0)), 3),
                    "trust_score": round(float(r.get("trust_score", 0.0)), 3),
                    "support_summary": str(r.get("support_summary", ""))[:120],
                }
                for p, r in top_support
            ],
            "board_signal": {
                "info_dense": sum(1 for _, r in player_view.items() if float(r.get("information_gain", 0.0)) >= 0.15),
                "hard_claims": sum(1 for _, r in player_view.items() if str(r.get("evidence_type", "")) == "hard"),
                "support_links": sum(len(r.get("supports", []) or []) for _, r in player_view.items()),
                "accusation_links": sum(int(r.get("accusation_count", 0)) for _, r in player_view.items()),
            },
        }
    except Exception:
        pass
    return enriched


def merge_empathy_reports(
    llm_reports: Optional[Dict[str, Any]],
    rule_reports: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Merge LLM empathy output with rule-based reports (LLM primary, rules fill gaps)."""
    merged: Dict[str, Dict[str, Any]] = {}
    all_players = {
        p
        for p in set((llm_reports or {}).keys()) | set((rule_reports or {}).keys())
        if _is_player_report_key(p)
    }
    for player in all_players:
        base = _default_player_report()
        rule = (rule_reports or {}).get(player, {})
        llm = (llm_reports or {}).get(player, {})
        for src in (rule, llm):
            if not isinstance(src, dict):
                continue
            for k, v in src.items():
                if k == "role_probability" and not isinstance(v, dict):
                    if isinstance(v, (int, float)):
                        base["role_probability"]["werewolf"] = float(v)
                    continue
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    base[k].update(v)
                else:
                    base[k] = v
        base = _ensure_report_schema(base)
        merged[player] = base
    return merged


def parse_empathy_json_response(raw: str) -> Dict[str, Dict[str, Any]]:
    """
    Robustly parse LLM empathy JSON.
    Handles markdown fences, EOS suffix, truncation, and compact schemas.
    """
    if not raw:
        return {}

    text = str(raw).strip()
    text = re.sub(r"<EOS>.*", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)

    def _normalize_report(player: str, report: Dict[str, Any]) -> Dict[str, Any]:
        base = _default_player_report()
        if not isinstance(report, dict):
            return base
        # Compact schema aliases
        if "werewolf_prob" in report and "role_probability" not in report:
            wp = float(report.get("werewolf_prob", 0.3))
            base["role_probability"]["werewolf"] = wp
            base["role_probability"]["villager"] = max(0.0, 1.0 - wp - 0.2)
        if "hard_wolf_prob" in report:
            base["hard_wolf_prob"] = float(report["hard_wolf_prob"])
        if "soft_wolf_prob" in report:
            base["soft_wolf_prob"] = float(report["soft_wolf_prob"])
        if "evidence_type" in report:
            base["evidence_type"] = str(report["evidence_type"])
        if "trust" in report:
            base["trust_score"] = float(report["trust"])
        if "stance" in report:
            base["stance_to_me"] = float(report["stance"])
        if "speech_strategy" in report:
            base["speech_acts_recent"] = [str(report["speech_strategy"])]
        if "semantic_memory" in report:
            base["semantic_memory"] = str(report.get("semantic_memory", ""))
        if "uncertainty_notes" in report:
            base["uncertainty_notes"] = str(report.get("uncertainty_notes", ""))
        if "reflection_sketch" in report and isinstance(report.get("reflection_sketch"), dict):
            base["reflection_sketch"].update(report.get("reflection_sketch", {}))
        rel_index = report.get("relation_index")
        if isinstance(rel_index, list):
            base["relation_index"] = [str(x) for x in rel_index[:8]]
        for k, v in report.items():
            if k in ("werewolf_prob", "trust", "stance", "speech_strategy", "hard_wolf_prob", "soft_wolf_prob", "evidence_type", "semantic_memory", "uncertainty_notes", "reflection_sketch", "relation_index"):
                continue
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k].update(v)
            else:
                base[k] = v
        if "trust_score" not in base or base.get("trust_score") == 0.5:
            base["trust_score"] = 1.0 - base.get("role_probability", {}).get("werewolf", 0.3)
        if "claim_type" not in base:
            base["claim_type"] = ""
        if "claim_target" not in base:
            base["claim_target"] = ""
        if "claim_strength" not in base:
            base["claim_strength"] = 0.0
        if "misdirection_risk" not in base:
            base["misdirection_risk"] = 0.0
        base["recommended_action"] = report.get(
            "recommended_action",
            "accuse" if base.get("hard_wolf_prob", 0) > 0.4 else (
                "question" if base["role_probability"].get("werewolf", 0.3) > 0.5 else "observe"
            ),
        )
        if base.get("hard_wolf_prob", 0) >= 0.35:
            base["evidence_type"] = "hard"
        elif base.get("soft_wolf_prob", 0) >= 0.45:
            base["evidence_type"] = "soft"
        return base

    def _extract_from_obj(obj) -> Dict[str, Dict[str, Any]]:
        if not isinstance(obj, dict):
            return {}
        reports = obj.get("player_reports", obj)
        if not isinstance(reports, dict):
            return {}
        out = {}
        for player, report in reports.items():
            if not str(player).startswith("Player"):
                continue
            out[player] = _normalize_report(player, report)
        return out

    # 1) Direct parse
    for candidate in (text,):
        try:
            parsed = _extract_from_obj(json.loads(candidate))
            if parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    # 2) Extract outer JSON object
    json_match = re.search(r"\{.*", text, re.DOTALL)
    if json_match:
        blob = json_match.group()
        for suffix in ("", "}" , "}}", "}}}"):
            try:
                parsed = _extract_from_obj(json.loads(blob + suffix))
                if parsed:
                    return parsed
            except json.JSONDecodeError:
                continue

    # 3) Per-player block extraction (truncated output)
    out: Dict[str, Dict[str, Any]] = {}
    for player, block in re.findall(
        r'"(Player \d+)"\s*:\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',
        text,
    ):
        try:
            report = json.loads(block)
            out[player] = _normalize_report(player, report)
        except json.JSONDecodeError:
            continue

    return out


def format_empathy_for_speech(empathy_data: Dict[str, Dict[str, Any]], target_player: str = "") -> str:
    """Format empathy reports into concise speech-strategy hints."""
    if not empathy_data:
        return ""
    reports = empathy_data.get("player_reports", empathy_data) if isinstance(empathy_data, dict) else empathy_data
    lines = []
    items = [(k, v) for k, v in reports.items() if k != "_game"]
    if target_player and target_player in reports:
        items = [(target_player, reports[target_player])]
    for player, report in list(items)[:4]:
        hard = report.get("hard_wolf_prob", 0.0)
        soft = report.get("soft_wolf_prob", report.get("role_probability", {}).get("werewolf", 0.3))
        ev = report.get("evidence_type", "none")
        ig = report.get("information_gain", 0.0)
        strategy = report.get("recommended_action") or (
            "accuse" if hard > 0.4 else ("question" if soft > 0.5 and hard < 0.2 else "observe")
        )
        bw = " bandwagon_risk" if report.get("bandwagon_risk") else ""
        vp = report.get("current_round_vote_pressure", 0.0)
        vr = report.get("votes_received", 0)
        svc = report.get("speech_vote_consistency", 1.0)
        post = report.get("posterior_wolf_prob", report.get("role_probability", {}).get("werewolf", 0.3))
        lines.append(
            f"- {player}: post={post:.2f} hard={hard:.2f} soft={soft:.2f} ev={ev} info_gain={ig:.2f} "
            f"action={strategy}{bw} vote_pressure={vp:.2f} votes_recv={vr} speech_vote_align={svc:.2f}"
        )
    return "\n".join(lines)


def llm_empathy_extract(
    observation: Union[str, Any] = "",
    player_name: str = "",
    backend=None,
    agent_name: str = None,
    game_state=None,
    args=None,
    msgs=None,
    ques=None,
    empathy_data: Optional[Dict] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Rule-based empathy fallback (no LLM).
    Production path: openai.py calls _internal_empathy_extract (LLM) + merge_empathy_reports.
    Returns {"player_reports": {...}} for compatibility.
    """
    agent = agent_name or player_name or "unknown"
    state = game_state if game_state is not None else observation
    reports = build_empathy_reports(state, agent, empathy_data)
    logger.info(f"[Empathy] Rule-based reports for {agent}: {len(reports)} players")
    return {"player_reports": reports}


def _generate_empathy_based_speech(
    action: Tuple[str, str, str],
    target_empathy: Dict[str, Any],
    game_state,
) -> Dict[str, Any]:
    """Map empathy analysis to speech guidance (no LLM)."""
    vote_target, target_player, speech_style = action
    report = target_empathy or {}
    role_probs = report.get("role_probability", {})
    werewolf_prob = role_probs.get("werewolf", 0.3)
    emotion = report.get("emotion", {})
    arousal = emotion.get("arousal", 0.5)
    stance = report.get("stance_to_me", 0.0)

    emotional_approach = "analytical"
    tone = "neutral"
    confidence_level = "moderate"
    agreement_level = "moderate"
    opposition_strength = "moderate"
    revelation_level = "partial"

    my_role = _get_state_field(game_state, "my_role", "villager")
    phase = _get_state_field(game_state, "game_phase", "discussion")
    day_night = _get_state_field(game_state, "day_night", "daytime")
    round_no = int(_get_state_field(game_state, "round_no", 1) or 1)

    # Phase-first semantics: night is action-only, discussion is speech-only.
    if day_night == "night" or phase in ("voting",):
        emotional_approach = "action"
        tone = "decisive"
        revelation_level = "none"
    elif werewolf_prob > 0.55:
        emotional_approach = "confrontational" if stance < 0 else "analytical"
        tone = "suspicious"
        confidence_level = "high" if werewolf_prob > 0.7 else "moderate"
    elif werewolf_prob < 0.25:
        emotional_approach = "supportive"
        tone = "trusting"
        agreement_level = "strong" if stance > 0.2 else "moderate"
    elif arousal > 0.7:
        emotional_approach = "soothing"
        tone = "calm"
    elif stance < -0.3:
        emotional_approach = "counter"
        opposition_strength = "strong"
    elif stance > 0.3:
        emotional_approach = "align"
        agreement_level = "strong"

    hard = float(report.get("hard_wolf_prob", 0.0))
    soft = float(report.get("soft_wolf_prob", werewolf_prob))
    intent = resolve_agent_intent(
        (vote_target, target_player, speech_style), report, my_role, game_state,
        _get_state_field(game_state, "current_player", ""),
    )

    if phase == "discussion" and (intent.get("stance") == "reveal" or speech_style == "reveal"):
        # Reveal is discussion-only; it must never become an action cue at night.
        emotional_approach = "revelatory"
        revelation_level = "full" if _has_unshared_private_info(game_state, my_role, _get_state_field(game_state, "current_player", "")) else "partial"
    elif my_role in ("seer", "guard", "witch") and phase == "discussion" and round_no >= 2:
        if hard >= 0.4:
            emotional_approach = "evidence"
            revelation_level = "partial"
        else:
            emotional_approach = "revelatory"
            revelation_level = "partial"
    elif my_role == "werewolf":
        emotional_approach = "strategic"
        tone = "calculating"
        revelation_level = "none"

    return {
        "target_player": target_player,
        "speech_style": speech_style,
        "target_role_probability": role_probs,
        "target_emotion": emotion,
        "stance_to_me": stance,
        "emotional_approach": emotional_approach,
        "tone": tone,
        "confidence_level": confidence_level,
        "agreement_level": agreement_level,
        "opposition_strength": opposition_strength,
        "revelation_level": revelation_level,
        "werewolf_probability": werewolf_prob,
        "hard_wolf_prob": hard,
        "soft_wolf_prob": soft,
        "agent_intent": intent,
        "decision_brief": build_decision_brief(intent, (vote_target, target_player, speech_style), report, my_role),
        "vote_lean": intent.get("vote_lean", vote_target),
    }


def _select_speech_style(target_report: Dict[str, Any], my_role: str, game_state=None, agent_name: str = "") -> str:
    if not target_report:
        return "ambiguous"
    hard = float(target_report.get("hard_wolf_prob", 0.0))
    werewolf_prob = target_report.get("role_probability", {}).get("werewolf", 0.3)
    soft = float(target_report.get("soft_wolf_prob", werewolf_prob))
    stance = target_report.get("stance_to_me", 0.0)

    if game_state and _has_unshared_private_info(game_state, my_role, agent_name):
        if hard < 0.35 and soft < 0.5:
            return "reveal"
    if my_role in ("seer", "guard", "witch") and hard > 0.45:
        return "evidence"
    if hard > 0.5:
        return "evidence"
    if werewolf_prob > 0.55 and hard >= 0.2:
        return "evidence"
    if werewolf_prob > 0.55 and hard < 0.2:
        return "ambiguous"
    if werewolf_prob < 0.25 and stance > 0.2:
        return "align"
    if stance < -0.3:
        return "counter"
    if stance > 0.3:
        return "align"
    if target_report.get("bandwagon_risk"):
        return "question"
    return "ambiguous"


def _score_reveal_action(state, my_role: str, agent_name: str, game_phase: str) -> float:
    if game_phase != "discussion" or not _has_unshared_private_info(state, my_role, agent_name):
        return 0.0
    score = 0.38
    round_no = _get_state_field(state, "round_no", 1)
    history_text = " ".join(c for _, c in _get_state_field(state, "history", []) or []).lower()
    if my_role == "witch" and "peaceful night" in history_text:
        score += 0.22
    if my_role == "seer":
        score += 0.18
    if my_role == "guard":
        score += 0.12
    if round_no >= 2:
        score += 0.08
    return min(1.0, score)


# ============ Deep MCTS (Step 3): intent layer + rollout ============

DISCUSSION_INTENTS = ("reveal", "press", "accuse", "support", "probe")
NIGHT_INTENTS = ("vote",)
WITCH_NIGHT_INTENTS = ("save", "poison")


def _phase_allowed_intents(my_role: str, day_night: str, game_phase: str) -> Tuple[str, ...]:
    """Hard phase boundary: only legal intent families can enter MCTS."""
    if game_phase == "voting":
        return NIGHT_INTENTS
    if day_night == "night":
        if my_role == "witch":
            return WITCH_NIGHT_INTENTS
        return NIGHT_INTENTS
    return DISCUSSION_INTENTS
NIGHT_ROLE_ACTIONS = {
    "werewolf": "kill",
    "wolf": "kill",
    "guard": "protect",
    "seer": "verify",
    "witch": "witch",
}


def _empathy_field_imports():
    try:
        from .empathy_field import (
            get_public_field_from_state,
            PublicEmpathyField,
            UtteranceEffectModel,
        )
    except ImportError:
        from chatarena.empathy_field import (
            get_public_field_from_state,
            PublicEmpathyField,
            UtteranceEffectModel,
        )
    return get_public_field_from_state, PublicEmpathyField, UtteranceEffectModel


def _intent_to_stance(intent: str) -> str:
    return {
        "reveal": "reveal",
        "support": "support",
        "accuse": "suspect",
        "probe": "caution",
        "press": "neutral",
        "vote": "suspect",
    }.get(intent, "neutral")


def _intent_to_speech_style(
    intent: str,
    target_report: Dict[str, Any],
    my_role: str,
    state=None,
    agent_name: str = "",
) -> str:
    report = target_report or {}
    if intent == "reveal":
        if _get_state_field(state, "day_night", "daytime") == "night":
            return "neutral"
        return "reveal"
    if intent == "press":
        return "demand_info"
    if intent == "support":
        return "align"
    if intent == "accuse":
        return _select_speech_style(report, my_role, state, agent_name)
    if intent == "probe":
        if report.get("bandwagon_risk"):
            return "ambiguous"
        return _select_speech_style(report, my_role, state, agent_name)
    return "ambiguous"


def _field_entropy(field) -> float:
    """Public-board uncertainty (higher = more open / less resolved)."""
    if field is None:
        return 0.62
    acc_n = len(getattr(field, "accusations", []) or [])
    trust_n = len(getattr(field, "trust_edges", {}) or {})
    claims_n = len(getattr(field, "claims", []) or [])
    role_n = len(getattr(field, "role_claims", {}) or {})
    raw = 0.38 + 0.06 * acc_n + 0.04 * trust_n + 0.05 * claims_n + 0.08 * role_n
    return min(1.0, max(0.2, raw))


def _build_rollout_plan(
    intent: str,
    target: str,
    agent_name: str,
    my_role: str,
    speech_style: str,
) -> Dict[str, Any]:
    stance = _intent_to_stance(intent)
    plan: Dict[str, Any] = {
        "intent": intent,
        "target": target if target not in ("pass", "") else agent_name,
        "stance": stance,
        "speech_style": speech_style,
        "claims": [],
    }
    # Rollout plans are discussion-only; night actions must never inherit reveal/support logic.
    if intent == "reveal" and my_role in ("seer", "witch", "guard"):
        plan["claims"].append({
            "speaker": agent_name,
            "claim_type": f"{my_role}_pending",
            "target": plan["target"],
            "detail": speech_style,
            "evidence_type": "hard",
        })
    elif intent == "accuse" and target.startswith("Player"):
        plan["claims"].append({
            "speaker": agent_name,
            "claim_type": "accusation",
            "target": target,
            "detail": "mcts_rollout",
            "evidence_type": "soft",
        })
    elif intent == "support" and target.startswith("Player"):
        plan["claims"].append({
            "speaker": agent_name,
            "claim_type": "trust_signal",
            "target": target,
            "detail": "support",
            "evidence_type": "soft",
        })
    return plan


def _simulate_rollout_information_gain(
    state,
    agent_name: str,
    intent: str,
    target_player: str,
    my_role: str,
    speech_style: str,
) -> float:
    """Δ(public field entropy) after hypothetical utterance (Step 3 rollout)."""
    get_field, _, UtteranceEffectModel = _empathy_field_imports()
    field = get_field(state)
    if field is None:
        return 0.0
    tgt = target_player if target_player not in ("pass", "") else agent_name
    try:
        h0 = _field_entropy(field)
        plan = _build_rollout_plan(intent, tgt, agent_name, my_role, speech_style)
        new_field = UtteranceEffectModel.simulate_propagate(field, agent_name, plan)
        h1 = _field_entropy(new_field)
        return max(0.0, min(0.45, h0 - h1))
    except Exception as exc:
        logger.debug(f"[DeepMCTS] rollout skipped: {exc}")
        return 0.0


def _legal_discussion_intents(
    state,
    my_role: str,
    agent_name: str,
    game_phase: str,
) -> List[str]:
    intents = ["probe", "support", "accuse", "press"]
    reveal_score = _score_reveal_action(state, my_role, agent_name, game_phase)
    if my_role in ("seer", "guard", "witch") and reveal_score > 0.36:
        intents.insert(0, "reveal")
    if my_role in ("werewolf", "wolf"):
        intents = [i for i in intents if i != "reveal"]
        if "accuse" not in intents:
            intents.insert(0, "accuse")
    return intents


def _targets_for_intent(
    intent: str,
    legal_actions: List[str],
    empathy_reports: Dict[str, Dict[str, Any]],
    agent_name: str,
    max_targets: int = 3,
) -> List[str]:
    players = [p for p in legal_actions if p != agent_name and p != "pass"]
    if not players:
        return []

    def _wolf_prob(p: str) -> float:
        r = empathy_reports.get(p, {})
        return float(r.get("role_probability", {}).get("werewolf", 0.3))

    def _hard(p: str) -> float:
        return float(empathy_reports.get(p, {}).get("hard_wolf_prob", 0.0))

    def _trust_pub(p: str) -> float:
        return float(empathy_reports.get(p, {}).get("public_trust", 0.0))

    if intent == "support":
        ranked = sorted(players, key=lambda p: (_wolf_prob(p), -_trust_pub(p)))
    elif intent == "accuse":
        ranked = sorted(players, key=lambda p: (-_hard(p), -_wolf_prob(p)))
    else:
        ranked = sorted(
            players,
            key=lambda p: (-_wolf_prob(p), empathy_reports.get(p, {}).get("bandwagon_risk", False)),
        )
    return ranked[:max_targets]


def _expand_deep_action_candidates(
    legal_actions: List[str],
    state,
    player: str,
    my_role: str,
    game_phase: str,
    day_night: str,
    empathy_reports: Dict[str, Dict[str, Any]],
) -> List[Tuple[str, str, str, Dict[str, Any]]]:
    """Phase-first action bundles: discussion/vote/night are separate worlds."""
    out: List[Tuple[str, str, str, Dict[str, Any]]] = []
    seen: set = set()
    allowed_intents = _phase_allowed_intents(my_role, day_night, game_phase)

    if day_night == "night":
        if my_role in ("witch",):
            round_no = int(_get_state_field(state, "round_no", 1) or 1)
            if round_no <= 1:
                return [("pass", "pass", "neutral", {"intent": "pass", "layer": 0, "phase": "night"})]
            for a in ["pass"] + [x for x in legal_actions if re.match(r"^Player\s+\d+$", str(x).strip(), re.I)]:
                if a != "pass" and "poison" not in allowed_intents:
                    continue
                key = (a, a, "neutral")
                if key in seen:
                    continue
                seen.add(key)
                out.append((a, a, "neutral", {"intent": "witch_poison" if a != "pass" else "pass", "layer": 2, "phase": "night"}))
            return out
        role_action = NIGHT_ROLE_ACTIONS.get(my_role, "pass")
        if "vote" not in allowed_intents:
            role_action = "pass"
        for a in ["pass"] + [x for x in legal_actions if re.match(r"^Player\s+\d+$", str(x).strip(), re.I)]:
            key = (a, a, "neutral")
            if key in seen:
                continue
            seen.add(key)
            out.append((a, a, "neutral", {"intent": role_action, "layer": 2, "phase": "night"}))
        return out

    if game_phase == "voting":
        for t in [x for x in legal_actions if x != "pass"]:
            key = (t, t, "neutral")
            if key in seen:
                continue
            seen.add(key)
            out.append((t, t, "neutral", {"intent": "vote", "layer": 2, "phase": "voting"}))
        return out

    # discussion only
    for intent in _legal_discussion_intents(state, my_role, player, game_phase):
        if intent not in allowed_intents:
            continue
        if intent == "reveal":
            key = ("pass", player, "reveal")
            if key not in seen:
                seen.add(key)
                out.append(("pass", player, "reveal", {"intent": "reveal", "layer": 1, "phase": "discussion"}))
            continue
        if intent == "press":
            key = ("pass", player, "demand_info")
            if key not in seen:
                seen.add(key)
                out.append(("pass", player, "demand_info", {"intent": "press", "layer": 1, "phase": "discussion"}))
            continue
        for tgt in _targets_for_intent(intent, legal_actions, empathy_reports, player):
            style = _intent_to_speech_style(intent, empathy_reports.get(tgt, {}), my_role, state, player)
            vote_target, target_player = ("pass", tgt) if intent != "accuse" else (tgt, tgt)
            key = (vote_target, target_player, style)
            if key in seen:
                continue
            seen.add(key)
            out.append((vote_target, target_player, style, {"intent": intent, "layer": 2, "phase": "discussion"}))
    return out


def _patch_empathy_context_with_intent(
    empathy_context: Dict[str, Any],
    intent_meta: Dict[str, Any],
) -> Dict[str, Any]:
    if not empathy_context:
        empathy_context = {}
    intent = (intent_meta or {}).get("intent", "")
    if not intent:
        return empathy_context
    empathy_context = dict(empathy_context)
    empathy_context["mcts_intent"] = intent
    ai = dict(empathy_context.get("agent_intent") or {})
    ai["stance"] = _intent_to_stance(intent)
    ai["mcts_intent"] = intent
    empathy_context["agent_intent"] = ai
    return empathy_context


def _score_discussion_utility(
    intent: str,
    vote_target: str,
    target_player: str,
    speech_style: str,
    my_role: str,
    day_night: str,
    game_phase: str,
    target_report: Dict[str, Any],
    state=None,
    agent_name: str = "",
    round_signals: Optional[Dict[str, Any]] = None,
    empathy_reports: Optional[Dict[str, Dict[str, Any]]] = None,
) -> float:
    """U_disc: discussion-phase utility with rollout information gain."""
    score_target = target_player if target_player not in ("pass", "") else vote_target
    if intent == "reveal":
        score_target = "pass"
    base = _score_action_for_role(
        score_target,
        my_role,
        day_night,
        game_phase,
        target_report,
        state=state,
        agent_name=agent_name,
        round_signals=round_signals,
    )
    intent_prior = {
        "reveal": _score_reveal_action(state, my_role, agent_name, game_phase) if state else 0.0,
        "press": 0.44,
        "accuse": 0.46,
        "support": 0.42,
        "probe": 0.4,
    }.get(intent, 0.36)

    rollout_gain = _simulate_rollout_information_gain(
        state, agent_name, intent, target_player, my_role, speech_style
    )

    report = target_report or {}
    if intent == "reveal":
        combined = max(base, intent_prior)
    elif intent == "support" and target_player.startswith("Player"):
        combined = base + 0.18 * float(report.get("public_trust", report.get("trust_score", 0.5)))
        if report.get("bandwagon_risk"):
            combined -= 0.12
    elif intent == "accuse" and target_player.startswith("Player"):
        combined = base + 0.20 * float(report.get("hard_wolf_prob", 0.0))
        if report.get("bandwagon_risk") and float(report.get("hard_wolf_prob", 0)) < 0.2:
            combined -= 0.18
    elif intent == "probe":
        combined = base + 0.08 * float(report.get("information_gain", 0.0))
    else:
        combined = 0.55 * base + 0.45 * intent_prior

    utility = 0.42 * combined + 0.33 * intent_prior + 0.25 * (0.5 + rollout_gain)
    return max(0.0, min(1.0, utility + random.uniform(-0.03, 0.03)))


def _score_vote_utility(
    target: str,
    my_role: str,
    day_night: str,
    target_report: Dict[str, Any],
    state=None,
    agent_name: str = "",
    round_signals: Optional[Dict[str, Any]] = None,
    prior_decision: Optional[Dict[str, Any]] = None,
) -> float:
    """U_vote: voting utility using public trust + discussion commitment."""
    base = _score_action_for_role(
        target,
        my_role,
        day_night,
        "voting",
        target_report,
        state=state,
        agent_name=agent_name,
        round_signals=round_signals,
        prior_decision=prior_decision,
    )
    get_field, _, _ = _empathy_field_imports()
    field = get_field(state) if state else None
    report = target_report or {}

    if field and target.startswith("Player"):
        trust_pub = field.get_trust_public(target)
        if trust_pub >= 0.18:
            base -= 0.36 * min(1.0, trust_pub / 0.55)
        hard_acc = field.get_accusation_count(target, hard_only=True)
        soft_acc = field.get_accusation_count(target, hard_only=False) - hard_acc
        base += 0.18 * min(2, hard_acc) + 0.06 * min(2, max(0, soft_acc))

    pub_trust = float(report.get("public_trust", 0.0))
    if pub_trust >= 0.22:
        base -= 0.30 * min(1.0, pub_trust / 0.5)

    try:
        from .belief_state import get_belief_state
    except ImportError:
        from chatarena.belief_state import get_belief_state

    belief = get_belief_state(state) if state else None
    if belief and belief.viewer == agent_name:
        delta = belief.vote_consistency_delta(target, my_role)
        if my_role in ("werewolf", "wolf"):
            base += -delta * 0.35
        else:
            base += delta
        prior_decision = prior_decision or belief.get_prior_decision()
    elif prior_decision:
        ps = prior_decision.get("stance")
        pf = prior_decision.get("target_player")
        pl = prior_decision.get("vote_lean")
        if ps == "support" and target == pf:
            base -= 0.45
        if ps in ("suspect", "caution") and pl and target == pl:
            base += 0.2
        brief = prior_decision.get("decision_brief", "")
        if ps == "support" and pf and target != pf and "trust" in str(brief).lower():
            base -= 0.12

    is_wolf = my_role in ("werewolf", "wolf")
    analytics = get_game_analytics(state) if state else {}
    agent_prof = analytics.get("player_profiles", {}).get(agent_name, _empty_player_profile())
    hard = float(report.get("hard_wolf_prob", 0.0))
    votes_recv = int(report.get("votes_received", 0))
    vote_pressure = float(report.get("current_round_vote_pressure", 0.0))
    svc = float(report.get("speech_vote_consistency", 1.0))
    leader = analytics.get("current_round_leader", "")
    leader_count = int(analytics.get("current_round_leader_count", 0))

    if not is_wolf:
        if my_role == "witch" and day_night == "night":
            round_no = int(_get_state_field(state, "round_no", 1)) if state else 1
            if round_no <= 1:
                return 1.0 if target == "pass" else 0.0
        if target in agent_prof.get("trust_mentioned", []):
            base -= 0.4
        if svc < 0.45 and target in agent_prof.get("trust_mentioned", []):
            base -= 0.15
        if votes_recv >= 3 and hard < 0.28:
            base -= 0.22
        if leader == target and leader_count >= 3 and hard < 0.3:
            base -= 0.18
        if vote_pressure >= 0.45 and hard < 0.25:
            base -= 0.12
    else:
        if my_role in ("werewolf", "wolf") and day_night == "night":
            if target == "pass":
                return 0.0
            if leader == target and leader_count >= 2 and hard < 0.35:
                base += 0.14
            if votes_recv >= 2 and hard < 0.3:
                base += 0.1
            if svc >= 0.85:
                base += 0.06
        else:
            # Daytime wolves should behave like adversarial misleaders, not truth-seeking villagers.
            if target in agent_prof.get("trust_mentioned", []):
                base += 0.08
            if vote_pressure >= 0.45:
                base += 0.05

    return max(0.0, min(1.0, base + random.uniform(-0.03, 0.03)))


def _score_deep_action(
    action_tuple: Tuple[str, str, str],
    intent_meta: Dict[str, Any],
    my_role: str,
    day_night: str,
    game_phase: str,
    empathy_reports: Dict[str, Dict[str, Any]],
    state=None,
    agent_name: str = "",
    round_signals: Optional[Dict[str, Any]] = None,
    prior_decision: Optional[Dict[str, Any]] = None,
) -> float:
    vote_target, target_player, speech_style = action_tuple
    intent = (intent_meta or {}).get("intent", "vote")
    target_key = target_player if target_player not in ("pass", "") else vote_target
    report = empathy_reports.get(target_key, {}) if target_key not in ("pass", "") else {}

    if game_phase == "voting" or day_night == "night":
        if my_role == "witch" and day_night == "night":
            round_no = int(_get_state_field(state, "round_no", 1)) if state else 1
            if round_no <= 1:
                return 0.0
            if intent != "pass":
                return 0.0
        elif my_role in ("werewolf", "wolf") and day_night == "night":
            if intent not in ("kill", "vote", "pass"):
                return 0.0
        elif my_role == "guard" and day_night == "night":
            if intent not in ("protect", "pass"):
                return 0.0
        elif my_role == "seer" and day_night == "night":
            if intent not in ("verify", "pass"):
                return 0.0
        return _score_vote_utility(
            vote_target,
            my_role,
            day_night,
            report,
            state=state,
            agent_name=agent_name,
            round_signals=round_signals,
            prior_decision=prior_decision,
        )
    return _score_discussion_utility(
        intent,
        vote_target,
        target_player,
        speech_style,
        my_role,
        day_night,
        game_phase,
        report,
        state=state,
        agent_name=agent_name,
        round_signals=round_signals,
        empathy_reports=empathy_reports,
    )


# ============ MCTS scoring (no LLM) ============

def fast_evaluate_state(state, empathy_data=None) -> float:
    alive_count = len(_get_state_field(state, "alive_players", []) or [])
    base_score = min(1.0, alive_count / 7.0)
    return base_score + random.uniform(-0.05, 0.05)


def _score_action_for_role(
    target: str,
    my_role: str,
    day_night: str,
    game_phase: str,
    target_report: Dict[str, Any],
    state=None,
    agent_name: str = "",
    round_signals: Optional[Dict[str, Any]] = None,
    prior_decision: Optional[Dict[str, Any]] = None,
) -> float:
    """Score a candidate target: evidence-weighted suspicion + information gain - bandwagon."""
    if target == "pass" or not target:
        if game_phase == "discussion" and state:
            return _score_reveal_action(state, my_role, agent_name, game_phase) + random.uniform(-0.03, 0.03)
        if my_role == "witch" and day_night == "night":
            round_no = int(_get_state_field(state, "round_no", 1)) if state else 1
            if round_no <= 1:
                return 1.0
        return 0.35 + random.uniform(-0.05, 0.05)

    report = target_report or _default_player_report()
    hard = float(report.get("hard_wolf_prob", 0.0))
    soft = float(report.get("soft_wolf_prob", report.get("role_probability", {}).get("werewolf", 0.3)))
    werewolf_prob = _posterior_wolf_prob(report)
    trust = report.get("trust_score", 1.0 - werewolf_prob)
    stance = report.get("stance_to_me", 0.0)
    influence = report.get("influence", 0.5)
    round_signals = round_signals or {}

    bandwagon_pen = _compute_bandwagon_penalty(target, report, round_signals)
    info_gain = _compute_information_gain(
        state, my_role, agent_name, target, report, game_phase, round_signals
    ) if state else float(report.get("information_gain", 0.0))

    is_wolf = my_role in ("werewolf", "wolf")
    is_night = day_night == "night"
    effective_suspicion = hard * 0.68 + soft * 0.22

    if is_wolf:
        if is_night:
            # 狼人夜晚只应选择“要杀谁”，不应生成任何 reveal 风格动作
            score = trust * 0.55 + influence * 0.25 + (1.0 - werewolf_prob) * 0.15
            if stance > 0.2:
                score += 0.05
            if report.get("public_trust", 0.0) > 0.2:
                score += 0.03
            if report.get("information_gain", 0.0) > 0.2:
                score += 0.02
        else:
            # 白天狼人应表现为对抗性怀疑，而不是村民式求真
            score = effective_suspicion * 0.35 + (1.0 - abs(stance)) * 0.2 + influence * 0.15
            if stance < -0.2:
                score += 0.1
            if report.get("public_trust", 0.0) > 0.25:
                score += 0.05
            if report.get("information_gain", 0.0) > 0.22:
                score += 0.03
    else:
        if is_night:
            if my_role == "guard":
                score = effective_suspicion * 0.15 + trust * 0.5 + (1.0 - influence) * 0.1
            elif my_role == "seer":
                score = effective_suspicion * 0.5 + (1.0 - abs(stance)) * 0.15
            elif my_role == "witch":
                round_no = int(_get_state_field(state, "round_no", 1)) if state else 1
                early_night = round_no <= 1
                # 第一晚通常信息不足，毒药需要更强证据才值得出手
                if early_night:
                    # 第一夜女巫毒药应当几乎总是保留，除非极端硬证据
                    score = effective_suspicion * 0.03 + info_gain * 0.03 + trust * 0.04
                    score -= 0.65
                    if hard >= 0.75 or report.get("private_verified_wolf"):
                        score += 0.25
                    if report.get("public_trust", 0.0) >= 0.15:
                        score -= 0.12
                    if report.get("misdirection_risk", 0.0) > 0.35:
                        score += 0.03
                else:
                    score = effective_suspicion * 0.28 + info_gain * 0.18 + (1.0 - trust) * 0.08
                    if hard >= 0.4:
                        score += 0.18
                    if report.get("bandwagon_risk") and hard < 0.25:
                        score -= 0.10
            else:
                score = 0.4
        else:
            score = (
                effective_suspicion * 0.38
                + info_gain * 0.32
                + (0.5 - stance) * 0.12
                + influence * 0.08
                - bandwagon_pen * 0.35
            )
            if game_phase == "endgame":
                score += hard * 0.12
            if report.get("bandwagon_risk") and hard < 0.25:
                score -= 0.1

    if prior_decision and game_phase == "voting":
        prior_stance = prior_decision.get("stance")
        prior_lean = prior_decision.get("vote_lean")
        prior_focus = prior_decision.get("target_player")
        if prior_stance == "support" and target == prior_focus:
            score -= 0.35
        if prior_stance in ("suspect", "caution") and target == prior_lean:
            score += 0.18
        if prior_stance == "reveal" and hard >= 0.35:
            score += 0.12
        if prior_lean and prior_lean not in ("pass", "") and target == prior_lean:
            score += 0.1

    return max(0.0, min(1.0, score + random.uniform(-0.04, 0.04)))


def _run_mcts_tree(
    root_state,
    player: str,
    legal_actions: List[str],
    empathy_reports: Dict[str, Dict],
    my_role: str,
    day_night: str,
    game_phase: str,
    iter_num: int,
    prior_decision: Optional[Dict[str, Any]] = None,
) -> MCTSNode:
    """
    Step 3 deep MCTS: intent×target children, U_disc / U_vote scoring,
    optional simulate_propagate rollout for discussion.
    """
    root = MCTSNode(root_state, player)
    if not legal_actions:
        legal_actions = ["pass"]

    round_signals = extract_round_discussion_signals(_get_state_field(root_state, "history", []) or [])

    candidates = _expand_deep_action_candidates(
        legal_actions, root_state, player, my_role, game_phase, day_night, empathy_reports
    )
    if not candidates:
        candidates = [(legal_actions[0], legal_actions[0], "ambiguous", {"intent": "vote", "layer": 2})]

    for vote_target, target_player, style, meta in candidates:
        action = (vote_target, target_player, style)
        report_key = target_player if target_player not in ("pass", "") else vote_target
        report = empathy_reports.get(report_key, {}) if report_key not in ("pass", "") else {}
        child = MCTSNode(root_state, player, root, action)
        child.intent_meta = dict(meta)
        child.empathy_context = _patch_empathy_context_with_intent(
            _generate_empathy_based_speech(action, report, root_state),
            meta,
        )
        root.children.append(child)

    if len(root.children) == 1:
        child = root.children[0]
        child.visits = 1
        child.value = _score_deep_action(
            child.action, child.intent_meta, my_role, day_night, game_phase,
            empathy_reports, state=root_state, agent_name=player,
            round_signals=round_signals, prior_decision=prior_decision,
        )
        return child

    sims = max(iter_num, len(root.children))
    for _ in range(sims):
        node = root
        while node.children and node.visits > 0:
            node = node.best_child()
            if node is None:
                break
        if node is None:
            node = root

        reward = _score_deep_action(
            node.action, node.intent_meta, my_role, day_night, game_phase,
            empathy_reports, state=root_state, agent_name=player,
            round_signals=round_signals, prior_decision=prior_decision,
        )
        reward = 0.58 * reward + 0.42 * fast_evaluate_state(root_state, empathy_reports)

        cur = node
        while cur is not None:
            cur.visits += 1
            cur.value += reward
            cur = cur.parent

    best = max(root.children, key=lambda c: (c.visits, c.value / max(c.visits, 1)))
    intent_label = (best.intent_meta or {}).get("intent", "?")
    logger.info(
        f"[DeepMCTS] {player} intent={intent_label} target={best.action[0]} "
        f"(visits={best.visits}, avg={best.value / max(best.visits, 1):.3f})"
    )
    return best


def select_action_with_mcts(
    observation: str = "",
    player_name: str = "",
    legal_actions: List[str] = None,
    backend=None,
    n_simulations: int = 10,
    use_empathy: bool = True,
    root_state=None,
    empathy_data: Optional[Dict] = None,
    player_roles: Optional[Dict] = None,
    prior_decision: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, str, Dict[str, Any]]:
    """
    MCTS action selection. Returns (action, target, style, empathy_context).
    Never calls LLM.
    """
    if legal_actions is None:
        legal_actions = ["pass"]
    if not legal_actions:
        legal_actions = ["pass"]

    state = root_state if root_state is not None else {"history": [], "alive_players": legal_actions}
    my_role = player_roles.get(player_name, "") if player_roles else _get_state_field(state, "my_role", "villager")
    if not my_role and player_roles:
        my_role = player_roles.get(player_name, "villager")
    day_night = _get_state_field(state, "day_night", "daytime")
    game_phase = _get_state_field(state, "game_phase", detect_game_phase(_get_state_field(state, "alive_players", legal_actions)))

    # Phase-first routing: night actions must never inherit discussion-only intents like reveal.
    round_no = int(_get_state_field(state, "round_no", 1)) if state else 1
    filtered_legal_actions = list(legal_actions)
    if day_night == "night" or game_phase == "voting":
        filtered_legal_actions = [
            a for a in filtered_legal_actions
            if a == "pass" or re.match(r"^Player\s+\d+$", str(a).strip(), re.I)
        ]
        if my_role == "witch" and day_night == "night" and round_no <= 1:
            # First-night witch is pass-only for poison routing; save phase is handled elsewhere.
            filtered_legal_actions = ["pass"]

    reports = build_empathy_reports(state, player_name, empathy_data if use_empathy else None)

    player_reports = _player_reports_only(reports)
    best_node = _run_mcts_tree(
        state, player_name, filtered_legal_actions, player_reports,
        my_role, day_night, game_phase, n_simulations,
        prior_decision=prior_decision,
    )
    action, target, style = best_node.action
    return action, target, style, best_node.empathy_context


# ============ Speech metadata for openai.py (no LLM here) ============

def generate_llm_speech(agent, action, empathy_json, game_state, args, msgs, ques):
    """
    返回共情发言指导元数据（不调用 LLM）。
    实际 LLM 调用在 openai.py 的 _generate_mcts_phase_response 中按阶段执行。
    """
    if isinstance(action, tuple) and len(action) >= 3:
        vote_target, target_player, speech_style = action
    else:
        vote_target = str(action)
        target_player = vote_target
        speech_style = "neutral"

    player_reports = {}
    if empathy_json and isinstance(empathy_json, dict):
        player_reports = empathy_json.get("player_reports", empathy_json)

    target_empathy = player_reports.get(target_player, {}) if target_player else {}
    empathy_context = _generate_empathy_based_speech(
        (vote_target, target_player, speech_style), target_empathy, game_state
    )

    # Only discussion needs natural-language generation.
    # Voting/night use MCTS+empathy to pick the target, then openai.py renders a template.
    phase = _get_state_field(game_state, "game_phase", "")
    day_night = _get_state_field(game_state, "day_night", "daytime")
    if phase == "voting" or day_night == "night":
        return {
            "use_llm": False,
            "empathy_context": empathy_context,
            "template_action": (vote_target, target_player, speech_style),
        }

    return {
        "use_llm": True,
        "empathy_context": empathy_context,
        "speech_style": speech_style,
        "target_player": target_player,
    }


def _generate_fallback_speech(action, player_name: str, my_role: str = "villager") -> str:
    if isinstance(action, tuple):
        target = action[0]
    else:
        target = str(action)
    if target and target != "pass":
        return f"Based on my analysis, {target}'s behavior deserves attention."
    return "I need more information before making a judgment."


# Legacy helper – kept for compatibility, no LLM
def _generate_speech_for_action(action, player_name, backend=None, observation: str = "") -> str:
    return _generate_fallback_speech(action, player_name)


# ============ Eval / simulation helpers ============

def fast_generate_action(player: str, state: GameState) -> Tuple[str, str, str]:
    alive = [p for p in state.alive_players if p != player and p != "pass"]
    if not alive:
        return ("pass", "pass", "ambiguous")
    target = random.choice(alive)
    return (target, target, "neutral")


def fast_process_voting(state: GameState) -> GameState:
    votes = state.get("votes", [])
    if not votes:
        return state
    tally: Dict[str, int] = {}
    for _, target in votes:
        if target and target != "pass":
            tally[target] = tally.get(target, 0) + 1
    if not tally:
        return state
    lynched = max(tally, key=tally.get)
    new_state = GameState(dict(state))
    new_state.kill_player(lynched)
    new_state["votes"] = []
    return new_state


def fast_check_game_end(state: GameState) -> bool:
    roles = state.player_roles or {}
    alive = state.alive_players
    wolves = [p for p in alive if roles.get(p) == "werewolf"]
    goods = [p for p in alive if roles.get(p) != "werewolf"]
    if not wolves or not goods:
        return True
    return len(wolves) >= len(goods)


# ============ Public API ============

def MCTS(
    root_state=None,
    current_player=None,
    backend=None,
    agent_name=None,
    iter_num=10,
    args=None,
    msgs=None,
    ques=None,
    empathy_data=None,
    player_roles=None,
    prior_decision=None,
    observation: str = "",
    player_name: str = "",
    legal_actions: List[str] = None,
    **kwargs,
):
    """
    Main MCTS entry. Returns MCTSNode with:
      - action: (target, target, speech_style)
      - empathy_context: dict for final LLM speech in openai.py

    NO LLM calls inside this function.
    """
    player = current_player or agent_name or player_name
    n_simulations = iter_num if iter_num else 10
    use_mcts = getattr(args, "is_mcts", 1) == 1 if args else True
    use_empathy = getattr(args, "use_empathy", True) if args else True

    if root_state is not None:
        if isinstance(root_state, dict):
            alive_players = root_state.get("alive_players", [])
            observation_text = root_state.get("observation", "")
            my_role = player_roles.get(player, "") if player_roles else root_state.get("my_role", "")
        else:
            alive_players = getattr(root_state, "alive_players", [])
            observation_text = getattr(root_state, "observation", "")
            my_role = player_roles.get(player, "") if player_roles else getattr(root_state, "my_role", "")

        legal = [p for p in alive_players if p != player and p != "pass"]
        if not legal:
            legal = ["pass"]

        if not use_mcts:
            target = legal[0] if legal else "pass"
            style = "neutral"
            result_node = MCTSNode(root_state, player or "unknown")
            result_node.action = (target, target, style)
            result_node.empathy_context = {}
            return result_node

        action, target, style, empathy_context = select_action_with_mcts(
            observation=observation_text,
            player_name=player,
            legal_actions=legal,
            backend=None,
            n_simulations=n_simulations,
            use_empathy=use_empathy,
            root_state=root_state,
            empathy_data=empathy_data,
            player_roles=player_roles,
            prior_decision=prior_decision,
        )

        result_node = MCTSNode(root_state, player)
        result_node.action = (action, target, style)
        result_node.empathy_context = empathy_context
        result_node.visits = n_simulations
        logger.info(f"[MCTS] Decision: action={result_node.action}")
        return result_node

    # Old calling convention
    if legal_actions is None:
        legal_actions = ["pass"]
    if not use_mcts:
        action = legal_actions[0] if legal_actions else "pass"
        return action, _generate_fallback_speech(action, player_name)

    action, target, style, _ = select_action_with_mcts(
        observation=observation,
        player_name=player_name,
        legal_actions=legal_actions,
        backend=None,
        n_simulations=n_simulations,
        use_empathy=use_empathy,
        empathy_data=empathy_data,
        player_roles=player_roles,
    )
    return action, _generate_fallback_speech((action, target, style), player_name)
