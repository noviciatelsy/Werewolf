"""
Public empathy field + utterance propagation (Step 1–2 architecture).

- PublicEmpathyField: shared observable claims/trust/accusations (not private MCTS trees).
- StrategicPlan / speech_act: structured output from MCTS → LLM speech.
- UtteranceEffectModel: update field after each public utterance; viewer-specific report patches.
"""
from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

# Reuse evidence patterns from MCTS when available
try:
    from .MCTS import (
        _classify_evidence_in_text,
        _extract_accusations_from_message,
        _default_player_report,
    )
except ImportError:
    from chatarena.MCTS import (
        _classify_evidence_in_text,
        _extract_accusations_from_message,
        _default_player_report,
    )


class ClaimDict(TypedDict, total=False):
    speaker: str
    claim_type: str
    target: str
    detail: str
    evidence_type: str


class SpeechActDict(TypedDict, total=False):
    speaker: str
    content: str
    intent: str
    target: str
    claims: List[ClaimDict]
    stance: str
    vote_lean: str
    speech_style: str


class StrategicPlanDict(TypedDict, total=False):
    intent: str
    target: str
    claims: List[ClaimDict]
    stance: str
    vote_lean: str
    speech_style: str
    decision_brief: str
    vote_target: str
    target_player: str


_TRUST_PATTERNS = (
    r"trust(?:s|ing)?\s+(player \d+)",
    r"(player \d+).{0,40}(?:solid|thoughtful|good).{0,20}reasoning",
    r"agree with (player \d+)",
    r"leaning toward trusting (player \d+)",
    r"(player \d+).{0,30}on our side",
    r"not a werewolf",
    r"verified .+ (?:good|not)",
)
_ROLE_CLAIM_PATTERNS = (
    (r"i am the seer|i am seer", "seer"),
    (r"i am the witch|i am witch", "witch"),
    (r"i am the guard|i am guard", "guard"),
)
_SAVE_PATTERNS = (
    r"i saved (player \d+)",
    r"used my antidote",
    r"witch.{0,20}saved",
)
_VERIFY_GOOD_PATTERNS = (
    r"verified (player \d+).{0,30}not a werewolf",
    r"(player \d+) is not a werewolf",
    r"checked (player \d+).{0,20}good",
)
_VERIFY_WOLF_PATTERNS = (
    r"verified (player \d+).{0,30}werewolf",
    r"(player \d+) is a werewolf",
)


def _norm_player(raw: str) -> str:
    m = re.search(r"player\s*(\d+)", str(raw).lower())
    if m:
        return f"Player {m.group(1)}"
    if str(raw).startswith("Player"):
        return str(raw).split()[0] + " " + str(raw).split()[1] if len(str(raw).split()) > 1 else raw
    return raw


class PublicEmpathyField:
    """Observable game-wide empathy / claim board (derived from public speech)."""

    def __init__(self, round_no: int = 1):
        self.round_no = round_no
        self.claims: List[ClaimDict] = []
        self.trust_edges: Dict[str, float] = {}  # "from|to" -> weight increment sum
        self.role_claims: Dict[str, str] = {}  # speaker -> seer|witch|guard
        self.accusations: List[Dict[str, Any]] = []
        self.speech_acts: List[SpeechActDict] = []
        self.version: int = 0

    def edge_key(self, from_p: str, to_p: str) -> str:
        return f"{from_p}|{to_p}"

    def add_trust(self, from_p: str, to_p: str, delta: float = 0.15) -> None:
        if not from_p or not to_p or from_p == to_p:
            return
        k = self.edge_key(from_p, to_p)
        self.trust_edges[k] = min(1.0, self.trust_edges.get(k, 0.0) + delta)

    def get_trust_public(self, target: str) -> float:
        """Aggregate trust directed at target from all public speakers."""
        total = 0.0
        for k, w in self.trust_edges.items():
            _, to_p = k.split("|", 1)
            if to_p == target:
                total += w
        return min(1.0, total)

    def get_accusation_count(self, target: str, hard_only: bool = False) -> int:
        n = 0
        for acc in self.accusations:
            if acc.get("target") != target:
                continue
            if hard_only and acc.get("evidence_type") != "hard":
                continue
            n += 1
        return n

    def bump_version(self) -> None:
        self.version += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_no": self.round_no,
            "claims": deepcopy(self.claims),
            "trust_edges": dict(self.trust_edges),
            "role_claims": dict(self.role_claims),
            "accusations": deepcopy(self.accusations),
            "speech_acts": deepcopy(self.speech_acts),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PublicEmpathyField":
        field = cls(round_no=int((data or {}).get("round_no", 1)))
        if not data:
            return field
        field.claims = list(data.get("claims", []))
        field.trust_edges = dict(data.get("trust_edges", {}))
        field.role_claims = dict(data.get("role_claims", {}))
        field.accusations = list(data.get("accusations", []))
        field.speech_acts = list(data.get("speech_acts", []))
        field.version = int(data.get("version", 0))
        return field


def parse_speech_act_from_text(speaker: str, content: str) -> SpeechActDict:
    """Parse structured speech_act from natural language (rule-based)."""
    text = (content or "").strip()
    lower = text.lower()
    act: SpeechActDict = {
        "speaker": speaker,
        "content": text[:500],
        "claims": [],
        "intent": "neutral",
        "target": "",
        "stance": "neutral",
    }
    ev = _classify_evidence_in_text(text)

    for pat, role in _ROLE_CLAIM_PATTERNS:
        if re.search(pat, lower):
            act["intent"] = "reveal"
            act["claims"].append({
                "speaker": speaker,
                "claim_type": "role_claim",
                "target": speaker,
                "detail": role,
                "evidence_type": "hard",
            })

    for pat in _SAVE_PATTERNS:
        m = re.search(pat, lower)
        if m:
            tgt = _norm_player(m.group(1)) if m.lastindex else ""
            act["intent"] = "reveal"
            act["claims"].append({
                "speaker": speaker,
                "claim_type": "witch_save",
                "target": tgt,
                "detail": "saved",
                "evidence_type": "hard",
            })

    for pat in _VERIFY_GOOD_PATTERNS:
        m = re.search(pat, lower)
        if m:
            tgt = _norm_player(m.group(1))
            act["intent"] = "reveal"
            act["claims"].append({
                "speaker": speaker,
                "claim_type": "seer_verify_good",
                "target": tgt,
                "detail": "good",
                "evidence_type": "hard",
            })
            act["target"] = tgt
            act["stance"] = "support"

    for pat in _VERIFY_WOLF_PATTERNS:
        m = re.search(pat, lower)
        if m:
            tgt = _norm_player(m.group(1))
            act["intent"] = "reveal"
            act["claims"].append({
                "speaker": speaker,
                "claim_type": "seer_verify_wolf",
                "target": tgt,
                "detail": "wolf",
                "evidence_type": "hard",
            })
            act["target"] = tgt
            act["stance"] = "suspect"

    for pat in _TRUST_PATTERNS:
        for m in re.finditer(pat, lower):
            tgt = _norm_player(m.group(1)) if m.lastindex else ""
            if tgt and tgt.startswith("Player"):
                act["stance"] = "support"
                if not act["target"]:
                    act["target"] = tgt
                if act["intent"] == "neutral":
                    act["intent"] = "support"

    accs = _extract_accusations_from_message(speaker, text)
    if accs:
        act["intent"] = "accuse"
        for acc in accs:
            act["claims"].append({
                "speaker": speaker,
                "claim_type": "accusation",
                "target": acc["target"],
                "detail": acc.get("snippet", "")[:80],
                "evidence_type": acc.get("evidence_type", "soft"),
            })
            if not act["target"]:
                act["target"] = acc["target"]
            act["stance"] = "suspect"

    if ev == "soft" and act["intent"] == "neutral" and any(
        k in lower for k in ("suspicious", "watch", "doubt")
    ):
        act["intent"] = "probe"
        act["stance"] = "caution"

    if any(k in lower for k in ("who did you check", "share", "claim", "reveal")):
        if act["intent"] == "neutral":
            act["intent"] = "press"

    return act


def strategic_plan_from_mcts(
    action: Union[Tuple[str, str, str], List[str]],
    empathy_context: Optional[Dict[str, Any]],
    role: str,
    agent_name: str,
) -> StrategicPlanDict:
    """Build StrategicPlan from MCTS action tuple + empathy context."""
    vote_target, target_player, speech_style = action
    ctx = empathy_context or {}
    intent_info = ctx.get("agent_intent") or {}
    mcts_intent = ctx.get("mcts_intent") or intent_info.get("mcts_intent") or intent_info.get("stance", "neutral")
    plan: StrategicPlanDict = {
        "intent": mcts_intent,
        "target": target_player or vote_target or "",
        "target_player": target_player,
        "vote_target": vote_target,
        "vote_lean": ctx.get("vote_lean", vote_target),
        "speech_style": speech_style,
        "decision_brief": ctx.get("decision_brief", ""),
        "stance": intent_info.get("stance", "neutral"),
        "claims": [],
    }
    if speech_style == "reveal" or plan["intent"] == "reveal":
        plan["intent"] = "reveal"
    elif speech_style in ("evidence", "counter"):
        plan["intent"] = "accuse" if plan.get("stance") == "suspect" else plan["intent"]
    elif speech_style in ("align", "soothe"):
        plan["intent"] = "support"

    if plan["intent"] == "reveal":
        if role in ("seer", "guard", "witch"):
            plan["claims"].append({
                "speaker": agent_name,
                "claim_type": f"{role}_pending",
                "target": target_player if target_player not in ("pass", agent_name) else "",
                "detail": speech_style,
                "evidence_type": "hard",
            })
    return plan


def merge_plan_and_text(plan: Optional[StrategicPlanDict], speaker: str, text: str) -> SpeechActDict:
    """Prefer explicit plan claims; fill gaps from parsed text."""
    parsed = parse_speech_act_from_text(speaker, text)
    if not plan:
        return parsed
    merged: SpeechActDict = {
        "speaker": speaker,
        "content": text[:500],
        "intent": plan.get("intent") or parsed.get("intent", "neutral"),
        "target": plan.get("target") or parsed.get("target", ""),
        "stance": plan.get("stance") or parsed.get("stance", "neutral"),
        "vote_lean": plan.get("vote_lean", ""),
        "speech_style": plan.get("speech_style", ""),
        "claims": list(plan.get("claims") or []) + list(parsed.get("claims") or []),
    }
    if not merged["claims"]:
        merged["claims"] = parsed.get("claims", [])
    return merged


class UtteranceEffectModel:
    """Apply public utterances to PublicEmpathyField (real + simulated)."""

    @staticmethod
    def propagate(
        field: PublicEmpathyField,
        speaker: str,
        content: str = "",
        speech_act: Optional[SpeechActDict] = None,
        plan: Optional[StrategicPlanDict] = None,
    ) -> SpeechActDict:
        if speech_act is None:
            speech_act = merge_plan_and_text(plan, speaker, content)
        else:
            speech_act = dict(speech_act)
            if content and not speech_act.get("content"):
                speech_act["content"] = content[:500]

        field.speech_acts.append(speech_act)
        claims = speech_act.get("claims") or []

        for claim in claims:
            field.claims.append(dict(claim))
            ctype = claim.get("claim_type", "")
            target = claim.get("target", "")
            if ctype == "role_claim" and claim.get("detail"):
                field.role_claims[speaker] = str(claim["detail"])
            if target and target.startswith("Player") and ctype.startswith("seer_verify_good"):
                field.add_trust(speaker, target, 0.25)
            if target and ctype.startswith("seer_verify_wolf"):
                field.accusations.append({
                    "accuser": speaker,
                    "target": target,
                    "evidence_type": "hard",
                    "snippet": claim.get("detail", ""),
                })

        # Trust from speech_act stance
        tgt = speech_act.get("target", "")
        if speech_act.get("stance") == "support" and tgt.startswith("Player"):
            field.add_trust(speaker, tgt, 0.2)

        # Parse additional trust from raw text
        parsed = parse_speech_act_from_text(speaker, content or speech_act.get("content", ""))
        for pat in _TRUST_PATTERNS:
            for m in re.finditer(pat, (content or "").lower()):
                if m.lastindex:
                    field.add_trust(speaker, _norm_player(m.group(1)), 0.12)

        for acc in _extract_accusations_from_message(speaker, content or ""):
            field.accusations.append(acc)

        field.bump_version()
        return speech_act

    @staticmethod
    def simulate_propagate(
        field: PublicEmpathyField,
        speaker: str,
        plan: StrategicPlanDict,
    ) -> PublicEmpathyField:
        """Copy field and apply hypothetical utterance (for MCTS rollout, future step)."""
        copy = PublicEmpathyField.from_dict(field.to_dict())
        UtteranceEffectModel.propagate(
            copy, speaker, content="", speech_act=merge_plan_and_text(plan, speaker, "")
        )
        return copy


def sync_field_from_history(
    field: PublicEmpathyField,
    history: List[Tuple[str, str]],
    up_to_round: Optional[int] = None,
) -> PublicEmpathyField:
    """Rebuild field from conversation history (player speeches only)."""
    field.claims.clear()
    field.trust_edges.clear()
    field.role_claims.clear()
    field.accusations.clear()
    field.speech_acts.clear()
    in_discussion = False
    for speaker, content in history or []:
        if not speaker.startswith("Player"):
            cl = (content or "").lower()
            if "discussion phase" in cl or "freely talk" in cl:
                in_discussion = True
            if "voting phase" in cl:
                in_discussion = False
            continue
        if in_discussion or _classify_evidence_in_text(content) != "none":
            UtteranceEffectModel.propagate(field, speaker, content)
        if "i vote to kill" in (content or "").lower():
            m = re.search(r"i vote to kill (player \d+)", content.lower())
            if m:
                tgt = _norm_player(m.group(1))
                field.accusations.append({
                    "accuser": speaker,
                    "target": tgt,
                    "evidence_type": "hard",
                    "snippet": "vote",
                })
    field.bump_version()
    return field


def apply_field_to_viewer_reports(
    field: Optional[PublicEmpathyField],
    viewer: str,
    reports: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Patch per-viewer empathy reports using public field (trust, claims, bandwagon).
    viewer = current agent; stance_to_me unchanged here (set in enrich from history).
    """
    if not field or not reports:
        return reports

    for player, report in reports.items():
        if not str(player).startswith("Player"):
            continue
        if player == viewer:
            continue
        base = report if isinstance(report, dict) else _default_player_report()
        if not isinstance(base.get("role_probability"), dict):
            try:
                from .MCTS import _ensure_report_schema
            except ImportError:
                from chatarena.MCTS import _ensure_report_schema
            base = _ensure_report_schema(base)
        hard = float(base.get("hard_wolf_prob", 0.0))
        soft = float(base.get("soft_wolf_prob", base.get("role_probability", {}).get("werewolf", 0.3)))

        trust_pub = field.get_trust_public(player)
        if trust_pub >= 0.25:
            soft = max(0.05, soft - 0.15 * min(1.0, trust_pub / 0.5))
            hard = max(0.0, hard - 0.1 * min(1.0, trust_pub / 0.5))

        if player in field.role_claims:
            claimed = field.role_claims[player]
            if claimed in ("seer", "witch", "guard"):
                for c in field.claims:
                    if c.get("speaker") == player and c.get("evidence_type") == "hard":
                        if "verify_good" in c.get("claim_type", "") and c.get("target", "").startswith("Player"):
                            hard = max(0.0, hard - 0.08)

        acc_count = field.get_accusation_count(player, hard_only=False)
        hard_acc = field.get_accusation_count(player, hard_only=True)
        if acc_count >= 2 and hard_acc == 0:
            base["bandwagon_risk"] = True
            soft *= 0.6
        else:
            base["bandwagon_risk"] = base.get("bandwagon_risk", False)

        if hard_acc > 0:
            hard = min(0.95, hard + 0.1 * hard_acc)

        combined = min(0.95, max(0.05, hard * 0.72 + soft * 0.28))
        base["hard_wolf_prob"] = round(hard, 3)
        base["soft_wolf_prob"] = round(soft, 3)
        base["role_probability"]["werewolf"] = round(combined, 3)
        base["trust_score"] = round(1.0 - combined, 3)
        base["public_trust"] = round(trust_pub, 3)
        reports[player] = base

    return reports


def get_public_field_from_state(game_state) -> Optional[PublicEmpathyField]:
    if game_state is None:
        return None
    raw = None
    if isinstance(game_state, dict):
        raw = game_state.get("public_empathy_field")
    else:
        raw = getattr(game_state, "public_empathy_field", None)
    if isinstance(raw, PublicEmpathyField):
        return raw
    if isinstance(raw, dict):
        return PublicEmpathyField.from_dict(raw)
    return None
