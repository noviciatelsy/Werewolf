"""
Unified BeliefState: single posterior P(wolf|player) for empathy + MCTS.

All decision modules read/write through BeliefState attached to GameState.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DiscussionCommitment:
    """This-round integrated stance (discussion → vote coherence)."""
    round_no: int = 0
    stance: str = "neutral"
    focus_player: str = ""
    vote_lean: str = "pass"
    mcts_intent: str = ""
    decision_brief: str = ""
    public_trust_targets: List[str] = field(default_factory=list)
    public_accuse_targets: List[str] = field(default_factory=list)

    def to_prior_decision(self) -> Dict[str, Any]:
        return {
            "stance": self.stance,
            "target_player": self.focus_player,
            "vote_lean": self.vote_lean,
            "decision_brief": self.decision_brief,
            "mcts_intent": self.mcts_intent,
            "round_no": self.round_no,
        }


@dataclass
class PlayerBelief:
    """Scalar + diagnostic fields for one player (viewer-relative)."""
    player: str
    posterior_wolf: float = 0.3
    llm_wolf: float = 0.3
    hard_wolf: float = 0.0
    soft_wolf: float = 0.3
    trust: float = 0.7
    public_trust: float = 0.0
    stance_to_me: float = 0.0
    votes_received: int = 0
    vote_pressure: float = 0.0
    speech_vote_align: float = 1.0
    bandwagon_risk: bool = False
    private_verified_good: bool = False
    private_verified_wolf: bool = False
    public_contradiction: str = ""
    recommended_action: str = "observe"
    information_gain: float = 0.0

    def to_report(self) -> Dict[str, Any]:
        return {
            "posterior_wolf_prob": round(self.posterior_wolf, 3),
            "role_probability": {
                "werewolf": round(self.posterior_wolf, 3),
                "villager": round(max(0.0, 1.0 - self.posterior_wolf - 0.15), 3),
                "seer": 0.08,
                "witch": 0.08,
                "guard": 0.08,
            },
            "hard_wolf_prob": round(self.hard_wolf, 3),
            "soft_wolf_prob": round(self.soft_wolf, 3),
            "trust_score": round(self.trust, 3),
            "public_trust": round(self.public_trust, 3),
            "stance_to_me": round(self.stance_to_me, 3),
            "votes_received": self.votes_received,
            "current_round_vote_pressure": round(self.vote_pressure, 3),
            "speech_vote_consistency": round(self.speech_vote_align, 3),
            "bandwagon_risk": self.bandwagon_risk,
            "private_verified_good": self.private_verified_good,
            "private_verified_wolf": self.private_verified_wolf,
            "public_contradiction": self.public_contradiction,
            "recommended_action": self.recommended_action,
            "information_gain": round(self.information_gain, 3),
            "evidence_type": (
                "hard" if self.hard_wolf >= 0.4 else ("soft" if self.soft_wolf >= 0.45 else "none")
            ),
        }


@dataclass
class BeliefState:
    viewer: str
    my_role: str = "villager"
    round_no: int = 1
    game_phase: str = "discussion"
    players: Dict[str, PlayerBelief] = field(default_factory=dict)
    commitment: DiscussionCommitment = field(default_factory=DiscussionCommitment)
    private: Dict[str, Any] = field(default_factory=dict)
    version: int = 0

    def wolf_prob(self, player: str) -> float:
        if player in self.players:
            return float(self.players[player].posterior_wolf)
        return 0.3

    def get_player_belief(self, player: str) -> Optional[PlayerBelief]:
        return self.players.get(player)

    def fuse(self, analytics: Dict[str, Any]) -> "BeliefState":
        """Fuse LLM + public + analytics + private into posterior_wolf."""
        verified_good = set(self.private.get("verified_good", []))
        verified_wolf = set(self.private.get("verified_wolf", []))
        agent_prof = (analytics.get("player_profiles") or {}).get(self.viewer, {})
        trust_mentioned = list(agent_prof.get("trust_mentioned", []))
        accuse_mentioned = list(agent_prof.get("accuse_mentioned", []))

        self.commitment.public_trust_targets = trust_mentioned
        self.commitment.public_accuse_targets = accuse_mentioned

        for name, pb in self.players.items():
            behavioral = pb.hard_wolf * 0.68 + pb.soft_wolf * 0.32
            post = 0.32 * pb.llm_wolf + 0.38 * behavioral + 0.2 * (1.0 - pb.trust)
            post += 0.1 * min(1.0, pb.vote_pressure)
            if pb.bandwagon_risk:
                post = min(0.9, post + 0.06)
            if pb.speech_vote_align < 0.45:
                post = min(0.92, post + 0.1)

            if name in verified_good:
                post = min(post, 0.06)
                pb.private_verified_good = True
                pb.public_trust = max(pb.public_trust, 0.85)
                pb.recommended_action = "support"
            elif name in verified_wolf:
                post = max(0.92, post)
                pb.private_verified_wolf = True
                pb.public_trust = min(pb.public_trust, 0.05)
                pb.recommended_action = "accuse"

            pb.posterior_wolf = max(0.02, min(0.95, post))
            pb.trust = round(1.0 - pb.posterior_wolf, 3)

        self._project_private_vote_contradictions(analytics, verified_good)
        self.version += 1
        return self

    def _project_private_vote_contradictions(
        self,
        analytics: Dict[str, Any],
        verified_good: set,
    ) -> None:
        if not verified_good:
            return
        current_votes = analytics.get("current_round_votes", {}) or {}
        tally = analytics.get("current_round_tally", {}) or {}

        for good_p in verified_good:
            if good_p in self.players:
                pb = self.players[good_p]
                pb.private_verified_good = True
                pb.posterior_wolf = min(pb.posterior_wolf, 0.05)
                pb.trust = round(1.0 - pb.posterior_wolf, 3)
                pb.public_trust = max(pb.public_trust, 0.9)
                pb.recommended_action = "support"

            vote_count = tally.get(good_p, 0)
            if vote_count < 2 and good_p not in tally:
                continue

            for voter, tgt in current_votes.items():
                if tgt != good_p or voter == self.viewer:
                    continue
                if voter not in self.players:
                    continue
                vb = self.players[voter]
                vb.posterior_wolf = min(0.95, vb.posterior_wolf + 0.30 + 0.05 * vote_count)
                vb.soft_wolf = max(vb.soft_wolf, vb.posterior_wolf * 0.88)
                vb.trust = round(1.0 - vb.posterior_wolf, 3)
                vb.public_contradiction = f"voted_verified_good:{good_p}"
                vb.recommended_action = "accuse"

    def set_commitment_from_mcts(
        self,
        intent: Dict[str, Any],
        action: Tuple[str, str, str],
        decision_brief: str = "",
        round_no: Optional[int] = None,
    ) -> None:
        vote_target, target_player, _ = action
        self.commitment.round_no = round_no if round_no is not None else self.round_no
        self.commitment.stance = intent.get("stance", "neutral")
        self.commitment.focus_player = target_player or vote_target or ""
        self.commitment.vote_lean = intent.get("vote_lean", vote_target)
        self.commitment.mcts_intent = intent.get("mcts_intent", intent.get("stance", ""))
        self.commitment.decision_brief = decision_brief or ""
        self.version += 1

    def get_prior_decision(self) -> Dict[str, Any]:
        return self.commitment.to_prior_decision()

    def vote_consistency_delta(self, target: str, my_role: str) -> float:
        """
        Utility adjustment for voting target (negative = penalize).
        Same formula for all roles; wolves invert bandwagon benefit in MCTS layer.
        """
        if not target or target == "pass":
            return 0.0
        delta = 0.0
        c = self.commitment
        if c.stance == "support" and target == c.focus_player:
            delta -= 0.58
        if c.stance in ("suspect", "caution") and target == c.vote_lean:
            delta += 0.28
        if target in c.public_trust_targets:
            delta -= 0.62
        pb = self.players.get(target)
        if pb:
            if pb.private_verified_good:
                delta -= 0.5
            if pb.votes_received >= 3 and pb.hard_wolf < 0.28:
                delta -= 0.22
            if pb.vote_pressure >= 0.45 and pb.hard_wolf < 0.25:
                delta -= 0.12
            if pb.speech_vote_align < 0.45:
                delta += 0.08
        return delta

    def to_empathy_reports(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for name, pb in self.players.items():
            out[name] = pb.to_report()
        out["_belief"] = {
            "viewer": self.viewer,
            "round_no": self.round_no,
            "version": self.version,
            "commitment": self.commitment.to_prior_decision(),
        }
        if self.private:
            out["_private"] = {
                "verified_good": list(self.private.get("verified_good", [])),
                "verified_wolf": list(self.private.get("verified_wolf", [])),
                "private_facts": list(self.private.get("private_facts", [])),
            }
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "viewer": self.viewer,
            "my_role": self.my_role,
            "round_no": self.round_no,
            "game_phase": self.game_phase,
            "version": self.version,
            "private": deepcopy(self.private),
            "commitment": self.commitment.to_prior_decision(),
            "players": {
                name: {
                    "posterior_wolf": pb.posterior_wolf,
                    "llm_wolf": pb.llm_wolf,
                    "hard_wolf": pb.hard_wolf,
                    "soft_wolf": pb.soft_wolf,
                    "trust": pb.trust,
                    "public_trust": pb.public_trust,
                    "stance_to_me": pb.stance_to_me,
                    "votes_received": pb.votes_received,
                    "vote_pressure": pb.vote_pressure,
                    "speech_vote_align": pb.speech_vote_align,
                    "bandwagon_risk": pb.bandwagon_risk,
                    "private_verified_good": pb.private_verified_good,
                    "private_verified_wolf": pb.private_verified_wolf,
                    "public_contradiction": pb.public_contradiction,
                    "recommended_action": pb.recommended_action,
                    "information_gain": pb.information_gain,
                }
                for name, pb in self.players.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["BeliefState"]:
        if not data or not isinstance(data, dict):
            return None
        bs = cls(
            viewer=str(data.get("viewer", "")),
            my_role=str(data.get("my_role", "villager")),
            round_no=int(data.get("round_no", 1)),
            game_phase=str(data.get("game_phase", "discussion")),
            version=int(data.get("version", 0)),
            private=dict(data.get("private", {})),
        )
        c = data.get("commitment", {}) or {}
        bs.commitment = DiscussionCommitment(
            round_no=int(c.get("round_no", bs.round_no)),
            stance=str(c.get("stance", "neutral")),
            focus_player=str(c.get("target_player", c.get("focus_player", ""))),
            vote_lean=str(c.get("vote_lean", "pass")),
            mcts_intent=str(c.get("mcts_intent", "")),
            decision_brief=str(c.get("decision_brief", "")),
            public_trust_targets=list(c.get("public_trust_targets", [])),
            public_accuse_targets=list(c.get("public_accuse_targets", [])),
        )
        for name, pd in (data.get("players") or {}).items():
            if not str(name).startswith("Player"):
                continue
            bs.players[name] = PlayerBelief(
                player=name,
                posterior_wolf=float(pd.get("posterior_wolf", 0.3)),
                llm_wolf=float(pd.get("llm_wolf", 0.3)),
                hard_wolf=float(pd.get("hard_wolf", 0.0)),
                soft_wolf=float(pd.get("soft_wolf", 0.3)),
                trust=float(pd.get("trust", 0.7)),
                public_trust=float(pd.get("public_trust", 0.0)),
                stance_to_me=float(pd.get("stance_to_me", 0.0)),
                votes_received=int(pd.get("votes_received", 0)),
                vote_pressure=float(pd.get("vote_pressure", 0.0)),
                speech_vote_align=float(pd.get("speech_vote_align", 1.0)),
                bandwagon_risk=bool(pd.get("bandwagon_risk", False)),
                private_verified_good=bool(pd.get("private_verified_good", False)),
                private_verified_wolf=bool(pd.get("private_verified_wolf", False)),
                public_contradiction=str(pd.get("public_contradiction", "")),
                recommended_action=str(pd.get("recommended_action", "observe")),
                information_gain=float(pd.get("information_gain", 0.0)),
            )
        return bs

    @classmethod
    def build(
        cls,
        game_state,
        agent_name: str,
        empathy_reports: Dict[str, Dict[str, Any]],
        analytics: Optional[Dict[str, Any]] = None,
        private_beliefs: Optional[Dict[str, Any]] = None,
    ) -> "BeliefState":
        try:
            from .MCTS import _get_state_field
        except ImportError:
            from chatarena.MCTS import _get_state_field

        my_role = _get_state_field(game_state, "my_role", "villager")
        round_no = int(_get_state_field(game_state, "round_no", 1))
        phase = _get_state_field(game_state, "game_phase", "discussion")

        existing = get_belief_state(game_state)
        bs = cls(
            viewer=agent_name,
            my_role=my_role,
            round_no=round_no,
            game_phase=phase,
        )
        if existing and existing.viewer == agent_name and existing.round_no == round_no:
            bs.commitment = deepcopy(existing.commitment)

        private = private_beliefs or _get_state_field(game_state, "private_beliefs", {}) or {}
        bs.private = dict(private)

        try:
            from .MCTS import _ensure_report_schema
        except ImportError:
            from chatarena.MCTS import _ensure_report_schema

        for player, report in empathy_reports.items():
            if player.startswith("_") or not str(player).startswith("Player"):
                continue
            if not isinstance(report, dict):
                continue
            report = _ensure_report_schema(report)
            rp = report.get("role_probability", {})
            if not isinstance(rp, dict):
                rp = {}
            bs.players[player] = PlayerBelief(
                player=player,
                llm_wolf=float(rp.get("werewolf", 0.3)),
                hard_wolf=float(report.get("hard_wolf_prob", 0.0)),
                soft_wolf=float(report.get("soft_wolf_prob", 0.3)),
                trust=float(report.get("trust_score", 0.5)),
                public_trust=float(report.get("public_trust", 0.0)),
                stance_to_me=float(report.get("stance_to_me", 0.0)),
                votes_received=int(report.get("votes_received", 0)),
                vote_pressure=float(report.get("current_round_vote_pressure", 0.0)),
                speech_vote_align=float(report.get("speech_vote_consistency", 1.0)),
                bandwagon_risk=bool(report.get("bandwagon_risk", False)),
                recommended_action=str(report.get("recommended_action", "observe")),
                information_gain=float(report.get("information_gain", 0.0)),
                posterior_wolf=float(report.get("posterior_wolf_prob", rp.get("werewolf", 0.3))),
            )

        if analytics is None:
            try:
                from .MCTS import get_game_analytics
            except ImportError:
                from chatarena.MCTS import get_game_analytics
            analytics = get_game_analytics(game_state)

        bs.fuse(analytics)
        return bs

    def attach(self, game_state) -> None:
        payload = self.to_dict()
        if isinstance(game_state, dict):
            game_state["belief_state"] = payload
        else:
            game_state["belief_state"] = payload


class BeliefStateStore:
    """Single-writer helper to reduce cross-module state drift."""

    @staticmethod
    def sync(game_state, agent_name: str, empathy_reports: Dict[str, Dict[str, Any]], analytics: Optional[Dict[str, Any]] = None) -> Tuple[BeliefState, Dict[str, Dict[str, Any]]]:
        belief, reports = build_and_attach_belief_state(
            game_state,
            agent_name,
            empathy_reports,
            analytics=analytics,
        )
        belief.attach(game_state)
        return belief, reports

    @staticmethod
    def commit(game_state, agent_name: str, intent: Dict[str, Any], action: Tuple[str, str, str], decision_brief: str = "") -> None:
        update_commitment_on_game_state(
            game_state,
            agent_name,
            intent,
            action,
            decision_brief=decision_brief,
        )


def get_belief_state(game_state) -> Optional[BeliefState]:
    try:
        from .MCTS import _get_state_field
    except ImportError:
        from chatarena.MCTS import _get_state_field
    raw = _get_state_field(game_state, "belief_state", None)
    return BeliefState.from_dict(raw)


def build_and_attach_belief_state(
    game_state,
    agent_name: str,
    empathy_reports: Dict[str, Dict[str, Any]],
    analytics: Optional[Dict[str, Any]] = None,
) -> Tuple[BeliefState, Dict[str, Dict[str, Any]]]:
    """Single entry: fuse beliefs and return synced empathy reports."""
    private = None
    try:
        from .MCTS import _get_state_field
    except ImportError:
        from chatarena.MCTS import _get_state_field
    private = _get_state_field(game_state, "private_beliefs", None)

    cached = _get_state_field(game_state, "belief_state", None)
    if isinstance(cached, dict):
        cached_viewer = str(cached.get("viewer", ""))
        cached_round = int(cached.get("round_no", 0))
        current_round = int(_get_state_field(game_state, "round_no", 1))
        if cached_viewer == agent_name and cached_round == current_round:
            belief = BeliefState.from_dict(cached)
            if belief is not None:
                if analytics is None:
                    try:
                        from .MCTS import get_game_analytics
                    except ImportError:
                        from chatarena.MCTS import get_game_analytics
                    analytics = get_game_analytics(game_state)
                belief.private = dict(private or belief.private)
                belief.fuse(analytics)
                belief.attach(game_state)
                return belief, belief.to_empathy_reports()

    belief = BeliefState.build(
        game_state, agent_name, empathy_reports, analytics=analytics, private_beliefs=private
    )
    belief.attach(game_state)
    return belief, belief.to_empathy_reports()


def update_commitment_on_game_state(
    game_state,
    agent_name: str,
    intent: Dict[str, Any],
    action: Tuple[str, str, str],
    decision_brief: str = "",
) -> None:
    belief = get_belief_state(game_state)
    if belief is None or belief.viewer != agent_name:
        try:
            from .MCTS import _get_state_field
        except ImportError:
            from chatarena.MCTS import _get_state_field
        belief = BeliefState(
            viewer=agent_name,
            my_role=_get_state_field(game_state, "my_role", "villager"),
            round_no=int(_get_state_field(game_state, "round_no", 1)),
        )
    belief.set_commitment_from_mcts(
        intent,
        action,
        decision_brief=decision_brief,
        round_no=belief.round_no,
    )
    belief.attach(game_state)
