#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation logger for Werewolf LLM experiments.

This module records and computes the following metrics WITHOUT performance stats:
- Win rates: overall and per role (werewolf/villager/special)
- Voting accuracy: lynch correctness per day; per-voter hit rate
- Survival rounds: averages and distributions by role/system
- Empathy accuracy: compare llm_empathy_extract outputs (role_probability, stance_to_me)
  against truth and observable behavior
- Speech consistency: speech_style vs target susceptibility/PAD
- Vote-swing (persuasion): whether target's later vote aligns after speaker's speech
- Seer utilization: seer correct peek → next day lynch/speech direction

Integrate by calling the on_* methods at appropriate points in your game loop.
Call finalize_and_export() at game end to compute and dump metrics as JSON or TXT.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple


def _safe_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


@dataclass
class VoteEvent:
    day: int
    voter: str
    target: str


@dataclass
class SpeechEvent:
    day: int
    speaker: str
    target_player: str
    speech_style: str
    planned_vote_target: Optional[str] = None


@dataclass
class LynchEvent:
    day: int
    lynched: str


@dataclass
class SeerPeek:
    night: int
    seer: str
    target: str
    is_wolf: bool


@dataclass
class EmpathySnapshot:
    round_no: int
    player_reports: Dict[str, Dict[str, Any]]  # output of llm_empathy_extract["player_reports"]


@dataclass
class EvaluationSession:
    system_name: str
    model_name: str
    seed: int
    game_id: str

    # Static truth
    players: List[str] = field(default_factory=list)
    roles: Dict[str, str] = field(default_factory=dict)  # {player: role}

    # Timeline
    votes: List[VoteEvent] = field(default_factory=list)
    speeches: List[SpeechEvent] = field(default_factory=list)
    lynches: List[LynchEvent] = field(default_factory=list)
    seer_peeks: List[SeerPeek] = field(default_factory=list)
    empathy_snaps: List[EmpathySnapshot] = field(default_factory=list)

    # Runtime state
    current_day: int = 1
    current_night: int = 0
    alive_players: List[str] = field(default_factory=list)
    survival_round: Dict[str, int] = field(default_factory=dict)  # last round number the player is alive at day end
    winner: Optional[str] = None  # "werewolves" or "villagers"

    # Public API: record events
    def on_game_start(self, players: List[str], roles: Dict[str, str]):
        self.players = list(players)
        self.roles = dict(roles)
        self.alive_players = list(players)
        for p in players:
            self.survival_round[p] = 0

    def on_empathy_snapshot(self, round_no: int, player_reports: Dict[str, Dict[str, Any]]):
        self.empathy_snaps.append(EmpathySnapshot(round_no=round_no, player_reports=player_reports or {}))

    def on_day_begin(self, day: int):
        self.current_day = day
        for p in list(self.alive_players):
            self.survival_round[p] = max(self.survival_round.get(p, 0), day)

    def on_speech(self, speaker: str, target_player: str, speech_style: str, planned_vote_target: Optional[str] = None):
        self.speeches.append(SpeechEvent(day=self.current_day, speaker=speaker, target_player=target_player,
                                         speech_style=speech_style, planned_vote_target=planned_vote_target))

    def on_vote(self, voter: str, target: str):
        self.votes.append(VoteEvent(day=self.current_day, voter=voter, target=target))

    def on_day_end_lynch(self, lynched: str):
        self.lynches.append(LynchEvent(day=self.current_day, lynched=lynched))
        if lynched in self.alive_players:
            self.alive_players = [p for p in self.alive_players if p != lynched]

    def on_night_begin(self, night: int):
        self.current_night = night

    def on_seer_peek(self, seer: str, target: str, is_wolf: bool):
        self.seer_peeks.append(SeerPeek(night=self.current_night, seer=seer, target=target, is_wolf=is_wolf))

    def on_game_end(self, winner: str):
        self.winner = winner  # "werewolves" or "villagers"

    # Metrics computation
    def compute_metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}

        # 1) Win rates (per game: 0/1; aggregate out of scope here)
        metrics["winner"] = self.winner
        metrics["winner_is_wolves"] = 1 if self.winner == "werewolves" else 0
        metrics["role_win"] = self._role_win_flags()

        # 2) Voting accuracy
        lynch_acc, voter_hit = self._voting_accuracy()
        metrics["lynch_correct"] = lynch_acc  # [{day, lynched, is_wolf(0/1)}]
        metrics["per_voter_hit_rate"] = voter_hit  # {player: hit_rate}

        # 3) Survival rounds distribution
        metrics["survival_rounds_by_role"] = self._survival_by_role()

        # 4) Empathy accuracy
        metrics["empathy_accuracy"] = self._empathy_accuracy()

        # 5) Speech consistency
        metrics["speech_consistency"] = self._speech_consistency()

        # 6) Vote-swing (persuasion)
        metrics["vote_swing"] = self._vote_swing()

        # 7) Seer utilization
        metrics["seer_utilization"] = self._seer_utilization()

        return metrics

    def _role_win_flags(self) -> Dict[str, int]:
        # werewolves win flag
        return {
            "werewolves": 1 if self.winner == "werewolves" else 0,
            "villagers": 1 if self.winner == "villagers" else 0
        }

    def _voting_accuracy(self) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        # Day-lynch correctness
        lynch_acc: List[Dict[str, Any]] = []
        for ev in self.lynches:
            role = self.roles.get(ev.lynched, "")
            lynch_acc.append({
                "day": ev.day,
                "lynched": ev.lynched,
                "is_wolf": 1 if role == "werewolf" else 0
            })

        # Per-voter hit rate (votes that target true werewolf)
        votes_by_voter: Dict[str, List[int]] = {}
        for v in self.votes:
            is_hit = 1 if self.roles.get(v.target, "") == "werewolf" else 0
            votes_by_voter.setdefault(v.voter, []).append(is_hit)
        voter_hit = {voter: (sum(xs) / len(xs) if xs else 0.0) for voter, xs in votes_by_voter.items()}
        return lynch_acc, voter_hit

    def _survival_by_role(self) -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = {"werewolf": [], "villager": [], "seer": [], "witch": [], "guard": []}
        for p, r in self.roles.items():
            out.setdefault(r, []).append(self.survival_round.get(p, 0))
        return out

    def _latest_empathy(self) -> Dict[str, Dict[str, Any]]:
        if not self.empathy_snaps:
            return {}
        return self.empathy_snaps[-1].player_reports

    def _empathy_accuracy(self) -> Dict[str, Any]:
        reports = self._latest_empathy()
        if not reports:
            return {"true_role_prob_mean": None, "stance_alignment": None}

        # True role probability: average predicted prob of the ground-truth role
        probs: List[float] = []
        for player, role in self.roles.items():
            rp = reports.get(player, {})
            prob = _safe_get(rp, "role_probability", role, default=None)
            if isinstance(prob, (int, float)):
                probs.append(float(prob))
        true_role_prob_mean = sum(probs) / len(probs) if probs else None

        # Stance alignment: if stance_to_me > 0 then player should not vote against "me"; we approximate
        # here by pairwise consistency: for each vote voter->target, compare stance_to_me in target's report
        align_marks: List[int] = []
        align_total: int = 0
        for v in self.votes:
            voter_rp = reports.get(voter := v.voter, {})
            target_rp = reports.get(v.target, {})
            # If voter has positive stance_to_me in target's view (target likes voter), consider alignment if voter did not attack target.
            stance_to_voter = _safe_get(target_rp, "stance_to_me", default=None)
            if isinstance(stance_to_voter, (int, float)):
                align_total += 1
                if stance_to_voter > 0:
                    # friendly relation → voting against counts as misalignment
                    align_marks.append(0 if v.target == voter else 1)  # weak proxy
                else:
                    align_marks.append(1)
        stance_alignment = (sum(align_marks) / align_total) if align_total else None

        return {
            "true_role_prob_mean": true_role_prob_mean,
            "stance_alignment": stance_alignment
        }

    def _speech_consistency(self) -> Dict[str, Any]:
        reports = self._latest_empathy()
        if not reports:
            return {"mean_match": None, "details": []}

        def style_match(style: str, target_report: Dict[str, Any]) -> float:
            sus = _safe_get(target_report, "susceptibility", default={}) or {}
            emo = _safe_get(target_report, "emotion", default={}) or {}
            arousal = float(emo.get("arousal", 0.5))
            logic = float(sus.get("logic", 0.5))
            consensus = float(sus.get("consensus", 0.5))
            authority = float(sus.get("authority", 0.5))
            if style == "soothe":
                return 1.0 if arousal >= 0.6 else 0.0
            if style == "evidence":
                return 1.0 if logic >= 0.6 else 0.0
            if style == "align":
                return 1.0 if consensus >= 0.6 else 0.0
            if style == "counter":
                return 1.0 if authority >= 0.5 else 0.0
            if style == "redirect":
                return 1.0 if arousal >= 0.5 and consensus >= 0.5 else 0.0
            if style == "bargain":
                reciprocity = float(sus.get("reciprocity", 0.5))
                commitment = float(sus.get("commitment", 0.5))
                return 1.0 if (reciprocity + commitment) / 2.0 >= 0.6 else 0.0
            if style == "humor":
                return 1.0 if authority <= 0.5 else 0.0
            # ambiguous: always weakly consistent
            return 0.5

        details: List[Dict[str, Any]] = []
        matches: List[float] = []
        for sp in self.speeches:
            tr = reports.get(sp.target_player, {})
            score = style_match(sp.speech_style, tr)
            matches.append(score)
            details.append({
                "day": sp.day,
                "speaker": sp.speaker,
                "target": sp.target_player,
                "style": sp.speech_style,
                "match": score
            })
        return {"mean_match": (sum(matches) / len(matches)) if matches else None, "details": details}

    def _vote_swing(self) -> Dict[str, Any]:
        # For each speech, if the target's final vote that day equals speaker's planned vote target, count as success.
        swings: List[int] = []
        count: int = 0
        # Build final vote per player per day
        final_vote: Dict[Tuple[int, str], str] = {}
        for v in self.votes:
            final_vote[(v.day, v.voter)] = v.target  # last one stands as final
        for sp in self.speeches:
            if sp.planned_vote_target and (sp.day, sp.target_player) in final_vote:
                count += 1
                swings.append(1 if final_vote[(sp.day, sp.target_player)] == sp.planned_vote_target else 0)
        return {"success_rate": (sum(swings) / count) if count else None, "n": count}

    def _seer_utilization(self) -> Dict[str, Any]:
        if not self.seer_peeks:
            return {"cases": 0, "success_rate": None}
        # A simple proxy: if peeked target is wolf and the next day's lynch equals that target → success.
        cases = 0
        success = 0
        # Map lynch by day
        lynch_by_day: Dict[int, str] = {l.day: l.lynched for l in self.lynches}
        for pk in self.seer_peeks:
            next_day = pk.night + 1
            if pk.is_wolf:
                cases += 1
                if lynch_by_day.get(next_day) == pk.target:
                    success += 1
        return {"cases": cases, "success_rate": (success / cases) if cases else None}

    # Export
    def finalize_and_export(self, out_dir: str, fmt: str = "json") -> str:
        os.makedirs(out_dir, exist_ok=True)
        metrics = self.compute_metrics()
        base = f"{self.game_id}_{self.system_name}_{self.model_name}_seed{self.seed}"
        path = os.path.join(out_dir, f"{base}.{'json' if fmt == 'json' else 'txt'}")
        if fmt == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "meta": {
                        "game_id": self.game_id,
                        "system": self.system_name,
                        "model": self.model_name,
                        "seed": self.seed
                    },
                    "roles": self.roles,
                    "metrics": metrics,
                    "lynches": [l.__dict__ for l in self.lynches],
                    "votes": [v.__dict__ for v in self.votes],
                    "speeches": [s.__dict__ for s in self.speeches]
                }, f, ensure_ascii=False, indent=2)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"game_id: {self.game_id}\n")
                f.write(f"system: {self.system_name}\n")
                f.write(f"model: {self.model_name}\n")
                f.write(f"seed: {self.seed}\n\n")
                f.write("roles:\n")
                for p, r in self.roles.items():
                    f.write(f"  - {p}: {r}\n")
                f.write("\nmetrics:\n")
                f.write(json.dumps(self.compute_metrics(), ensure_ascii=False, indent=2))
                f.write("\n")
        return path


