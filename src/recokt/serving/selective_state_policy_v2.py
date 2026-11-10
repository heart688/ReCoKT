"""Selective component-state retention during withdrawal.

The injected guard is an explicit contract: the stream uses only its component's
immutable parameters and the student's permitted history, with no aggregate
prediction/state feedback. A shared frozen base is allowed. An unknown stream
falls back to fresh replay. Artifact versions are checked against the manifest;
in-place parameter mutation is outside this frozen-inference API contract.

"""
from __future__ import annotations

import hashlib
import json
import math

from .managed_serving_bundle_v1 import ManagedServingBundle


def _component_version(entry):
    return hashlib.sha256(json.dumps(entry, sort_keys=True, separators=(",", ":"))
                          .encode()).hexdigest()


def _route_matches(states, route):
    if len(states) != len(route):
        return False
    for (name, weight, _), (expected_name, expected_weight) in zip(states, route):
        if name != expected_name or not math.isclose(float(weight), float(expected_weight),
                                                     rel_tol=0.0, abs_tol=1e-12):
            return False
    return True


def _clear_stream(state):
    # FactorStream.clear resets recurrent/KC state but not adapted token features.
    # Also clear that component-dependent cache before dropping the managed ref.
    try:
        state.clear()
    finally:
        cache = getattr(state, "token_cache", None)
        if cache is not None:
            cache.clear()


def exact_stream_type_guard(stream_type):
    """Opt in only a reviewed, component-isolated stream implementation.

    This checks type identity, not semantic independence; the caller must have
    verified the supplied implementation. In particular, do not use this helper
    to certify an arbitrary fused/shared learned-state implementation.
    """
    return lambda state: type(state) is stream_type


class SelectiveStateMixin:
    state_policy_version = "component-state-retention-v2"

    def __init__(self, *args, state_reuse_guard=None, **kwargs):
        self._state_reuse_guard = state_reuse_guard or (lambda state: False)
        self._last_state_policy_summary = {}
        super().__init__(*args, **kwargs)

    def open_session(self, session_id, school, stream_factory):
        super().open_session(session_id, school, stream_factory)
        entry = self.sessions[session_id]
        entry["_state_versions_v2"] = {
            name: {"state_id": id(state), "component": _component_version(self.active[name])}
            for name, _, state in entry["states"]
        }

    def close_session(self, session_id):
        entry = self.sessions.pop(session_id)
        for _, _, state in entry["states"]:
            _clear_stream(state)

    def _can_retain(self, entry, old_route, new_route, reference_counts):
        states = entry["states"]
        if not _route_matches(states, old_route):
            return False
        by_name = {name: state for name, _, state in states}
        if len(by_name) != len(states):
            return False
        versions = entry.get("_state_versions_v2", {})
        for name, _ in new_route:
            state = by_name.get(name)
            saved = versions.get(name)
            if (state is None or saved is None or reference_counts[id(state)] != 1
                    or saved["state_id"] != id(state)
                    or saved["component"] != _component_version(self.active[name])):
                return False
            try:
                if not self._state_reuse_guard(state):
                    return False
            except Exception:
                return False
        return True

    def withdraw(self, school):
        """Revoke in-memory sources; the existing managed transaction owns disk IO."""
        school = str(school)
        if school in self.withdrawn:
            raise ValueError("already withdrawn")
        removed = sorted(name for name, entry in self.active.items()
                         if school in map(str, entry["members"]))
        if not removed:
            raise ValueError("school has no active components")

        old_routes = {sid: self.route(entry["school"])
                      for sid, entry in self.sessions.items()}
        reference_counts = {}
        for entry in self.sessions.values():
            for _, _, state in entry["states"]:
                reference_counts[id(state)] = reference_counts.get(id(state), 0) + 1
        for name in removed:
            self.active.pop(name)
            self.resident.pop(name, None)
        self.withdrawn.add(school)

        replay, closed, preserved, unchanged = [], [], [], []
        cleared_states = 0
        for session_id, entry in list(self.sessions.items()):
            if str(entry["school"]) == school:
                closed.append(session_id)
                cleared_states += len(entry["states"])
                self.close_session(session_id)
                continue
            new_route = self.route(entry["school"])
            if not self._can_retain(entry, old_routes[session_id], new_route, reference_counts):
                replay.append(session_id)
                cleared_states += len(entry["states"])
                self.close_session(session_id)
                continue
            old_states = entry["states"]
            keep = {name for name, _ in new_route}
            by_name = {name: state for name, _, state in old_states}
            for name, _, state in old_states:
                if name not in keep:
                    _clear_stream(state)
                    cleared_states += 1
            entry["states"] = [(name, weight, by_name[name]) for name, weight in new_route]
            entry["_state_versions_v2"] = {
                name: entry["_state_versions_v2"][name] for name, _ in new_route
            }
            (unchanged if _route_matches(old_states, new_route) else preserved).append(session_id)

        if any(school in map(str, entry["members"]) for entry in self.active.values()):
            raise RuntimeError("stale route")
        summary = {"sessions_state_preserved": preserved, "sessions_unchanged": unchanged,
                   "states_cleared": cleared_states, "state_policy": self.state_policy_version}
        self._last_state_policy_summary = summary
        return {"school": school, "deactivated": removed,
                "sessions_replay_required": replay, "owner_sessions_closed": closed,
                "plugin_owned_states_cleared": True, "storage_deleted": False,
                "concurrency_managed": False, **summary}


def selective_plugin_type(base_plugin_type):
    """Create an injectable subclass of DirectDKTPlugin or a compatible adapter."""
    class SelectivePlugin(SelectiveStateMixin, base_plugin_type):
        pass
    SelectivePlugin.__name__ = "Selective" + base_plugin_type.__name__
    return SelectivePlugin


def direct_plugin_type():
    from .direct_dkt_plugin_v1 import DirectDKTPlugin
    return selective_plugin_type(DirectDKTPlugin)


def make_quantized_dkt_factory(*, state_reuse_guard):
    from .managed_quant_serving_factory_v1 import QuantizedProductionAdapter
    plugin_type = selective_plugin_type(QuantizedProductionAdapter)
    return lambda folder: plugin_type(folder, "dkt", state_reuse_guard=state_reuse_guard)


class SelectiveManagedServingBundle(ManagedServingBundle):
    """Original durable transaction/restart behavior plus route/weight auditing."""
    def _check_plugin(self):
        super()._check_plugin()
        if not isinstance(self._plugin, SelectiveStateMixin):
            raise ValueError("selective managed bundle requires an explicitly injected selective plugin")

    def audit(self):
        with self._serving():
            result = super().audit()
            mismatches = []
            for session_id, entry in self._plugin.sessions.items():
                if not _route_matches(entry["states"], self._plugin.route(entry["school"])):
                    mismatches.append(session_id)
            if mismatches:
                raise RuntimeError("session route or weight mismatch: " + repr(mismatches))
            return {**result, "session_route_mismatches": 0}

    def withdraw(self, school):
        with self._serving():
            result = super().withdraw(school)
            if result["idempotent"]:
                return {**result, "sessions_state_preserved": [], "sessions_unchanged": [],
                        "sessions_replay_required": [], "owner_sessions_closed": [],
                        "states_cleared": 0, "state_policy": SelectiveStateMixin.state_policy_version}
            return {**result, **self._plugin._last_state_policy_summary}
