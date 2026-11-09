"""Control/API tests only: real legacy session API, deterministic mock streams.

No torch, real weights, training, server deployment, or benchmark runs needed.
"""
from collections import OrderedDict
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from recokt.serving.managed_serving_bundle_v1 import clone_serving_bundle
from recokt.serving.selective_state_policy_v2 import (SelectiveManagedServingBundle,
                                                       exact_stream_type_guard, selective_plugin_type)
from recokt.serving.shared_dkt_plugin_v3 import RecoDKTPlugin

def _legacy_class():
    return RecoDKTPlugin


class MemoryPlugin(_legacy_class()):
    def __init__(self, folder):
        self.folder = Path(folder)
        self.manifest = json.loads((self.folder / "manifest.json").read_text())
        self.active = dict(self.manifest["components"])
        self.withdrawn = set(self.manifest["initially_withdrawn"])
        self.resident, self.sessions = OrderedDict(), {}

    def component(self, name):
        if name not in self.active:
            raise ValueError("inactive component")
        self.resident[name] = (name, None)
        return self.resident[name]


class MockStream:
    """Independent component recurrence with a component-dependent token cache."""
    def __init__(self, component):
        self.component = component
        self.coefficient = 0.02 * sum(map(ord, component))
        self.token_cache = {}
        self.total_updates = 0
        self.clear()

    def clear(self):
        self.h = self.c = self.current = None
        self.latest, self.counts = {}, {}
        self.pos = 0

    def observe(self, skill, response):
        token = (skill, response)
        x = self.token_cache.setdefault(token, self.coefficient + skill + response)
        self.h = 0.7 * (self.h or 0) + x
        self.c = 0.6 * (self.c or 0) + self.h
        self.current = self.h
        self.latest[skill] = (self.h, response, self.pos)
        self.counts[skill] = self.counts.get(skill, 0) + 1
        self.pos += 1
        self.total_updates += 1

    def predict(self, skill):
        if self.h is None:
            raise ValueError("empty history")
        anchor = self.latest.get(skill, (0, 0, 0))[0]
        return 1 / (1 + math.exp(-(self.h + 0.1 * anchor - skill) / 10))


SelectiveMemoryPlugin = selective_plugin_type(MemoryPlugin)
HISTORY = [(1, 1), (2, 0), (1, 0), (3, 1)]


class StatePolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        specs = {name: [name] for name in "ABCD"}
        specs.update({pair: list(pair) for pair in ("AB", "BC", "CD", "AD")})
        components = {}
        for name, members in specs.items():
            data = ("immutable mock parameters " + name).encode()
            filename = name + ".bin"
            (self.source / filename).write_bytes(data)
            components[name] = {"members": members, "file": filename,
                                "sha256": hashlib.sha256(data).hexdigest()}
        self.manifest = {"status": "complete", "initially_withdrawn": [],
                         "components": components}
        (self.source / "manifest.json").write_text(json.dumps(self.manifest))
        self.services = []
        self.serial = 0

    def tearDown(self):
        for service in self.services:
            service.close()
        self.temp.cleanup()

    def service(self, guard=None, folder=None):
        if folder is None:
            self.serial += 1
            folder = clone_serving_bundle(self.source, self.root / ("owned" + str(self.serial)))
        if guard is None:
            guard = exact_stream_type_guard(MockStream)
        factory = lambda path: SelectiveMemoryPlugin(path, state_reuse_guard=guard)
        service = SelectiveManagedServingBundle(folder, factory)
        self.services.append(service)
        return service

    def open(self, service, sid, school):
        service.open_session(sid, school, lambda component, head: MockStream(component))
        for skill, response in HISTORY:
            service.observe_session(sid, skill, response)
        return {name: state for name, _, state in service._plugin.sessions[sid]["states"]}

    def assert_clean_continuation(self, service, sid, school):
        # An independent empty post-withdrawal session reconstructs the same history.
        fresh = "clean-" + sid
        self.open(service, fresh, school)
        for skill, response in [(1, 1), (4, 0), (2, 1)]:
            self.assertEqual(service.predict_session(sid, skill),
                             service.predict_session(fresh, skill))
            service.observe_session(sid, skill, response)
            service.observe_session(fresh, skill, response)
        service.close_session(fresh)

    def test_affected_unaffected_owner_and_clean_continuation(self):
        service = self.service()
        affected = self.open(service, "affected", "A")
        unaffected = self.open(service, "unaffected", "D")
        owner = self.open(service, "owner", "B")
        snapshots = {name: copy.deepcopy(state.__dict__) for name, state in affected.items()}
        result = service.withdraw("B")
        self.assertEqual(result["sessions_state_preserved"], ["affected"])
        self.assertEqual(result["sessions_unchanged"], ["unaffected"])
        self.assertEqual(result["owner_sessions_closed"], ["owner"])
        self.assertEqual(result["sessions_replay_required"], [])
        self.assertNotIn("owner", service._plugin.sessions)
        remaining = service._plugin.sessions["affected"]["states"]
        self.assertEqual([(n, w) for n, w, _ in remaining], [("A", 0.7), ("AD", 0.3)])
        for name, _, state in remaining:
            self.assertIs(state, affected[name])
            self.assertEqual(state.__dict__, snapshots[name])
        for name, _, state in service._plugin.sessions["unaffected"]["states"]:
            self.assertIs(state, unaffected[name])
        for state in [affected["AB"], *owner.values()]:
            self.assertIsNone(state.h)
            self.assertEqual(state.latest, {})
            self.assertEqual(state.token_cache, {})
        for name in ("B", "AB", "BC"):
            self.assertFalse((service.folder / (name + ".bin")).exists())
        self.assertEqual(service.audit()["session_route_mismatches"], 0)
        self.assert_clean_continuation(service, "affected", "A")

    def test_successive_withdrawals_retain_local_object(self):
        service = self.service()
        states = self.open(service, "a", "A")
        service.withdraw("B")
        result = service.withdraw("D")
        self.assertEqual(result["sessions_replay_required"], [])
        self.assertEqual(service._plugin.sessions["a"]["states"], [("A", 1.0, states["A"])])
        self.assertEqual(states["A"].total_updates, len(HISTORY))
        self.assertEqual(states["AD"].token_cache, {})
        self.assert_clean_continuation(service, "a", "A")

    def test_audit_rejects_stale_weights(self):
        service = self.service()
        self.open(service, "a", "A")
        service.withdraw("B")
        row = service._plugin.sessions["a"]
        name, _, state = row["states"][0]
        row["states"][0] = (name, 0.4, state)
        with self.assertRaisesRegex(RuntimeError, "route or weight mismatch"):
            service.audit()

    def test_idempotency_and_restart(self):
        service = self.service()
        states = self.open(service, "a", "A")
        service.withdraw("B")
        result = service.withdraw("B")
        self.assertTrue(result["idempotent"])
        self.assertEqual(result["states_cleared"], 0)
        self.assertIs(service._plugin.sessions["a"]["states"][0][2], states["A"])
        folder = service.folder
        service.close()
        restarted = self.service(folder=folder)
        self.assertEqual(restarted.route("A"), [("A", 0.7), ("AD", 0.3)])
        self.assertEqual(restarted._plugin.sessions, {})
        self.assertEqual(restarted.audit()["active_components"], 5)

    def test_unknown_stream_falls_back_to_correct_replay(self):
        service = self.service(guard=lambda state: False)
        old = self.open(service, "a", "A")
        result = service.withdraw("B")
        self.assertEqual(result["sessions_replay_required"], ["a"])
        self.assertNotIn("a", service._plugin.sessions)
        for state in old.values():
            self.assertEqual(state.token_cache, {})
        fresh = self.open(service, "a", "A")
        self.assertTrue(all(state.total_updates == len(HISTORY) for state in fresh.values()))
        self.assert_clean_continuation(service, "a", "A")

    def test_component_version_mismatch_falls_back(self):
        service = self.service()
        self.open(service, "a", "A")
        # A stale state certificate simulates a parameter-generation change.
        service._plugin.sessions["a"]["_state_versions_v2"]["A"]["component"] = "old-version"
        result = service.withdraw("B")
        self.assertEqual(result["sessions_replay_required"], ["a"])
        self.assertNotIn("a", service._plugin.sessions)

    def test_cross_session_stream_alias_forces_both_sessions_to_replay(self):
        service = self.service()
        shared = MockStream("AD")

        def aliased_factory(component, head):
            return shared if component == "AD" else MockStream(component)

        old = {}
        for sid, school in [("a", "A"), ("d", "D")]:
            service.open_session(sid, school, aliased_factory)
            for skill, response in HISTORY:
                service.observe_session(sid, skill, response)
            entry = service._plugin.sessions[sid]
            old[sid] = {name: state for name, _, state in entry["states"]}
            # The factory created the alias before certification; fallback must
            # detect sharing, not merely reject a replaced stream identity.
            self.assertEqual(entry["_state_versions_v2"]["AD"]["state_id"], id(shared))
        self.assertIs(old["a"]["AD"], old["d"]["AD"])

        result = service.withdraw("B")
        # D's route is unchanged, but A's closure clears their shared object.
        # D must still replay even after A has been removed from the session map.
        self.assertEqual(result["sessions_replay_required"], ["a", "d"])
        self.assertEqual(result["sessions_state_preserved"], [])
        self.assertEqual(result["sessions_unchanged"], [])
        self.assertEqual(service._plugin.sessions, {})
        for states in old.values():
            for state in states.values():
                self.assertIsNone(state.h)
                self.assertEqual(state.latest, {})
                self.assertEqual(state.token_cache, {})
        self.assertEqual(service.audit()["session_route_mismatches"], 0)

        fresh_a = self.open(service, "a", "A")
        fresh_d = self.open(service, "d", "D")
        self.assertIsNot(fresh_a["AD"], fresh_d["AD"])
        self.assert_clean_continuation(service, "a", "A")
        self.assert_clean_continuation(service, "d", "D")

    def test_raising_reuse_guard_falls_back_and_commits_consistently(self):
        def raising_guard(state):
            raise RuntimeError("reuse check unavailable")

        service = self.service(guard=raising_guard)
        old = self.open(service, "a", "A")
        result = service.withdraw("B")
        self.assertFalse(result["idempotent"])
        self.assertEqual(result["sessions_replay_required"], ["a"])
        self.assertEqual(result["sessions_state_preserved"], [])
        self.assertNotIn("a", service._plugin.sessions)
        for state in old.values():
            self.assertIsNone(state.h)
            self.assertEqual(state.latest, {})
            self.assertEqual(state.token_cache, {})

        persisted = json.loads((service.folder / "manifest.json").read_text())
        self.assertEqual(persisted["components"], service._plugin.active)
        self.assertEqual(persisted["initially_withdrawn"], ["B"])
        self.assertFalse((service.folder / "withdrawal.pending.json").exists())
        for name in ("B", "AB", "BC"):
            self.assertFalse((service.folder / (name + ".bin")).exists())
        audit = service.audit()
        self.assertEqual(audit["active_components"], 5)
        self.assertEqual(audit["affected_sessions_remaining"], 0)
        self.assertEqual(audit["session_route_mismatches"], 0)
        self.open(service, "a", "A")
        self.assert_clean_continuation(service, "a", "A")

    def test_interrupted_transaction_uses_original_restart_recovery(self):
        service = self.service()
        folder = service.folder
        manifest = copy.deepcopy(service._manifest)
        service.close()
        removed = [name for name, entry in manifest["components"].items() if "B" in entry["members"]]
        removed_files = [manifest["components"].pop(name)["file"] for name in removed]
        manifest["initially_withdrawn"] = ["B"]
        manifest["managed_serving"]["generation"] += 1
        transaction = {"bundle_id": manifest["managed_serving"]["bundle_id"],
                       "manifest": manifest, "removed_files": removed_files}
        (folder / "withdrawal.pending.json").write_text(json.dumps(transaction))
        restarted = self.service(folder=folder)
        self.assertEqual(restarted.route("A"), [("A", 0.7), ("AD", 0.3)])
        self.assertFalse((folder / "withdrawal.pending.json").exists())
        self.assertTrue(all(not (folder / name).exists() for name in removed_files))
        self.assertEqual(restarted.audit()["affected_sessions_remaining"], 0)

    def test_isolated_stale_cache_diagnostic_is_rejected(self):
        service = self.service()
        old = self.open(service, "a", "A")
        retired = old["AB"]
        service.withdraw("B")
        # Deliberate corruption stays in this isolated test. No production gate
        # is disabled, and this configuration is never called a safe policy.
        service._plugin.sessions["a"]["states"].append(("AB", 0.0, retired))
        with self.assertRaisesRegex(RuntimeError, "route or weight mismatch"):
            service.audit()


if __name__ == "__main__":
    unittest.main(verbosity=2)
