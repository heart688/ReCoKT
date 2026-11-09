"""Persistent withdrawal for an explicitly owned copy of a KT serving bundle.

Only files listed by the copied manifest are owned. Training runs, research
checkpoints, input records, and backups outside this directory are untouched.
This manages filesystem names and plugin-owned references, not media erasure.
All serving calls must use ManagedServingBundle; raw model objects are not exposed.
A bundle has one live owner process. Its calls are serialized across threads.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path, PurePosixPath
import copy
import fcntl
import hashlib
import json
import os
import shutil
import stat
import threading
import time
import uuid

SCHEMA = 'recokt-managed-serving-v1'
CONTROL = {'manifest.json', 'ownership.json', '.serving.lock', 'withdrawal.pending.json'}


def _sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _relative(value):
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError('invalid artifact path')
    p = PurePosixPath(value)
    if p.is_absolute() or any(part in ('', '.', '..') for part in value.split('/')):
        raise ValueError('artifact path must be a normalized relative path')
    if str(p) in CONTROL or p.parts[0].startswith('.'):
        raise ValueError('artifact conflicts with serving metadata')
    return str(p)


def _safe_file(root, relative, *, missing=False, single_link=True):
    p = root
    parts = PurePosixPath(relative).parts
    for i, part in enumerate(parts):
        p = p / part
        try:
            info = p.lstat()
        except FileNotFoundError:
            if missing and i == len(parts) - 1:
                return p
            raise
        if stat.S_ISLNK(info.st_mode):
            raise ValueError('symlinks are not permitted in a serving bundle')
        if i < len(parts) - 1:
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError('artifact ancestor is not a directory')
        elif not stat.S_ISREG(info.st_mode) or (single_link and info.st_nlink != 1):
            raise ValueError('artifact must be a regular, unshared file')
    return p


def _read(root, name):
    return json.loads(_safe_file(root, name).read_text())


def _sync_dir(directory):
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write(root, name, value):
    # Deterministic staging names make an interrupted metadata write discoverable.
    staged = root / ('.' + name + '.tmp')
    if staged.exists() or staged.is_symlink():
        raise ValueError('unexpected metadata staging file')
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(staged, flags, 0o600)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(staged, root / name)
        _sync_dir(root)
    except BaseException:
        # The original committed name remains authoritative if replace did not occur.
        if staged.exists():
            staged.unlink()
        raise


def _files(manifest):
    result = {}
    for name, entry in manifest['components'].items():
        members = entry['members']
        if not members or len(set(map(str, members))) != len(members):
            raise ValueError('invalid component sources')
        for field in ['file']:
            rel = _relative(entry[field])
            if rel in result:
                raise ValueError('component artifacts must not share files')
            result[rel] = name
        # Optional persistent head, optimizer, checkpoint, or state files.
        for artifact in entry.get('artifacts', []):
            rel = _relative(artifact)
            if rel in result:
                raise ValueError('duplicate component artifact')
            result[rel] = name
    if manifest.get('base_file'):
        rel = _relative(manifest['base_file'])
        if rel in result:
            raise ValueError('shared base overlaps a component')
        result[rel] = None
    return result


def _inventory(root, allowed):
    found = set()
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs:
            if (Path(directory) / name).is_symlink():
                raise ValueError('symlink directory in serving bundle')
        for name in files:
            rel = (Path(directory) / name).relative_to(root).as_posix()
            _safe_file(root, rel)
            if rel not in allowed:
                raise ValueError('unregistered serving artifact: ' + rel)
            found.add(rel)
    return found


def clone_serving_bundle(source, destination):
    """Copy only declared serving files into a new owned directory.

    The source is read-only. Copies are independent files, never hard links.
    Existing destinations are rejected; this never adopts a research directory.
    """
    source = Path(source).absolute()
    destination = Path(destination).absolute()
    if source.is_symlink() or destination.exists() or destination.is_symlink():
        raise ValueError('source must be a real directory and destination must be new')
    manifest = _read(source, 'manifest.json')
    if manifest.get('status') != 'complete' or manifest.get('managed_serving'):
        raise ValueError('source must be a complete unmanaged serving manifest')
    files = _files(manifest)
    for rel in files:
        _safe_file(source, rel, single_link=False)
    # Verify file hashes already supplied by the actual plugin manifest.
    for entry in manifest['components'].values():
        if entry.get('sha256') and _sha(source / entry['file']) != entry['sha256']:
            raise ValueError('source component hash mismatch')
    if manifest.get('base_file') and manifest.get('base_sha256'):
        if _sha(source / manifest['base_file']) != manifest['base_sha256']:
            raise ValueError('source base hash mismatch')
    destination.mkdir(mode=0o700)
    bundle_id = str(uuid.uuid4())
    for rel in files:
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / rel, target)
        os.chmod(target, 0o600)
        with target.open('rb') as f:
            os.fsync(f.fileno())
        if _sha(target) != _sha(source / rel):
            raise ValueError('serving copy verification failed')
    manifest['managed_serving'] = {'schema': SCHEMA, 'bundle_id': bundle_id,
                                   'generation': 0, 'withdrawal_events': []}
    _write(destination, 'ownership.json', {'schema': SCHEMA, 'bundle_id': bundle_id,
                                          'scope': 'managed serving state only'})
    _write(destination, 'manifest.json', manifest)
    _sync_dir(destination)
    return destination


class ManagedServingBundle:
    """Serialize real plugin calls with durable, restart-safe withdrawal.

    plugin_factory(folder) constructs RecoDKTPlugin, DirectDKTPlugin, or
    RecipeKTPlugin using its existing model/head/predictor hooks. Only this
    wrapper may serve the owned directory while it is open.
    """
    def __init__(self, folder, plugin_factory):
        self.folder = Path(folder).absolute()
        if self.folder.is_symlink() or not self.folder.is_dir():
            raise ValueError('managed bundle must be a real directory')
        self._mutex = threading.RLock()
        self._closed = False
        self._failed = False
        self._plugin = None
        # Reject research/unowned directories before creating any file in them.
        self._validate_owner()
        lock_path = self.folder / '.serving.lock'
        if lock_path.exists() or lock_path.is_symlink():
            _safe_file(self.folder, '.serving.lock')
        self._lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._validate_owner()
            self._recover()
            self._manifest = _read(self.folder, 'manifest.json')
            self._validate_inventory(self._manifest)
            self._plugin = plugin_factory(self.folder)
            self._check_plugin()
        except BaseException:
            self._clear_runtime()
            os.close(self._lock_fd)
            self._closed = True
            raise

    def _validate_owner(self):
        owner = _read(self.folder, 'ownership.json')
        manifest = _read(self.folder, 'manifest.json')
        policy = manifest.get('managed_serving', {})
        if owner.get('schema') != SCHEMA or policy.get('schema') != SCHEMA or owner.get('bundle_id') != policy.get('bundle_id'):
            raise ValueError('not an owned serving bundle')

    def _validate_inventory(self, manifest, removed=()):
        files = _files(manifest)
        allowed = set(files) | CONTROL | set(removed)
        found = _inventory(self.folder, allowed)
        if not set(files).issubset(found):
            raise ValueError('missing active serving artifact')
        withdrawn = set(map(str, manifest.get('initially_withdrawn', [])))
        if any(withdrawn.intersection(map(str, v['members'])) for v in manifest['components'].values()):
            raise ValueError('active component depends on a withdrawn school')

    def _check_plugin(self):
        if self._plugin.active != self._manifest['components'] or set(self._plugin.withdrawn) != set(self._manifest['initially_withdrawn']):
            raise ValueError('plugin does not honor the durable manifest')

    def _recover(self):
        # A process may stop after writing/fsyncing metadata but before renaming it.
        for name in ('manifest.json', 'withdrawal.pending.json'):
            staged = self.folder / ('.' + name + '.tmp')
            if staged.exists() or staged.is_symlink():
                _safe_file(self.folder, staged.name)
                staged.unlink(); _sync_dir(self.folder)
        pending_path = self.folder / 'withdrawal.pending.json'
        if not pending_path.exists():
            return
        transaction = _read(self.folder, 'withdrawal.pending.json')
        owner = _read(self.folder, 'ownership.json')
        if transaction.get('bundle_id') != owner['bundle_id']:
            raise ValueError('withdrawal transaction belongs to another bundle')
        target = transaction['manifest']
        removed = [_relative(x) for x in transaction['removed_files']]
        if set(removed).intersection(_files(target)):
            raise ValueError('transaction removes an active artifact')
        self._validate_inventory(target, removed)
        _write(self.folder, 'manifest.json', target)
        for rel in removed:
            path = _safe_file(self.folder, rel, missing=True)
            if path.exists():
                path.unlink(); _sync_dir(path.parent)
        self._validate_inventory(target)
        pending_path.unlink(); _sync_dir(self.folder)

    @contextmanager
    def _serving(self):
        with self._mutex:
            if self._closed or self._failed:
                raise RuntimeError('serving is closed or withdrawal recovery is required')
            yield

    def route(self, school):
        with self._serving():
            return list(self._plugin.route(school))

    def predict_history(self, school, skills, responses):
        with self._serving():
            return self._plugin.predict_history(school, skills, responses)

    def open_session(self, session_id, school, stream_factory):
        with self._serving():
            return self._plugin.open_session(session_id, school, stream_factory)

    def observe_session(self, session_id, skill, response):
        with self._serving():
            return self._plugin.observe_session(session_id, skill, response)

    def predict_session(self, session_id, skill):
        with self._serving():
            return self._plugin.predict_session(session_id, skill)

    def close_session(self, session_id):
        with self._serving():
            return self._plugin.close_session(session_id)

    def audit(self):
        with self._serving():
            self._check_plugin()
            self._validate_inventory(self._manifest)
            files = _files(self._manifest)
            withdrawn = set(self._manifest['initially_withdrawn'])
            bad_states = sum(any(n not in self._plugin.active for n, _, _ in row['states'])
                             or row['school'] in withdrawn for row in self._plugin.sessions.values())
            return {'scope': 'managed serving state', 'active_components': len(self._plugin.active),
                    'artifact_count': len(files), 'artifact_bytes': sum((self.folder / f).stat().st_size for f in files),
                    'withdrawn': sorted(withdrawn), 'affected_artifacts_remaining': 0,
                    'affected_bytes_remaining': 0, 'affected_sessions_remaining': bad_states,
                    'affected_resident_components': sum(n not in self._plugin.active for n in self._plugin.resident)}

    def withdraw(self, school):
        started = time.perf_counter()
        school = str(school)
        with self._serving():
            self._validate_inventory(self._manifest)
            if school in self._plugin.withdrawn:
                return {'school': school, 'idempotent': True, 'removed_components': [],
                        'before_affected_artifacts': 0, 'before_affected_bytes': 0,
                        'after_affected_artifacts': 0, 'after_affected_bytes': 0,
                        'withdrawal_processing_seconds': time.perf_counter() - started}
            removed = sorted(n for n, v in self._manifest['components'].items() if school in map(str, v['members']))
            if not removed:
                raise ValueError('school has no active components')
            owned_files = _files(self._manifest)
            removed_files = sorted(f for f, c in owned_files.items() if c in removed)
            before_bytes = sum(_safe_file(self.folder, f).stat().st_size for f in removed_files)
            target = copy.deepcopy(self._manifest)
            for name in removed:
                del target['components'][name]
            target['initially_withdrawn'] = sorted(set(map(str, target['initially_withdrawn'])) | {school})
            metadata = target['managed_serving']
            metadata['generation'] += 1
            metadata['withdrawal_events'].append({'school': school, 'components': removed,
                                                   'artifacts': removed_files, 'bytes': before_bytes})
            transaction = {'bundle_id': metadata['bundle_id'], 'manifest': target, 'removed_files': removed_files}
            try:
                _write(self.folder, 'withdrawal.pending.json', transaction)
                runtime = self._plugin.withdraw(school)
                self._recover()
                self._manifest = target
                self._check_plugin()
                audit = self.audit()
                if audit['affected_sessions_remaining'] or audit['affected_resident_components']:
                    raise RuntimeError('runtime still retains revoked state')
            except BaseException:
                self._failed = True
                self._clear_runtime()
                raise
            return {'school': school, 'idempotent': False, 'removed_components': removed,
                    'before_affected_artifacts': len(removed_files), 'before_affected_bytes': before_bytes,
                    'after_affected_artifacts': audit['affected_artifacts_remaining'],
                    'after_affected_bytes': audit['affected_bytes_remaining'],
                    'sessions_replay_required': runtime['sessions_replay_required'],
                    'owner_sessions_closed': runtime['owner_sessions_closed'],
                    'remaining_components': len(self._plugin.active),
                    'withdrawal_processing_seconds': time.perf_counter() - started,
                    'persistent_serving_state_removed': True, 'exclusive_serving_owner': True}

    def _clear_runtime(self):
        if self._plugin is None:
            return
        for session_id in list(self._plugin.sessions):
            try:
                self._plugin.close_session(session_id)
            except BaseException:
                pass
        self._plugin.sessions.clear()
        self._plugin.resident.clear()
        self._plugin.active.clear()

    def close(self):
        with self._mutex:
            if self._closed:
                return
            self._clear_runtime()
            self._closed = True
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
