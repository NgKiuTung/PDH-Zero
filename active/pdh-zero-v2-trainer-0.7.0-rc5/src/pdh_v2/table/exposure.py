"""Import known historical exposure and enforce explicit common-wall study pairing."""
from __future__ import annotations

import json
from pathlib import Path

from pdh_v2.game.session import GameSession
from pdh_v2.table.data import contract, records
from pdh_v2.training.candidate_store import file_digest
from pdh_v2.training.refinement_control import atomic_json
from pdh_v2.training.wall_registry import WallRegistry, exposed_walls, wall_fingerprint


class ExposureStore(WallRegistry):
    """Add provenance without rewriting the original registry or claiming complete history."""

    def __init__(self, path: Path):
        super().__init__(path)
        self.connection.executescript('''
            CREATE TABLE IF NOT EXISTS rc5_roles (fingerprint TEXT, role TEXT,
                PRIMARY KEY(fingerprint,role));
            CREATE TABLE IF NOT EXISTS rc5_sources (digest TEXT PRIMARY KEY, role TEXT,
                records INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS rc5_completed (arm TEXT, origin TEXT,
                PRIMARY KEY(arm,origin));
        ''')
        self.connection.commit()

    def import_file(self, path: Path, role: str) -> dict:
        """Read complete gzip before commit; a renamed file cannot change its data role."""
        if role not in {"train", "development", "heldout_used"}:
            raise ValueError("rc5.invalid_historical_role")
        digest = file_digest(path)
        known = self.connection.execute(
            'SELECT role,records FROM rc5_sources WHERE digest=?', (digest,)).fetchone()
        if known is not None:
            if known[0] != role:
                raise ValueError("rc5.source_role_conflict")
            return dict(path=str(path), sha256=digest, role=role, records=known[1], cached=True)
        count = 0
        with self.connection:
            for row in records(path):
                for _, wall in exposed_walls(row['transcript'], row['audit']):
                    fingerprint = wall_fingerprint(wall)
                    self.import_history_wall(wall, digest)
                    self.connection.execute('INSERT OR IGNORE INTO rc5_roles VALUES (?,?)',
                                            (fingerprint, role))
                count += 1
            if not count:
                raise ValueError("rc5.empty_history")
            self.connection.execute('INSERT INTO rc5_sources VALUES (?,?,?)',
                                    (digest, role, count))
        return dict(path=str(path), sha256=digest, role=role, records=count, cached=False)

    def require_training_source(self, path: Path) -> str:
        """Only the frozen, archived training replay is authorized for auxiliary fitting."""
        digest = file_digest(path)
        if digest != contract()['training_replay_sha256']:
            raise ValueError("rc5.training_source_not_authorized")
        row = self.connection.execute(
            'SELECT role FROM rc5_sources WHERE digest=?', (digest,)).fetchone()
        if row != ('train',):
            raise ValueError("rc5.training_source_not_registered")
        return digest

    def require_known_history(self) -> None:
        """An empty/new registry cannot silently bypass the eight known archived sources."""
        known = set(self.connection.execute('SELECT digest,role FROM rc5_sources'))
        sources = contract()['historical_replay_sha256']
        required = {(digest, role) for role, digests in sources.items() for digest in digests}
        if not required.issubset(known):
            raise ValueError('rc5.known_historical_exposure_not_imported')

    def reserve_initial(self, seeds, owner: str, purpose: str) -> None:
        """Reserve before the first Actor call; an interrupted hand is already exposed."""
        for seed in seeds:
            hand = GameSession(seed).current_hand
            self.register(hand.wall.order, purpose, owner, f'{seed}:{hand.hand_id}')

    def check_label_wall(self, wall) -> None:
        """Guard labels against known dev/old-heldout collision even in an allowed file."""
        roles = {row[0] for row in self.connection.execute(
            'SELECT role FROM rc5_roles WHERE fingerprint=?', (wall_fingerprint(wall),))}
        if roles != {'train'}:
            raise ValueError("rc5.label_wall_cross_split")

    def terminal_callback(self, owner: str, purpose: str, arm: str):
        """Permit deliberately paired arms; reject a second completion inside one arm."""
        def register_terminal(terminal):
            """Keep each completed evaluation seat distinct while pairing wall exposure."""
            audit = terminal.audit
            origin = f"{audit['session_seed']}:{terminal.hand_id}"
            with self.connection:
                self.connection.execute('INSERT INTO rc5_completed VALUES (?,?)',
                                        (arm, f"{origin}:env={terminal.env_index}"))
            transcript = json.loads(terminal.transcript.to_json())
            for hand_id, wall in exposed_walls(transcript, audit):
                self.register(wall, purpose, owner, f"{audit['session_seed']}:{hand_id}")
        return register_terminal


def prepare_registry(train_run: Path, heldout_run: Path, registry: Path, output: Path):
    """Import both explicit archives, including every saved historical dev candidate."""
    if output.exists():
        raise FileExistsError(output)
    training = train_run / 'replays/hands.jsonl.gz'
    if file_digest(training) != contract()['training_replay_sha256']:
        raise ValueError("rc5.training_source_not_authorized")
    paths = [(training, 'train')]
    paths.extend((p, 'development') for p in sorted(
        (train_run / 'evaluations').glob('*/hands.jsonl.gz')))
    paths.extend((heldout_run / name / 'hands.jsonl.gz', 'heldout_used')
                 for name in ('baseline', 'candidate'))
    if len(paths) != 8:
        raise ValueError("rc5.historical_file_set_incomplete")
    for path, role in paths:
        if file_digest(path) not in contract()['historical_replay_sha256'][role]:
            raise ValueError('rc5.archived_replay_digest_mismatch')
    with ExposureStore(registry) as store:
        imported = [store.import_file(path, role) for path, role in paths]
        store.require_known_history()
        overlaps = store.connection.execute('''SELECT COUNT(*) FROM
            (SELECT fingerprint FROM rc5_roles
             GROUP BY fingerprint HAVING COUNT(*)>1)''').fetchone()[0]
        counts = dict(store.connection.execute(
            'SELECT role,COUNT(*) FROM rc5_roles GROUP BY role'))
    report = dict(status='PASS', files=imported, historical_coverage='partial',
                  original_registry_in_source_archives=False, known_role_wall_counts=counts,
                  cross_role_fingerprints=overlaps,
                  claim='known_archived_exposure_only_not_complete_ancestry',
                  used_heldout='historical_evidence_and_diagnostics_never_auxiliary_training')
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    return report