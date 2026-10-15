"""PDH研究代码：按牌种序列登记已知暴露，阻断跨用途或训练流中的意外重复。"""
from __future__ import annotations

from contextlib import AbstractContextManager
import hashlib
from pathlib import Path
import sqlite3


def wall_fingerprint(wall) -> str:
    """区分完整有序牌墙；忽略同牌种四个实体编号的互换，不把仅编号变化当新局。"""
    if (len(wall) != 136 or any(type(tile) is not int for tile in wall)
            or set(wall) != set(range(136))):
        raise ValueError('data_protocol.invalid_136_tile_wall')
    return hashlib.sha256(bytes(tile // 4 for tile in wall)).hexdigest()


def exposed_walls(transcript: dict, audit: dict | None):
    """包括当前局、结算时预发的下一局与单独马牌；后一类保守计作已经暴露。"""
    yield transcript['hand_id'], transcript['wall']
    for event in (audit or {}).get('session_events', ()):
        if event.get('event') == 'HAND_DEALT':
            yield event['hand_id'], event['wall']
        elif event.get('event') == 'SEPARATE_HORSE_DECK':
            for index, wall in enumerate(event['cycles']):
                yield f"horse:{transcript['hand_id']}:{index}", wall


class WallRegistry(AbstractContextManager):
    """单任务主进程登记；SQLite事务也防止不同任务静默占用同一已知牌墙。"""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path, timeout=5)
        self.connection.execute('PRAGMA foreign_keys=ON')
        self.connection.executescript('''
            CREATE TABLE IF NOT EXISTS walls (
                fingerprint TEXT PRIMARY KEY, purpose TEXT NOT NULL,
                owner TEXT NOT NULL, first_origin TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS origins (
                owner TEXT NOT NULL, origin TEXT NOT NULL, fingerprint TEXT NOT NULL,
                PRIMARY KEY(owner, origin));
            CREATE TABLE IF NOT EXISTS history_files (
                digest TEXT PRIMARY KEY, path TEXT NOT NULL, records INTEGER NOT NULL);
        ''')
        self.connection.commit()

    def __exit__(self, *args):
        self.connection.close()

    def register(self, wall, purpose: str, owner: str, origin: str) -> str:
        """同评估牌墙的四座位/多个候选允许配对复用；换用途或另一个训练局则拒绝。"""
        if purpose not in {'train', 'development', 'heldout'} or not owner or not origin:
            raise ValueError('data_protocol.invalid_identity')
        fingerprint = wall_fingerprint(wall)
        with self.connection:
            self.connection.execute('BEGIN IMMEDIATE')
            known = self.connection.execute(
                'SELECT fingerprint FROM origins WHERE owner=? AND origin=?',
                (owner, origin)).fetchone()
            if known is not None and known[0] != fingerprint:
                raise ValueError('data_protocol.origin_changed_wall')
            row = self.connection.execute(
                'SELECT purpose,owner,first_origin FROM walls WHERE fingerprint=?',
                (fingerprint,)).fetchone()
            if row is not None and row != (purpose, owner, origin):
                raise ValueError('data_protocol.wall_reused_across_origin_or_split')
            self.connection.execute('INSERT OR IGNORE INTO walls VALUES (?,?,?,?)',
                                    (fingerprint, purpose, owner, origin))
            self.connection.execute('INSERT OR IGNORE INTO origins VALUES (?,?,?)',
                                    (owner, origin, fingerprint))
        return fingerprint

    def import_history_wall(self, wall, source: str) -> None:
        """历史重复仅合并为已见集合；不能反推历史训练没有重复或没有泄漏。"""
        fingerprint = wall_fingerprint(wall)
        self.connection.execute('INSERT OR IGNORE INTO walls VALUES (?,?,?,?)',
                                (fingerprint, 'history', source, source))

    def history_loaded(self, digest: str) -> bool:
        """只以完整文件内容哈希复用导入记录，不以文件名或修改日期替代。"""
        return self.connection.execute(
            'SELECT 1 FROM history_files WHERE digest=?', (digest,)).fetchone() is not None

    def finish_history(self, digest: str, path: Path, records: int) -> None:
        """完整读取成功后提交；失败或半截gzip不能被标为已导入。"""
        self.connection.execute('INSERT INTO history_files VALUES (?,?,?)',
                                (digest, str(path), records))
        self.connection.commit()

    def counts(self, owner_prefix: str) -> dict:
        """统计本轮已登记的唯一暴露牌墙，含为抓马预发的下一手，不等同已打局数。"""
        rows = self.connection.execute(
            'SELECT purpose,COUNT(*) FROM walls WHERE substr(owner,1,?)=? GROUP BY purpose',
            (len(owner_prefix), owner_prefix)).fetchall()
        return dict(rows)
