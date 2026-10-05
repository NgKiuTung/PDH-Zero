"""Tests for the balanced, resumable Gate B campaign helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pdh_phase1.campaign import (
    CampaignSettings,
    build_participants,
    initial_state,
    parse_csv_ints,
    parse_csv_methods,
    write_resolved_config,
)


class CampaignTests(unittest.TestCase):
    """Verify participant pairing, state, and immutable config generation."""

    def test_participants_are_seed_paired_and_unique(self) -> None:
        participants = build_participants(
            parse_csv_methods("plain,privileged"),
            parse_csv_ints("20260929,3407,9102"),
        )

        self.assertEqual(
            [item.participant_id for item in participants],
            [
                "plain_seed20260929",
                "privileged_seed20260929",
                "plain_seed3407",
                "privileged_seed3407",
                "plain_seed9102",
                "privileged_seed9102",
            ],
        )
        self.assertEqual(len({item.participant_id for item in participants}), 6)

    def test_initial_state_tracks_resume_cursor(self) -> None:
        participants = build_participants(["plain", "privileged"], [1])
        settings = CampaignSettings(10.0, 1, 2, 100, "heuristic", "cpu", 60.0, 30, 1)

        state = initial_state("test-job", settings, participants)

        self.assertEqual(state["status"], "created")
        self.assertEqual(state["next_participant_index"], 0)
        self.assertEqual(set(state["participants"]), {"plain_seed1", "privileged_seed1"})

    def test_resolved_config_disables_inline_evaluation(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        participant = build_participants(["privileged"], [123])[0]
        with tempfile.TemporaryDirectory() as temporary:
            target = write_resolved_config(
                project_root,
                Path(temporary),
                participant,
                "cpu",
            )
            config = json.loads(target.read_text(encoding="utf-8"))

        self.assertEqual(config["seed"], 123)
        self.assertEqual(config["device"], "cpu")
        self.assertTrue(config["privileged_critic"])
        self.assertEqual(config["evaluation_every_updates"], 0)

    def test_duplicate_seed_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_csv_ints("7,7")


if __name__ == "__main__":
    unittest.main()
