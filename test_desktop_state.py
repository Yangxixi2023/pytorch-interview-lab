import json
from pathlib import Path
import tempfile
import unittest

from desktop_state import DesktopState


class StateTests(unittest.TestCase):
    def test_state_survives_new_instance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'progress.json'
            DesktopState(path).save({'drafts':{'ppo_loss':'print("草稿")'}})
            self.assertEqual(DesktopState(path).read()['drafts']['ppo_loss'],'print("草稿")')

    def test_close_snapshot_wins_over_pending_autosave(self):
        with tempfile.TemporaryDirectory() as directory:
            store=DesktopState(Path(directory)/'progress.json')
            store.save({'drafts':{'ppo_loss':'old'}})
            store.finish({'drafts':{'ppo_loss':'latest'}})
            self.assertFalse(store.save({'drafts':{'ppo_loss':'old'}}))
            self.assertEqual(store.read()['drafts']['ppo_loss'],'latest')

    def test_corrupt_existing_state_is_not_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'progress.json'
            path.write_text('{broken',encoding='utf-8')
            with self.assertRaises(json.JSONDecodeError):
                DesktopState(path).read()
            self.assertEqual(path.read_text(encoding='utf-8'),'{broken')


if __name__=='__main__':
    unittest.main()
