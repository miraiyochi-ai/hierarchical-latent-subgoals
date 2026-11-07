import json
from pathlib import Path
import tempfile
import unittest
from compare_results import compare, paired, read_cell


class PairedTests(unittest.TestCase):
    def test_win_loss_and_tie(self):
        result = paired({1: True, 2: False, 3: True, 4: False},
                        {1: False, 2: True, 3: True, 4: False})
        self.assertEqual(result['descriptive_winner'], 'tie')
        self.assertEqual(result['hierarchical_only'], 1)
        self.assertEqual(result['flat_only'], 1)
        self.assertEqual(result['exact_paired_p_two_sided_unadjusted'], 1.)

    def test_flat_can_win(self):
        result = paired({i: False for i in range(6)}, {i: True for i in range(6)})
        self.assertEqual(result['descriptive_winner'], 'flat')
        self.assertEqual(result['hierarchical_minus_flat_points'], -100)
        self.assertEqual(result['exact_paired_p_two_sided_unadjusted'], .03125)

    def test_mismatched_tasks_rejected(self):
        with self.assertRaises(ValueError):
            paired({1: True}, {2: True})

    def test_full_comparison_and_corrupt_membership(self):
        with tempfile.TemporaryDirectory(prefix='h3_pair_test_') as directory:
            roots = {name: Path(directory)/name for name in
                     ('old_hierarchical', 'old_flat', 'new_hierarchical', 'new_flat')}
            for label, root in roots.items():
                condition = 'predicted' if label.endswith('hierarchical') else 'flat-direct-goal'
                train_condition = 'hierarchical' if condition == 'predicted' else condition
                root.mkdir(parents=True)
                (root/'protocol.json').write_text(json.dumps(dict(horizons=[25, 50, 75], updates=30000)))
                (root/'train150_manifest.json').write_text(json.dumps(dict(condition=train_condition,
                    seed=42, updates=30000, training_horizons=[25, 50, 75, 100, 150],
                    initialization='from_scratch_not_finetuning')))
                for filename in ('parameter_manifest.json', 'source_manifest.json',
                                 f'effective_configs/{train_condition.replace("-", "_")}_seed_42.json'):
                    p = root/filename
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(json.dumps(dict(same_settings=True)))
                for d in (25, 50, 75, 100, 150):
                    folder = root/('evaluation_shards' if d <= 75 else 'h3x')
                    folder.mkdir(parents=True, exist_ok=True)
                    rows = [dict(condition=condition, d=d, model_seed=42, evaluation_seed=42,
                                 episode_idx=i, start_step=1, task_sha256=str(i),
                                 success=i < (35 if label=='new_flat' else 20),
                                 checkpoint_sha256=label, input_sha256=str(d)) for i in range(50)]
                    path = folder/f'raw_rollouts_{condition}_d{d}_m42_e42.jsonl'
                    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            result = compare(roots)
            self.assertEqual(result['state'], 'comparison_complete')
            self.assertEqual(result['cells']['100']['new']['descriptive_winner'], 'flat')
            self.assertEqual(result['cells']['150']['change_in_hierarchical_minus_flat_points'], -30)
            path = roots['new_flat']/'h3x/raw_rollouts_flat-direct-goal_d150_m42_e42.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[-1] = rows[0]
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaises(ValueError):
                read_cell(roots['new_flat'], 'flat-direct-goal', 150)


if __name__ == '__main__':
    unittest.main()
