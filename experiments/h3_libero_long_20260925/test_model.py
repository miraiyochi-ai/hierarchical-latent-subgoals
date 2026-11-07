"""CPU synthetic architecture checks, separate from real-data GPU preflight."""
import importlib.util
import unittest

if importlib.util.find_spec('torch') is None:
    raise unittest.SkipTest('Torch is required; temporal CPU tests remain runnable without it')

import torch
from model import Config, Policy


def synthetic():
    torch.manual_seed(91)
    return dict(lower_visual=torch.randn(1,8,16),lower_states=torch.randn(1,8,9),
                upper_visual=torch.randn(1,16,16),upper_actions=torch.randn(1,16,7),
                upper_mask=torch.ones(1,1),remaining=torch.ones(1,1),language=torch.randn(1,8),
                upper_teacher=torch.randn(1,16),future_actions=torch.randn(1,8,7))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.hier=Policy(Config(visual_dim=16,text_dim=8)).eval()
        cls.flat=Policy(Config(hierarchical=False,flat_width=512,visual_dim=16,text_dim=8)).eval()

    def test_hierarchical_output_shape_and_finite(self):
        batch=synthetic()
        with torch.no_grad():
            goal=self.hier.predict_subgoal(batch)
            actions=self.hier.actions(batch,goal)
        self.assertEqual(goal.shape,(1,16))
        self.assertEqual(actions.shape,(1,8,7))
        self.assertTrue(torch.isfinite(actions).all())

    def test_flat_outputs_actions_without_upper_module(self):
        batch=synthetic()
        self.assertIsNone(self.flat.layer2)
        self.assertEqual(self.flat.actions(batch,batch['language']).shape,(1,8,7))
        with self.assertRaises(ValueError):
            self.flat.predict_subgoal(batch)

    def test_future_teacher_cannot_change_predicted_subgoal(self):
        batch=synthetic()
        with torch.no_grad():
            expected=self.hier.predict_subgoal(batch)
            batch['upper_teacher'].fill_(1000)
            batch['future_actions'].fill_(-1000)
            observed=self.hier.predict_subgoal(batch)
        self.assertTrue(torch.equal(expected,observed))

    def test_lower_loss_reaches_upper_parameters(self):
        batch=synthetic(); self.hier.zero_grad(set_to_none=True)
        # Original H3 uses zero-initialized modulation. The first update opens
        # that path; test its learned connectivity without altering the model.
        optimizer=torch.optim.SGD(self.hier.parameters(),lr=1e-3)
        initial=self.hier.lower_loss(batch,self.hier.predict_subgoal(batch),torch.Generator().manual_seed(16))['total']
        initial.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        subgoal=self.hier.predict_subgoal(batch)
        loss=self.hier.lower_loss(batch,subgoal,torch.Generator().manual_seed(17))['total']
        loss.backward()
        gradients=[p.grad for p in self.hier.layer2.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(float(g.norm()) for g in gradients),0.)

    def test_initial_mask_removes_unobserved_actions(self):
        batch=synthetic(); batch['upper_mask'].zero_()
        with torch.no_grad():
            one=self.hier.predict_subgoal(batch)
            batch['upper_actions'].fill_(100)
            two=self.hier.predict_subgoal(batch)
        self.assertTrue(torch.equal(one,two))


if __name__=='__main__':
    unittest.main()
