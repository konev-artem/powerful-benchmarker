import logging
import argparse
from sam import SAM
from torch.optim import RMSprop
from pytorch_metric_learning import trainers
from pytorch_metric_learning.utils import  common_functions as c_f
from powerful_benchmarker import api_parsers
logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--pytorch_home", type=str, default=None)
parser.add_argument("--dataset_root", type=str, default="datasets")
parser.add_argument("--root_experiment_folder", type=str, default="experiments")
parser.add_argument("--global_db_path", type=str, default=None)
parser.add_argument("--merge_argparse_when_resuming", default=False, action='store_true')
parser.add_argument("--root_config_folder", type=str, default=None)
parser.add_argument("--bayes_opt_iters", type=int, default=0)
parser.add_argument("--reproductions", type=str, default="0")
args, _ = parser.parse_known_args()

if args.bayes_opt_iters > 0:
	from powerful_benchmarker.runners.bayes_opt_runner import BayesOptRunner
	args.reproductions = [int(x) for x in args.reproductions.split(",")]
	runner = BayesOptRunner
else:
	from powerful_benchmarker.runners.single_experiment_runner import SingleExperimentRunner
	runner = SingleExperimentRunner
	del args.bayes_opt_iters
	del args.reproductions

class SAMRMSprop(SAM):
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False):
        super().__init__(params, RMSprop, rho=0.05, adaptive=False,
                         lr=lr, alpha=alpha,
                         eps=eps, weight_decay=weight_decay,
                         momentum=momentum, centered=centered)


class SAMTrainer(trainers.MetricLossOnly):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward_and_backward(self):
        batch = self.get_batch()
        self.zero_losses()
        self.zero_grad()
        self.calculate_loss(batch)
        self.loss_tracker.update(self.loss_weights)
        self.backward()
        self.clip_gradients()
        self.first_step_optimizers()
        
        # self.zero_losses()
        self.zero_grad()
        self.calculate_loss(batch)
        self.loss_tracker.update(self.loss_weights)
        self.backward()
        self.clip_gradients()
        self.second_step_optimizers()
        
    def first_step_optimizers(self):
        for k, v in self.optimizers.items():
            if c_f.regex_replace("_optimizer$", "", k) not in self.freeze_these:
                v.first_step(zero_grad=True)
                
    def second_step_optimizers(self):
        for k, v in self.optimizers.items():
            if c_f.regex_replace("_optimizer$", "", k) not in self.freeze_these:
                v.second_step(zero_grad=True)


r = runner(**(args.__dict__))
r.register("optimizer", SAMRMSprop)
r.register("trainer", SAMTrainer)
r.run()
