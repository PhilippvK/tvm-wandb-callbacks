# TODO: license

import time
import os
import json
import numpy as np
import pathlib
import tvm
from tvm.relay.backend import Runtime
import tvm.micro.testing
from pathlib import Path
from typing import Optional, List, Union

import wandb

# TODO: wrap in a class
def init_wandb_callback(project: str = "TVM", run_name: Optional[str] = None, config: dict = None):
    core = True
    if core:
        wandb.require("core")
    wandb.init(
        # set the wandb project where this run will be logged
        project="TVM",
        name=run_name,
        # track hyperparameters and run metadata
        config=config,

    )


def wandb_callback(idx, per_trial=False):

    class _Context(object):
        """Context to store local variables"""

        def __init__(self):
            self.best_flops = 0
            self.cur_flops = 0
            self.ct = 0
            self.last_tic = 0
            self.num_invalid = 0

    ctx = _Context()
    tic = time.time()
    ctx.last_tic = tic

    def _callback(tuner, inputs, results):

        flops = 0
        delta = time.time() - ctx.last_tic
        delta_per_trial = delta / len(results)

        i = 0
        for inp, res in zip(inputs, results):
            i += 1
            ctx.ct += 1
            if res.error_no == 0:
                flops = inp.task.flop / np.mean(res.costs)
            else:
                ctx.num_invalid += 1
            ctx.cur_flops = flops
            ctx.best_flops = max(flops, ctx.best_flops)
            if per_trial:
                total = ctx.last_tic + i * delta_per_trial
                wandb.log(
                    {
                        f"task{idx}.trials": ctx.ct,
                        f"task{idx}.invalid": ctx.num_invalid,
                        f"task{idx}.cur_flops": ctx.cur_flops,
                        f"task{idx}.best_flops": ctx.best_flops,
                        f"task{idx}.time.total": total,
                        f"task{idx}.time.trial": delta_per_trial,
                        f"task{idx}.time.trial.mean": total / ctx.ct,
                    }
                )
        assert ctx.best_flops == tuner.best_flops
        if not per_trial:
            total = ctx.last_tic + delta
            wandb.log(
                {
                    f"task{idx}.trials": ctx.ct,
                    f"task{idx}.invalid": ctx.num_invalid,
                    f"task{idx}.cur_flops": ctx.cur_flops,
                    f"task{idx}.best_flops": ctx.best_flops,
                    f"task{idx}.time.total": total,
                    f"task{idx}.time.trial": delta_per_trial,
                    f"task{idx}.time.trial.mean": total / ctx.ct,
                }
            )

        ctx.last_tic = time.time()

    return _callback


def log_wandb_pre_tune(i, task):
    sz = len(task.config_space)
    wandb.log({f"task{i}.config_space_size": sz})


def log_wandb_post_tune(i, task):
    # wandb.log({"global.tuned_tasks": i+1})
    pass

def deinit_wandb_callback(artifacts: Optional[List[Union[str, Path]]] = None):
    if artifacts is None:
        artifacts = []
    for artifact in artifacts:
        wandb.save(artifact)
    wandb.finish()
