# TODO: license
import time
from pathlib import Path
from typing import List, Optional, Union
from enum import IntEnum, auto

from tvm import meta_schedule as ms

import wandb


class WandbCallbackState(IntEnum):
    UNINITIALIZED = auto()
    INITIALIZED = auto()
    DEINITIALIZED = auto()


class WandbCallback(ms.measure_callback.PyMeasureCallback):

    def __init__(self):
        self.state = WandbCallbackState.UNINITIALIZED
        self.best_flops = 0
        self.cur_flops = 0
        self.ct = 0
        self.last_tic = 0
        self.num_invalid = 0
        # TODO: batched/per_trial?

    def init_session(self, project: str = "TVM", run_name: Optional[str] = None, config: dict = None):
        assert self.state == WandbCallbackState.UNINITIALIZED, "Session already initialized"
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
        tic = time.time()
        self.last_tic = tic
        self.state = WandbCallbackState.INITIALIZED

    def deinit_session(self, artifacts: Optional[List[Union[str, Path]]] = None):
        assert self.state == WandbCallbackState.INITIALIZED, "Session not initialized"
        if artifacts is None:
            artifacts = []
        for artifact in artifacts:
            wandb.save(artifact)
        wandb.finish()
        self.state = WandbCallbackState.UNINITIALIZED

    def apply(
        self,
        task_scheduler: ms.task_scheduler.TaskScheduler,
        task_id: int,
        measure_candidates: List[ms.MeasureCandidate],
        builder_results: List[ms.builder.BuilderResult],
        runner_results: List[ms.runner.RunnerResult],
    ) -> None:
        assert self.state == WandbCallbackState.INITIALIZED, "Session not initialized"
        flops = 0
        delta = time.time() - self.last_tic
        delta_per_trial = delta / len(runner_results)
        print("task_scheduler", task_scheduler)
        print("measure_candidates", measure_candidates)
        print("builder_results", builder_results)
        print("runner_results", runner_results)
        print("delta_per_trial", delta_per_trial)
        print("flops", flops)
        # TODO: loop
        self.last_tic = time.time()
        # assert len(measure_candidates) == 1
        # assert (
        #     len(builder_results) == 1
        #     and builder_results[0].error_msg is None
        #     and builder_results[0].artifact_path == "test_build"
        # )
        # assert (
        #     len(runner_results) == 1
        #     and runner_results[0].error_msg is None
        #     and len(runner_results[0].run_secs) == 2
        # )
