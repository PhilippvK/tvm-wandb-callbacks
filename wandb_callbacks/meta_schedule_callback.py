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


@ms.derived_object
class WandbCallback(ms.measure_callback.PyMeasureCallback):

    def __init__(self):
        super().__init__()
        self.state = WandbCallbackState.UNINITIALIZED
        self.best_flops = 0
        self.cur_flops = 0
        self.ct = 0
        self.last_tic = 0
        self.num_invalid = 0
        # TODO: batched/per_trial?
        # TODO: pre/post

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
        print("task_id", task_id)
        print("task_scheduler", task_scheduler, dir(task_scheduler))
        # print("task_scheduler.tasks_", task_scheduler.tasks_)
        # print("task_scheduler.tasks_[0]",
        task = task_scheduler.tasks_[task_id]
        print("task", task)
        print("task.ctx", task.ctx)
        print("task.task_weight", task.task_weight)
        print("task.flop", task.flop)
        print("task.is_terminated", task.is_terminated)
        print("task.build_error_count", task.build_error_count)
        print("task.run_error_count", task.run_error_count)
        print("task.measure_candidates", task.measure_candidates)
        print("task.builder_results", task.builder_results)
        # print("task_scheduler.tasks_[0].runner_results", task_scheduler.tasks_[0].runner_results)
        # task_id =
        # task_name_short = f"task{task_id}"
        # task_name_long =
        print("measure_candidates", measure_candidates)
        print("measure_candidates[0]", measure_candidates[0], dir(measure_candidates[0]))
        print("measure_candidates[0].args_info", measure_candidates[0].args_info)
        print("measure_candidates[0].handle", measure_candidates[0].handle)
        # print("measure_candidates[0].legacy_repr", measure_candidates[0].legacy_repr)
        print("measure_candidates[0].sch", measure_candidates[0].sch)
        # args_info handle legacy_repr sch
        print("builder_results", builder_results)
        print("builder_results[0]", builder_results[0], dir(builder_results[0]))
        print("builder_results[0].artifact_path", builder_results[0].artifact_path)
        print("builder_results[0].error_msg", builder_results[0].error_msg)
        print("builder_results[0].handle", builder_results[0].handle)
        # print("builder_results[0].legacy_repr", builder_results[0].legacy_repr)
        # artifact_path error_msg handle legacy_repr
        print("runner_results", runner_results)
        print("runner_results[0]", runner_results[0], dir(runner_results[0]))
        print("runner_results[0].error_msg", runner_results[0].error_msg)
        print("runner_results[0].handle", runner_results[0].handle)
        # print("runner_results[0].legacy_repr", runner_results[0].legacy_repr)
        print("runner_results[0].run_secs", runner_results[0].run_secs)
        # error_msg handle legacy_repr run_secs
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
