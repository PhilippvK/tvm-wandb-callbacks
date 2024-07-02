# TODO: license
import tempfile
from typing import List

import tvm
from tvm import meta_schedule as ms
from tvm.script import tir as T

from wandb_callbacks.meta_schedule_callback import WandbCallback


@tvm.script.ir_module
class Matmul:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        C = T.match_buffer(c, (1024, 1024), "float32")
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def test_meta_schedule_wandb_callback():
    @ms.derived_object
    class AllZeroRunnerFuture(ms.runner.PyRunnerFuture):
        def done(self) -> bool:
            return True

        def result(self) -> ms.runner.RunnerResult:
            return ms.runner.RunnerResult([0.0, 0.0], None)

    @ms.derived_object
    class AllZeroRunner(ms.runner.PyRunner):
        def run(self, runner_inputs: List[ms.runner.RunnerInput]) -> List[ms.runner.RunnerResult]:
            return [AllZeroRunnerFuture() for _ in runner_inputs]

    wandb_callback = WandbCallback()
    run_config = {
        "foo": "bar",
    }
    wandb_callback.init_session(project="TVM", config=run_config)
    with tempfile.TemporaryDirectory() as work_dir:
        ms.tune_tir(
            mod=Matmul,
            target="llvm -num-cores=1",
            work_dir=work_dir,
            max_trials_global=10,
            runner=AllZeroRunner(),
            measure_callbacks=[wandb_callback]
        )
    wandb_callback.deinit_session(artifacts=[])


if __name__ == "__main__":
    test_meta_schedule_wandb_callback()
