from packages.core.types import GpuClass, JobState, TERMINAL_STATES, TaskType


def test_task_type_values():
    assert TaskType.SEGMENTATION_TEXT == "segmentation.text"
    assert TaskType.DEPTH_MULTIVIEW == "depth.multiview"
    assert {t.value for t in TaskType} == {
        "segmentation.text",
        "segmentation.point",
        "segmentation.box",
        "depth.monocular",
        "depth.multiview",
    }


def test_job_state_terminal_set():
    assert TERMINAL_STATES == {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELED}
    assert JobState.RUNNING not in TERMINAL_STATES


def test_gpu_class_includes_cpu_and_h100():
    assert GpuClass.CPU == "cpu"
    assert GpuClass.H100_80G == "h100_80g"
