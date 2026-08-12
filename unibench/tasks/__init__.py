from .base import Task, TaskItem
from .fairness_task import FairnessTask
from .summarization_task import SummarizationTask

TASK_REGISTRY = {
    "fairness": FairnessTask,
    "summarization": SummarizationTask,
}


def build_task(name: str) -> Task:
    if name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{name}'. Available: {list(TASK_REGISTRY)}")
    return TASK_REGISTRY[name]()
