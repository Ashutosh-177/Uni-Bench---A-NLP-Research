from .base import Task, TaskItem
from .bbq_task import BBQTask
from .fairness_task import FairnessTask
from .summarization_task import SummarizationTask
from .wikipedia_summarization_task import WikipediaSummarizationTask

TASK_REGISTRY = {
    "fairness": FairnessTask,
    "summarization": SummarizationTask,
    "bbq_bias": BBQTask,
    "wikipedia_summarization": WikipediaSummarizationTask,
}


def build_task(name: str) -> Task:
    if name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{name}'. Available: {list(TASK_REGISTRY)}")
    return TASK_REGISTRY[name]()
