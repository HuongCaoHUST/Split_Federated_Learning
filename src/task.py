def get_task_name(config):
    """Return the normalized task name while preserving detection as default."""
    task = config.get("task", "detection")
    if isinstance(task, dict):
        task = task.get("type", task.get("name", "detection"))
    task = str(task).strip().lower()
    aliases = {
        "classification": "classification",
        "classify": "classification",
        "detection": "detection",
        "detect": "detection",
    }
    if task not in aliases:
        raise ValueError(
            f"Unsupported task '{task}'; expected classification or detection."
        )
    return aliases[task]
