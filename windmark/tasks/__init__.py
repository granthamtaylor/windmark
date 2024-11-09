from windmark.tasks.digest import create_digest_centroids_from_lifestream as digest
from windmark.tasks.pretrain import pretrain_sequence_encoder as pretrain
from windmark.tasks.finetune import finetune_sequence_encoder as finetune
from windmark.tasks.sample import create_sample_manager as sample
from windmark.tasks.predict import predict_sequence_encoder as predict
from windmark.tasks.system import create_system_manager as system
from windmark.tasks.task import create_task_manager as task
from windmark.tasks.split import create_split_manager as split
from windmark.tasks.levels import create_unique_levels_from_lifestream as levels
from windmark.tasks.compare import compare_sequence_lengths as compare
from windmark.tasks.label import create_experiment_label as label


__all__ = [
    pretrain,
    finetune,
    levels,
    digest,
    sample,
    predict,
    system,
    task,
    split,
    compare,
    label,
]
