from pathlib import Path

import flytekit as fk

from windmark.core.architecture import SequenceModule

@fk.task
def export_module_to_onnx(module: SequenceModule):

    filepath = Path(fk.current_context().working_directory) / "model.onnx"

    module.to_onnx(file_path=filepath, export_params=True)

    # return fk.types.file.FlyteFile(filepath)
