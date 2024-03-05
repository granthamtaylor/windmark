from pathlib import Path

import flytekit as fl

from source.core import SequenceModule

@fl.task
def export_module_to_onnx(module: SequenceModule):

    filepath = Path(fl.current_context().working_directory) / "model.onnx"

    module.to_torchscript(file_path=filepath, method="script")

    # return fl.types.file.FlyteFile(filepath)
