from omegaconf import OmegaConf
from street_sign_project import train

"""Goal of this script is to check, whether the training orchestrator in our train.py correctly
passes arguments from hydra into the YOLOv26.train function. It's bad if stuff fails here silently.
We are not using the "Real" YOLOv26 class, as this might pull large models etc. etc.
We are:
1.) creating a fake yolo class
2.) tracking which arguments the .train() method of this class receives
3.) temporarily create our own fake hydra config to check wirering (so we don't always use the default values)
4.) Assert if wireing worked"""


class FakeYOLOv26:
    """Fake YOLOv26 wrapper used to test the training orchestrator."""

    init_kwargs: dict[str, object] = {}  # keeps track of the init arguments
    train_kwargs: dict[str, object] = {}  # keeps track of the train arguments passed in

    def __init__(self, **kwargs: object) -> None:
        """Store initialization arguments passed by train_model."""
        type(self).init_kwargs = kwargs

    def train(self, **kwargs: object) -> str:
        """Store training arguments passed by train_model."""
        type(self).train_kwargs = kwargs
        return "fake-results"


def test_train_model_passes_config_to_yolov26(monkeypatch) -> None:
    """Test that train_model maps Hydra config values to YOLOv26 training arguments.
    We need monkeypatch here to "force" the train orchestrator of train.py to use
    the FakeYOLOv26 class instead of the "real" class
    """
    monkeypatch.setattr(train, "YOLOv26", FakeYOLOv26)  # monkeypatch the YOLOv26 class to our FakeYOLOv26 class
    # Create fake hydra config
    config = OmegaConf.create(
        {
            "paths": {
                "data_yaml": "data/test-dataset.yaml",
                "model_path": "models/test-model.pt",
            },
            "model": {
                "base_model_name": "base.pt",
                "yolo_model_size": "s",
            },
            "training": {
                "epochs": 3,
                "batch_size": 8,
                "seed": 123,
                "optimizer": "AdamW",
                "lr0": 0.001,
                "freeze": 5,
                "device": "cpu",
                "workers": 2,
                "patience": 3,
            },
            "wandb": {
                "entity": "test-entity",
                "project": "test-project",
                "mode": "disabled",
                "dir": "/tmp",
            },
        }
    )

    # ACT: Call the train_model function with the fake config
    train.train_model.__wrapped__(config)

    # ASSERT, if the the train_model function was called with the correct config
    # Asserts the constructor:
    assert FakeYOLOv26.init_kwargs == {
        "local_model_name": "base.pt",
        "model_size": "s",
    }, "train_model did not initialize YOLOv26 from the model config"

    # Asserts the train arguments
    assert FakeYOLOv26.train_kwargs == {
        "data": "data/test-dataset.yaml",
        "model_path": "models/test-model.pt",
        "epochs": 3,
        "batch_size": 8,
        "seed": 123,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "freeze": 5,
        "device": "cpu",
        "workers": 2,
        "patience": 3,
        "wb_entity": "test-entity",
        "wb_project": "test-project",
        "wb_mode": "disabled",
        "wb_dir": "/tmp",
    }, "train_model did not pass the expected training config to YOLOv26.train"
