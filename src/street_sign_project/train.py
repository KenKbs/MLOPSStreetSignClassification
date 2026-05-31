from street_sign_project.model import YOLOv26
import typer

def train_model() -> None:
    model = YOLOv26()
    results = model.train()
    print(f"train results: \n {results}")

if __name__ == "__main__":
    typer.run(train_model)
