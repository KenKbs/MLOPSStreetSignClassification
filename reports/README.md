# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

`--- question 1 fill here ---`

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

`![my_image](figures/<image>.<extension>)`

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

or

```bash
uv add typer markdown
```

## Overall project checklist

The checklist is _exhaustive_ which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

- [x] Create a git repository (M5)
- [x] Make sure that all team members have write access to the GitHub repository (M5)
- [x] Create a dedicated environment for you project to keep track of your packages (M2)
- [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
- [x] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
- [x] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
- [x] Remember to either fill out the `requirements.txt`/`requirements_dev.txt` files or keeping your
      `pyproject.toml`/`uv.lock` up-to-date with whatever dependencies that you are using (M2+M6)
- [x] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
- [x] Do a bit of code typing and remember to document essential parts of your code (M7)
- [x] Setup version control for your data or part of your data (M8)
- [x] Add command line interfaces and project commands to your code where it makes sense (M9)
- [x] Construct one or multiple docker files for your code (M10)
- [x] Build the docker files locally and make sure they work as intended (M10)
- [x] Write one or multiple configurations files for your experiments (M11)
- [x] Used Hydra to load the configurations and manage your hyperparameters (M11)
- [x] Use profiling to optimize your code (M12)
- [x] Use logging to log important events in your code (M14)
- [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
- [x] Consider running a hyperparameter optimization sweep (M14)
- [not applicable] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

- [x] Write unit tests related to the data part of your code (M16)
- [x] Write unit tests related to model construction and or model training (M16)
- [x] Calculate the code coverage (M16)
- [x] Get some continuous integration running on the GitHub repository (M17)
- [x] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
- [x] Add a linting step to your continuous integration (M17)
- [x] Add pre-commit hooks to your version control setup (M18)
- [x] Add a continues workflow that triggers when data changes (M19)
- [x] Add a continues workflow that triggers when changes to the model registry is made (M19)
- [x] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
- [x] Create a trigger workflow for automatically building your docker images (M21)
- [x] Get your model training in GCP using either the Engine or Vertex AI (M21)
- [x] Create a FastAPI application that can do inference using your model (M22)
- [x] Deploy your model in GCP using either Functions or Run as the backend (M23)
- [x] Write API tests for your application and setup continues integration for these (M24)
- [x] Load test your application (M24)
- [x] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
- [x] Create a frontend for your API - FastAPI was used (M26)

### Week 3

- [x] Check how robust your model is towards data drifting (M27)
- [x] Setup collection of input-output data from your deployed application (M27)
- [x] Deploy to the cloud a drift detection API (M27)
- [ ] Instrument your API with a couple of system metrics (M28)
- [ ] Setup cloud monitoring of your instrumented application (M28)
- [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
- [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
- [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
- [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

- [x] Write some documentation for your application (M32)
- [ ] Publish the documentation to GitHub Pages (M32)
- [x] Revisit your initial project description. Did the project turn out as you wanted?
- [ ] Create an architectural diagram over your MLOps pipeline
- [x] Make sure all group members have an understanding about all parts of the project
- [x] Uploaded all your code to GitHub

## Group information

### Question 1

> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

_Wo finden wir die bzw haben wir eine???_

### Question 2

> **Enter the study number for each member in the group**
>
> Example:
>
> _sXXXXXX, sXXXXXX, sXXXXXX_
>
> Answer:

Matrikelnummern:
**Kenny Kubsch**:
**Finn Schmidt**: 13046019

### Question 3

> **Did you end up using any open-source frameworks/packages not covered in the course during your project? If so**
> **which did you use and how did they help you complete the project?**
>
> Recommended answer length: 0-200 words.
>
> Example:
> _We used the third-party framework ... in our project. We used functionality ... and functionality ... from the_
> _package to do ... and ... in our project_.
>
> Answer:

The single most important third-party framework we used is **Ultralytics** (`ultralytics`), which provides the
YOLO (version 26) object-detection model. It is the backbone of our project: we wrapped its `YOLO` class in our own
`YOLOv26` class (`model.py`) and used its `train`, `predict` and `val` functionality to fine-tune a pretrained model
on our street-sign dataset and to run inference. We also relied on **OpenCV** (`opencv-python-headless`) for reading
images and drawing the predicted bounding boxes and class labels onto the returned images in both the FastAPI and
BentoML services. For reading the hand-curated class-mapping spreadsheet (`street_sign_class_mapping.xlsx`) we used
**openpyxl**. These packages were not part of the core course material but let us build a working detection pipeline
much faster than writing the equivalent code ourselves.

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> _We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a_
> _complete copy of our development environment, one would have to run the following commands_
>
> Answer:

We used **`uv`** as our package and environment manager. All dependencies are declared in `pyproject.toml`,
split into the main runtime dependencies and a `dev` dependency group (pytest, coverage, ruff, mypy, pre-commit,
mkdocs, etc.), and the exact resolved versions are pinned in `uv.lock`. To get an exact copy of the
environment, a new team member would need to run:

```bash
git clone <repo-url>
cd MLOPSStreetSignClassification
# install uv (https://docs.astral.sh/uv/)
uv sync --dev --locked
uv run dvc pull
```

`uv sync --dev --locked` recreates the virtual environment reproducibly, and `--locked` guarantees the versions
match `uv.lock`. The same command is used in every CI workflow, so the local and CI environments are identical.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> _From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder_
> _because we did not use any ... in our project. We have added an ... folder that contains ... for running our_
> _experiments._
>
> Answer:

We initialised the project with the [`mlops_template`](https://github.com/SkafteNicki/mlops_template) cookiecutter
template. We filled out the `src/street_sign_project/` package (renamed from `project_name`) with `data.py`,
`dataset.py`, `model.py`, `train.py`, `evaluate.py`, `visualize.py`, `fast_api.py`, `bentoml_api.py`,
`streamlit_app.py`, `link_model.py` and `utils.py`. We also completed the `configs/` (Hydra config + sweep),
`dockerfiles/`, `tests/`, `.github/workflows/`, `docs/` and `models/` folders and `tasks.py` (invoke commands).

We deviated from / extended the template in several ways: we added a `monitoring/` sub-package (image feature
extraction, production records, drift report, GCS storage), a `scripts/` folder with Cloud Run deployment shell
scripts, an `API_uploads/` folder for API input/output images, a `plots/` folder for data-statistics figures and
several extra CI workflows. Tests were split into `unittests/`, `apitests/` and `performancetests/`.

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Answer:

We used **ruff** for both linting and formatting (line length 120, rule sets `E`, `W`, `I`, `B`, `NPY`, `PD`),
configured in `pyproject.toml`. Formatting and lint-with-autofix run as **pre-commit hooks** and also as a dedicated
CI workflow (`codecheck.yaml`), so no unformatted or lint-failing code reaches `main`. We added **mypy** as a dev
dependency for static type checking, and we use **type hints throughout the codebase** (function signatures, dataclasses,
`Literal`/`TypeAlias` types). Every function and class has a **docstring**, as required in our `AGENTS.md`
style guide.

These concepts matter in larger projects because multiple people edit the same code: consistent formatting removes
noisy diffs and pointless style discussions, linting catches likely bugs and bad patterns early, and type hints make
interfaces explicit so that mistakes (e.g. passing the wrong type into `YOLOv26.train`) are caught before runtime.
Documentation lets a new team member understand a module without reading every line, which is essential when the
project grows beyond what one person can keep in their head.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Answer:

In total we implemented **37 tests**. In the unit tests we cover the data pipeline (class-mapping loading, split-ratio
validation, YAML/CSV creation, preprocessing), the `YOLOv26` model wrapper (input validation for `predict`, saving,
loading) and the training orchestrator (that Hydra config values are correctly wired into `YOLOv26.train`). A large
group of tests covers our monitoring code (image feature extraction, production records, reference features, GCS
storage, drift report). API tests check the FastAPI routes with a `TestClient`, and one performance test checks that
a staged model is fast enough for deployment.

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Answer:

The total code coverage of our source code is **66%** (measured with `coverage` over `src/street_sign_project`).
Coverage is high for the parts that are pure logic and easy to test in isolation — the data pipeline (`data.py`, 91%),
the monitoring modules (85–96%) and the training orchestrator (`train.py`, 93%) — and low for `model.py` (23%) and
`evaluate.py` (20%), because those wrap Ultralytics YOLO and would require downloading real model weights and running
inference to exercise fully.

Even if we had 100% coverage we would **not** trust the code to be error-free. Coverage only tells us which lines were
executed, not whether the assertions actually check the right behaviour, and not whether the model produces correct
predictions. Many bugs (wrong bounding-box math, data drift, bad hyperparameters, race conditions in the API
background tasks) live in the interaction between components and depend on the input data, none of which line coverage
can capture. Coverage is a useful floor, not a proof of correctness.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Answer:

Yes. During the project, we started working with **feature branches and pull requests** rather than committing directly to `main`. Most new
functionalities (e.g. the API, BentoML service, data drift monitoring, cloud build) was developed on its own branch
and merged into `main` through a PR, which is visible in our git history (e.g. "Merge pull request for adding cloud
trained model to dvc"). We kept a dedicated `continuous_ml` branch that our model-registry workflow checks out, so
that the automation could run against development code that was not yet on `main`. We also enabled **Dependabot**,
which opens weekly PRs to bump dependencies; we reviewed and merged those PRs like any other change.

Pull requests improved our version control because CI (tests, ruff, pre-commit) runs on every PR, so broken or
unformatted code is caught before it reaches `main`, and because a PR gives the other team member a chance to review
the changes.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Answer:

Yes, we used **DVC** with a **Google Cloud Storage remote** (`gs://mlops-street-signs/dvcstore`). We version-control
the raw data (`data/raw.dvc`, ~900 MB, 16 399 files), the preprocessed data (`data/preprocessed.dvc`) and our trained
models (e.g. `models/YOLO_eps420_bs8_lr0.005_fr10_x.pt.dvc`), while keeping the large binaries out of git.

DVC improved the project because both team members (and every CI runner) can reproduce the exact same dataset and
models with a single `dvc pull`, instead of passing files around manually. Because the data is tied to git commits,
we can always check out an old commit and get exactly the data/model that belonged to it, which is essential for
reproducibility of experiments. It also enabled our automated **data-checker workflow** (`cml_data.yaml`): whenever a
`.dvc` file changes, the workflow pulls the data, computes dataset statistics and posts them as a CML comment on the
pull request, so data changes are reviewed just like code changes.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> _We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing_
> _and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen_
> _here: <weblink>_
>
> Answer:

We organised our continuous integration into several dedicated GitHub Actions workflows. **`tests.yaml`** runs our
pytest suite via `uv run invoke test` on a **matrix** of operating systems (`ubuntu-latest` and `windows-latest`) and
Python versions (**3.12** and **3.13**), so we test four combinations; it uses the `astral-sh/setup-uv` action with
**caching** keyed on `uv.lock` to speed up dependency installation. **`codecheck.yaml`** runs `ruff check` and
`ruff format` as our linting/formatting step. **`pre_commits.yaml`** runs all pre-commit hooks. These run on pushes
and pull requests to `main`/`master`.

Beyond classic CI we added **continuous machine learning** workflows. **`cml_data.yaml`** triggers when any `.dvc`
file changes, pulls the data with DVC, computes dataset statistics and posts them as a CML comment on the PR.
**`stage_model.yaml`** is triggered by a `repository_dispatch` event from the W&B model registry: it performance-tests
a newly staged model and, on success, promotes it to the `production` alias. **`build_container.yaml`** and
**`build_container_train.yaml`** submit Cloud Build jobs to build our Docker images, but only when relevant paths
change (source, Dockerfiles, `uv.lock`), so we do not trigger expensive builds on cheap changes such as README edits.

An example test workflow can be seen here:
https://github.com/KenKbs/MLOPSStreetSignClassification/actions/workflows/pre_commits.yaml

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Answer:

We used **Hydra** with a config file (`configs/config.yaml`) that holds all paths, model, training and W&B settings.
Experiments are launched through an **invoke** task that composes the Hydra config and lets us override any value on
the command line, for example:

```bash
uv run invoke train --epochs 100 --batch-size 8 --lr0 0.005 --freeze 10
```

Any parameter not passed uses the default from `config.yaml`. Hyperparameter sweeps are configured in
`configs/sweep.yaml` and run with `uv run invoke tune`.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Answer:

Reproducibility is secured on several levels. The **environment** is pinned with `uv.lock` and installed with
`uv sync --locked`. The **data and models** are versioned with DVC and tied to git commits, so a given commit always
corresponds to the same data. The **experiment configuration** is a Hydra config file, and every override is explicit
on the command line, so there are no hidden magic numbers. We set a fixed **random seed** (default 420) that is passed
into Ultralytics training. Finally, every run was logged to **Weights & Biases**, while the account was still active:
we log the config (lr, batch size, epochs, freeze), the metrics (mAP50, precision, miss rate), per-epoch loss curves,
and we upload the trained model as a **W&B artifact**. To reproduce a run one checks out the corresponding commit,
runs `uv sync --locked` and `dvc pull`, and re-runs the same invoke command; the config and metrics can always be
inspected in the W&B run.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Answer:

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Answer:

We wrote several Docker images: a **training** image
(`dockerfiles/train.dockerfile`), an **evaluate** image (builds the models-quality YAML), an **API** image
(`api.dockerfile`, serves the FastAPI app) and a **frontend** image (`frontend.dockerfile`, the Streamlit app). In
addition we build a specialised **BentoML** image via `bentoml containerize`. A `docker-compose.yaml` wires the
train/evaluate/api services together with the right volume mounts. All Dockerfiles install dependencies in a separate,
cached layer (`uv sync --frozen --no-install-project`) before copying the source, so rebuilds are fast.

Examples of how we run them:

```bash
uv run invoke docker-build      # build train, evaluate and api images
uv run invoke docker-train      # run one training run in a container
uv run invoke docker-api        # serve the API on localhost:8000
```

Link to the API Dockerfile: `dockerfiles/api.dockerfile`. Images are also built in the cloud via Cloud Build
(`cloudbuild.yaml`) and deployed to Cloud Run with the scripts in `scripts/`.

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Answer:

For most logic bugs we relied on our **loguru** logging (info/warning/critical messages) and on plain
print/breakpoint debugging, plus the failing **pytest** tests which pointed us at the broken component. For the
cloud/CI parts (W&B model-registry automation, Cloud Build, Cloud Run) we debugged mostly through the workflow logs
and by iterating on small test commits.

We did **profile** our code: `tasks.py` contains a `profile-train` task that runs one training run under `cProfile`,
writes a `.prof` file to `reports/profiling/` and then opens it in **snakeviz** for visual inspection.
We do not think our code is perfect — most of the runtime is inside Ultralytics/PyTorch, which we cannot meaningfully
optimise ourselves, so we focused our own optimisation on the data pipeline and Docker layer caching.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Answer:

We used the following GCP services:

- **Cloud Storage (Buckets):** stores our DVC-tracked data and models (`gs://mlops-street-signs/dvcstore`)
- **Cloud Build:** builds our Docker images in the cloud from `cloudbuild.yaml` / `cloudbuild_train.yaml`, triggered
  by GitHub Actions.
- **Artifact Registry:** stores the built Docker images (API, evaluate, train, frontend, BentoML).
- **Cloud Run:** serves our deployed containers (FastAPI API, Streamlit frontend and the BentoML service) as
  scalable, managed HTTP services.
- **IAM / Service Accounts:** service-account keys stored as GitHub secrets let the CI authenticate to GCP.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Answer:

We used the **Compute Engine** for the compute-heavy part of the project — training our YOLO model. Because
fine-tuning YOLO on our combined street-sign dataset is far too slow on CPU, we started a Compute Engine VM with a
GPU and ran the training there. The resulting model was then downloaded via scp and version-controlled with DVC,
so both members could use it. Specifications: g2-standard-4 (4 vCPUs, 16 GB Memory); 1 x NVIDIA L4 GPU.

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

Hier bitte noch kurz bilder beschreiben.

`![bucket](figures/bucket.png)`

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

Screenshot of artefact registry for api in `/figures/registry.png`

`![registry](figures/registry.png)`

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

`![build](figures/build.png)`

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Answer:

Yes, we trained our model in the cloud using the **Compute Engine**. We pulled our code and the DVC-tracked data onto it, and ran the
training with our reproducible `uv`/Hydra setup (`uv run invoke train`, using the same training code and
`config.yaml` as locally). The resulting trained model was then downloaded via scp and version-controlled with DVC.
We chose the Engine (rather than Vertex AI) because it gave us a straightforward GPU machine on which our
existing training pipeline runs unchanged, which kept the setup simple and fully reproducible.

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Answer:

Yes, we wrote an API with **FastAPI** (`fast_api.py`). The model is loaded once on startup via a `lifespan` context
manager. The main endpoint `POST /image_input/` accepts an uploaded image, runs `YOLOv26.predict`, draws the detected
bounding boxes with the human-readable class names and confidence scores using OpenCV, and returns the annotated image
as a `FileResponse`. We use FastAPI **BackgroundTasks** to write a monitoring record (image
features + prediction summary) to GCS _after_ the response is sent, so monitoring does not add latency. A second
endpoint `GET /monitoring/` builds and returns a data-drift report as HTML. In addition to the general
FastAPI app we built a **specialised BentoML service** (`bentoml_api.py`), which exposes the same detection
functionality through BentoML's `@bentoml.service`/`@bentoml.api` decorators, so the model can be packaged and served
as an optimised ML deployment artifact.

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Answer:

Yes, we deployed both locally and in the cloud. Locally the API runs with
`uv run invoke start-local-api` (`localhost:8000`, docs at `/docs`). The production deployment is automated by a
GitHub Actions workflow. It runs the test suite, downloads the configured model from DVC, submits the FastAPI image to
Google Cloud Build, pushes commit-specific and `latest` tags to Artifact Registry, and deploys the immutable commit tag
to Cloud Run. `scripts/deploy_api_cloudrun.sh` remains available as a local fallback. The Streamlit frontend can be
started with `uv run invoke start-local-frontend` and deployed separately with `uv run invoke deploy-frontend`.

### Question 25

> **Did you perform any functional testing and load testing of your API? If yes, explain how you did it and what**
> **results for the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Answer:

Yes. For **functional testing** we used **pytest** with FastAPI's `TestClient` (`tests/apitests/test_api.py`): we
check that the OpenAPI schema exposes the `/image_input/` and `/monitoring/` routes, that the monitoring endpoint
returns Evidently HTML, and that an uploaded image returns an annotated `image/jpeg` response (with the model and
drawing code mocked). These tests run in CI. For **load testing** we used **Locust** (`tests/performancetests/
locustfile.py`), where simulated users repeatedly upload random test images to `/image_input/`. It is run with
`uv run invoke stress-api` (headless, defaults to 500 users, spawn rate 10, 2 minutes) or with a UI via
`--ui`. The test reports requests per second, response times and the failure rate, and fails a response if the status
is not 200 or the content type is not `image/jpeg`.

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Answer:

Yes, we implemented **data-drift monitoring**. For every request, the deployed API extracts tabular image features and a
prediction summary and writes them as a JSONL **production record** to a GCS bucket via a background task.
We also generated a **reference** feature set
from our training images. The `GET /monitoring/` endpoint loads the reference and production features from GCS and
uses **Evidently** (`DataDriftPreset`) to build an HTML drift report, so we can see whether the images arriving in
production differ from the training distribution. We did **not** implement system-metric instrumentation
(request counts/latency via Prometheus) or GCP alerting, which would be the natural next step.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Answer:

Finn: cloud usage was around 35$

In general, working in the cloud was a very positive experience: managed services like Cloud Run and Cloud Build let
us go from a local Docker image to a publicly reachable, autoscaling service with a single script, without having to
manage any servers ourselves. The main downsides were the amount of configuration and IAM/permission setup.

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Answer:

We implemented several extras. (1) A **Streamlit frontend** (`streamlit_app.py`, deployed to Cloud Run) so that a
non-technical user can upload an image and see the annotated predictions in the browser instead of using `curl`.

(5) A **custom stratified multi-label data split** (`data.py`) that keeps even rare street-sign classes present in
every train/valid/test split, which normal stratification cannot do because one image can contain several classes.
(6) A **models-quality tracking** mechanism that keeps the best models by a simplified AP@50 metric.

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Answer:

`![overview](figures/overview.png)`

The starting point of our system is the **local development setup**. We manage the environment with `uv`, configure
experiments with **Hydra**, and version our large data and models with **DVC**, whose remote is a **GCS bucket**. Code
quality is enforced locally by **pre-commit** hooks (ruff lint + format). When we commit and push to **GitHub**,
several **GitHub Actions** workflows are triggered: a test workflow (pytest on a matrix of Ubuntu/Windows and Python
3.12/3.13, with uv caching), a ruff code-check workflow, a pre-commit workflow, and a data-checker workflow that runs
whenever a `.dvc` file changes and posts dataset statistics as a CML comment on the PR.

Training happens either locally or in the cloud: the resulting
model was logged to **Weights & Biases** (before the test account was closed) and stored via DVC. W&B
also acted as our **model registry**: assigning the `staging` alias to a model artifact triggered, via
`repository_dispatch`, a workflow that performance-tests the model and, on success, promotes it to `production`.

For deployment, GitHub Actions triggers **Cloud Build** to build the API / frontend / BentoML Docker images, which are
stored in **Artifact Registry** and deployed to **Cloud Run**. The **FastAPI** service loads the chosen model and
exposes `/image_input/` for inference and `/monitoring/` for drift reports; a **Streamlit** frontend calls this API.
On every request, the API writes **monitoring records** (image features + predictions) to a GCS bucket, which
**Evidently** compares against a reference feature set to detect **data drift**. The end user interacts with the
system either through the Streamlit frontend or directly through the API.

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Answer:

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Answer:

Our group had two members, **Finn Schmidt** and **Kenny Kubsch**, and the work was closely shared, with both members
touching most parts of the codebase (both have roughly the same number of commits).

Student **Finn Schmidt** focused on the `YOLOv26` model wrapper (`model.py`), the evaluation/model-quality logic, the
W&B experiment tracking and the automated **model-registry pipeline** (`link_model.py`, `stage_model.yaml`), the **FastAPI** API, and the Streamlit frontend.

Student **Kenny Kubsch** focused on the data pipeline and preprocessing (`data.py`, class mapping and stratified
split), the training orchestrator (`train.py`), the DVC
data-versioning and the `cml_data` data-checker workflow, the **BentoML** API, the deployment scripts and
Cloud Run setup, and the monitoring / data-drift subsystem.

Both members contributed to the Docker setup, the CI workflows, the tests and the documentation, and reviewed each
other's pull requests.

**Finn Schmidt** did not use any generative AI tools within VS Code (except for filling out the report). He only used the Gemini Browser version as an "extended Google".
**Kenny Kubsch** did use an API access for Codex.
