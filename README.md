# NerualFramework

## Usage
### Model
Write your model in directory `models/` and register your model in `models/__init__.py` as follows:
```python
from .your_model import YourModel

ALL_MODELS = {
    ...
    'your_model': YourModel,
}
```
We provide a `BaseModel` class in `models/base.py` for you to inherit.

### Dataset
Write your dataset in directory `datasets/` and register your dataset in `datasets/__init__.py` as follows:
```python
from .your_dataset import YourDataset
```
We provide a `BaseDataset` class in `datasets/base.py` for you to inherit. 

### Trainer
Write your trainer and procedure method in directory `trainers/` and register your trainer in `trainers/__init__.py` as follows:
```python
from .your_trainer import YourTrainer, your_procedure
```
We provide a `BaseTrainer` class and a `base_procedure` method in `trainers/base.py` for you to inherit. 

### Config
Write your config in directory `configs/`.
```shell
python main.py --config configs/your_config.yaml
```