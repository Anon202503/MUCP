# Redefining Machine Unlearning: A Conformal Prediction-Motivated Approach (code)

This is the code for the paper **Redefining Machine Unlearning: A Conformal Prediction-Motivated Approach**.

## File Tree

Project file structure and description:

```
MUCP
├─ README.md
├─ requirements.txt
├─ metrics	# package of metrics (CR and MIACR)
│    ├─ CR.py
│    ├─ MIACR.py
├─ models	# package of models (ResNet-18 and Vit)
│    ├─ resnet.py
│    ├─ vit.py
├─ main_original_model.py
├─ main_unlearn.py
├─ main_unlearn_framework.py
├─ main_evaluate.py
├─ unlearn.py
├─ unlearn_framework.py
└─ utils.py
```

## Installation

Installation requirements are described in `requirements.txt`.

- Use pip:

  ```
  pip install -r requirements.txt
  ```

- Use anaconda:

  ```
  conda install --file requirements.txt
  ```

## Usage

You need to train an original model with the ResNet-18 or ViT architecture. Use `main_original_model.py` for this:

```
python main_original_model.py --model_name resnet18 --data_name cifar10 --data_dir ./data --batch_size 64 --num_epochs 200 --learning_rate 0.1 --num_classes 10 
```

To use the implemented logging, you’ll need a `wandb.ai` account. Alternatively, you can replace it with any logger of your preference.

You can train an unlearning model with one of the existing unlearning methods via `main_unlearn.py` :

```
python main_unlearn.py --unlearn_name retrain --unlearn_type random --model_name resnet18 --data_name cifar10 --data_dir ./data --model_dir original_model.pth --num_epochs 200 --num_classes 10 --retain_ratio 0.9 --learning_rate 0.01
```

Or you can train an unlearning model with our unlearning framework via `main_unlearn_framework.py` :

```
python main_unlearn_framework.py --unlearn_name retrain --unlearn_type random --model_name resnet18 --data_name cifar10 --data_dir ./data --model_dir original_model.pth --num_epochs 200 --num_classes 10 --retain_ratio 0.9 --learning_rate 0.1 --delta 0.01 --alpha 0.05 --lamda 1
```

After unlearning the forget data, you can use `main_evaluate.py` to measure the unlearning model's performance by `CR` and `MIACR` metrics:

```python
python main_evaluate.py --unlearn_name retrain --unlearn_type random --model_name resnet18 --data_name cifar10 --data_dir ./data --model_dir unlearning_model.pth --num_classes 10 --retain_ratio 0.9 --alphas 0.05,0.1,0.15,0.2
```

























