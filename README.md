# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py




Passing Tasks in Collab:

![tasks 1-2](https://github.com/user-attachments/assets/a1f83540-5e81-426c-a66d-b549d930af65)
![task 3](https://github.com/user-attachments/assets/75e4a0e4-34aa-4686-86a3-c6959c638ef5)
![task 4](https://github.com/user-attachments/assets/7c37e774-3b6a-4de1-a242-8f39cf46e1e9)


3.4 Comparison Graph:


![matrix_multiplication_benchmark](https://github.com/user-attachments/assets/0cdb24dd-011b-445e-9d0f-43f172a99884)

3.5 :

# Simple

## CPU Simple

![cpu simple](https://github.com/user-attachments/assets/c70dfb6b-e671-4bcb-8c6d-c81723270703)

## GPU Simple
![gpu simple](https://github.com/user-attachments/assets/fef86e33-1b57-4a64-a620-55ee15d60d32)

# Xor

## GPU XOR

![xor gpu](https://github.com/user-attachments/assets/d1685923-0930-4352-949a-b118a95305a8)


## CPU XOR

![xor cpu](https://github.com/user-attachments/assets/44121f31-26f1-4bc6-88cf-f135b23f5eb4)

# Split

## GPU Split

![split hidden 100, rate 0 05](https://github.com/user-attachments/assets/2bc5be1c-7c89-452f-ba46-8bbef2cf5ba2)

## CPU Split

![split cpu](https://github.com/user-attachments/assets/00ae689e-0b50-4167-927c-7280318cf194)



# Larger Model - Simple 200

## GPU Simple 200

![200 GPU simple 200](https://github.com/user-attachments/assets/6c7f604c-cc25-4119-8ebb-1d699a71be15)


## CPU Simple 200

![200 CPU simple 200](https://github.com/user-attachments/assets/00c774a6-e6bd-45bd-b2eb-57710c16b507)



