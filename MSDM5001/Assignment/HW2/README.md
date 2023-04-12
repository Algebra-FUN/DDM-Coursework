# Guide for running these code

## 2.1

```bash
mpiexec -n 10 python 2.1.py
```

## 2.2

Running with default argument n (n=1000)
```bash
python 2.2.py
```

Running with specified argument n
```bash
python 2.2.py -n 2000
```

## 2.3

In order to reproduct the result, I add some code to manual random seed.

The default arguments
```python
parser = ArgumentParser()
parser.add_argument("-n", type=int, default=10000)
parser.add_argument("--seed", type=int, default=123)
```

Running with default argument:
```bash
python 2.3.py
```

Running with specified argument:
```bash
python 2.3.py -n 1000 --seed 114
```