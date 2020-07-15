# winograd

A simple python file for computing the coefficient matrix in Winograd convolution algorithms.

## Requirements

+ python: version 3.7
+ pytorch
+ torchvision
+ sympy: version 1.0
+ numpy


## Tips

+ Support any stride
+ The choice of polynomial interpolation points will affect the accuracy of final results. Here we use the same values with the original paper[1].  

## Useage
Use '--h' for help
Option '--n' represents the size of input tile, '--r' for filter size, and '--s' for stride. 

```bash
python main.py --n=4 --r=3 --s=1
```

The output looks like

```text
AT =
⎡1  1  1   1  1   0⎤
⎢                  ⎥
⎢0  1  -1  2  -2  0⎥
⎢                  ⎥
⎢0  1  1   4  4   0⎥
⎢                  ⎥
⎣0  1  -1  8  -8  1⎦


G =
⎡1/4     0     0  ⎤
⎢                 ⎥
⎢-1/6  -1/6   -1/6⎥
⎢                 ⎥
⎢-1/6   1/6   -1/6⎥
⎢                 ⎥
⎢1/24  1/12   1/6 ⎥
⎢                 ⎥
⎢1/24  -1/12  1/6 ⎥
⎢                 ⎥
⎣ 0      0     1  ⎦


BT =
⎡4  0   -5  0   1  0⎤
⎢                   ⎥
⎢0  -4  -4  1   1  0⎥
⎢                   ⎥
⎢0  4   -4  -1  1  0⎥
⎢                   ⎥
⎢0  -2  -1  2   1  0⎥
⎢                   ⎥
⎢0  2   -1  -2  1  0⎥
⎢                   ⎥
⎣0  4   0   -5  0  1⎦


Output correct.
```

[1] "Fast Algorithms for Convolutional Neural Networks" Lavin and Gray, CVPR 2016.
http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Lavin_Fast_Algorithms_for_CVPR_2016_paper.pdf
https://github.com/andravin/wincnn