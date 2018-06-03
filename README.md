Classification from Pairwise Similarity and Unlabeled Data
====

This repository provides an official implementation of _SU classification_,
which is a weakly-supervised classification problem only from pairwise similarity pairs (two data points belong to the same class) and unlabeled data points.

![image](https://github.com/levelfour/SU_Classification/raw/master/image.png)

## Dependencies

```
cvxopt==1.1.9
numpy==1.13.3
sklearn==0.18.1
```

## Run

```bash
python su_learning.py --loss squared --ns 200 --nu 200 --prior 0.7
```

## Notes

`mpe.py` is a (slightly-modified) implementation of mixture proportion estimation.
We used the author's implementation available [here](http://web.eecs.umich.edu/~cscott/code.html#kmpe).

## References

+ Bao, H., Niu, G., & Sugiyama, M. Classification from Pairwise Similarity and Unlabeled Data. In _Proceedings of International Conference on Machine Learning (ICML)_, 2018. [[arxiv]](https://arxiv.org/abs/1802.04381)
+ Ramaswamy, H. G., Scott, C., & Tewari, A. Mixture proportion estimation via kernel embedding of distributions. In _Proceedings of International Conference on Machine Learning (ICML)_, pp. 2052–2060, 2016.
