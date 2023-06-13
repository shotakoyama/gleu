# GLEU

This repository provides a python re-implementation code of the GLEU metric (https://github.com/cnap/gec-ranking/).

# Problem in GLEU implementation

The formula in the paper differs from that in the official implementation. 

Let us denote:

$s_i$: i-th source sentence (system input; The data size is $|D|$.)

$r_i$: i-th reference sentence (human-written corrected sentence)

$h_i$: i-th hypothesis sentence (system output)

$g_n$: n-gram

$\sigma_{i,g_n}=\mathrm{count}(g_n \in s_i)$ (the number of occurrences of $g_n$ in $s_i$)

$\rho_{i,g_n}=\mathrm{count}(g_n \in r_i)$ (the number of occurrences of $g_n$ in $r_i$)

$\eta_{i,g_n}=\mathrm{count}(g_n \in h_i)$ (the number of occurrences of $g_n$ in $h_i$)

$\sigma_{i,g_n}^{\mathrm{diff}}=[\rho_{i,g_n}=0]\sigma_{i,g_n}$ ($\sigma_{i,g_n}$ if $\rho_{i,g_n}=0$ else $0$) (https://en.wikipedia.org/wiki/Iverson_bracket)

The GLEU+ paper (https://arxiv.org/pdf/1605.02592.pdf) shows that the formula of precision $p_k$ is: 

$$ p_n = \cfrac{ \displaystyle \sum_i^{|D|} \sum_{g_n \in h_i} \min(\rho_{i,g_n}, \eta_{i, g_n}) - \max(0, \min(\sigma_{i, g_n}, \eta_{i, g_n}) - \min(\rho_{i, g_n}, \eta_{i, g_n}))} {\displaystyle \sum_i^{|D|} \sum_{g_n \in h_i} \eta_{i, g_n} } $$

However, GLEU+ in source code is:

$$ p_n = \cfrac{ \displaystyle \sum_i^{|D|} \max(0,  \sum_{g_n \in h_i} \min(\rho_{i,g_n}, \eta_{i, g_n}) - \min(\sigma_{i, g_n}^{\mathrm{diff}}, \eta_{i, g_n})) } {\displaystyle \sum_i^{|D|} \sum_{g_n \in h_i} \eta_{i, g_n} } $$

https://github.com/shotakoyama/gleu/blob/d20b995be142ff40a7e342cfe8e866a1fce09073/ref/gleu.py#L95-L105

These two formulae are not equivalent because the penalty terms differ. For example, let $(\sigma, \rho, \eta) = (2, 1, 3)$, $\max(0, \min(\sigma, \eta) - \min(\rho, \eta)) = 1$, while $\min(\sigma^{\mathrm{diff}}, \eta) = 0$.

Furthermore, the equation above equals to the next equation. In our implementation, we adopted this.

$$ p_n = \cfrac{ \displaystyle \sum_i^{|D|} \sum_{g_n \in h_i} \min(\rho_{i,g_n}, \eta_{i, g_n}) - \sum_i^{|D|} \min( \sum_{g_n \in h_i} \min(\sigma_{i, g_n}^{\mathrm{diff}}, \rho_{i,g_n}) \sum_{g_n \in h_i} \min(\rho_{i,g_n}, \eta_{i, g_n}) ) } {\displaystyle \sum_i^{|D|} \sum_{g_n \in h_i} \eta_{i, g_n} } $$

# Usage

You can install the code by running `pip install -e .` under the directry with `setup.py`. `pip install gleu` is also OK.

## corpus-level GLEU

Run the code below to get the same result as the original implementation. `-d` or `--digit` specifies the number of digits of decimal places. `-f` or `--fix-seed` is required to reproduce the original result.

```
$ gleu corpus -s INPUT -r REF0 REF1 -o AMU CAMB INPUT REF0 -d 4 -f
AMU     58.3256
CAMB    59.2553
INPUT   56.6048
REF0    81.8462
```

This result equals to the original GLEU+ output.

```
$ python2 gec-ranking/scripts/compute_gleu -s INPUT -r REF0 REF1 -o AMU CAMB INPUT REF0
AMU 0.583256
CAMB 0.592553
INPUT 0.566048
REF0 0.818462
```

`-p` is the number of process (default: 1). In my environment, `-p 8` is the fastest. `-n` is the number of the max size of n-gram (default: 4). `-i` is the number of iterations of reference sampling. `-t` specifies tokenization method (choices: `word` or `char`, default: `word`).

```
$ gleu corpus -s INPUT -r REF0 REF1 -o AMU CAMB INPUT REF0 -p 8 -n 6 -i 1000 -t char
AMU     83.06
CAMB    83.41
INPUT   83.30
REF0    92.22
```

Although default mode samples from multiple refereces, max mode (`-m`) uses the best reference for each sentence.

```
$ gleu corpus -s INPUT -r REF0 REF1 -o AMU CAMB INPUT REF0 -p 8 -m
AMU     68.81
CAMB    68.55
INPUT   68.34
REF0    100.00

$ gleu corpus -s INPUT -r REF0 REF1 -o AMU CAMB -p 8 -mv
AMU
+-------+-------+--------+-------+--------+-------+
|       | numer | denom  |   p   |   bp   |  gleu |
+-------+-------+--------+-------+--------+-------+
|   1   | 26538 | 30362  | 87.41 | 100.00 | 87.41 |
|   2   | 21521 | 29050  | 74.08 | 100.00 | 74.08 |
|   3   | 17600 | 27739  | 63.45 | 100.00 | 63.45 |
|   4   | 14423 | 26428  | 54.57 | 100.00 | 54.57 |
| total | 80082 | 113579 | 68.81 | 100.00 | 68.81 |
+-------+-------+--------+-------+--------+-------+
CAMB
+-------+-------+--------+-------+-------+-------+
|       | numer | denom  |   p   |   bp  |  gleu |
+-------+-------+--------+-------+-------+-------+
|   1   | 26154 | 29859  | 87.59 | 99.36 | 87.03 |
|   2   | 21156 | 28547  | 74.11 | 99.36 | 73.63 |
|   3   | 17327 | 27236  | 63.62 | 99.36 | 63.21 |
|   4   | 14227 | 25925  | 54.88 | 99.36 | 54.53 |
| total | 78864 | 111567 | 69.00 | 99.36 | 68.55 |
+-------+-------+--------+-------+-------+-------+
```

## sentence-level GLEU

```
$ gleu sent -s INPUT -r REF0 REF1 -o AMU CAMB INPUT REF0
100.00  100.00  100.00  100.00
100.00  0.00    100.00  100.00
53.34   0.00    53.34   100.00
0.00    67.53   0.00    100.00
84.92   56.30   100.00  100.00
100.00  80.96   100.00  100.00
100.00  0.00    100.00  100.00
92.93   92.93   92.93   92.93
86.35   72.42   86.35   100.00
26.93   0.00    0.00    69.62
...
```

```
$ gleu sent -s INPUT -r REF0 REF1 -o AMU CAMB INPUT REF0 -m
100.00  100.00  100.00  100.00
100.00  0.00    100.00  100.00
53.34   0.00    53.34   100.00
0.00    67.53   0.00    100.00
84.92   56.30   100.00  100.00
100.00  80.96   100.00  100.00
100.00  0.00    100.00  100.00
100.00  100.00  100.00  100.00
86.35   72.42   86.35   100.00
...
```

```
$ gleu sent -s INPUT -r REF0 REF1 -o AMU -v
S-1     Keeping the Secret of Genetic Testing
H-1-1   Keeping the Secret of Genetic Testing
R-1-1*  Keeping the Secret of Genetic Testing
+-------+-------+-------+-------+-------+--------+--------+--------+
|       | match | penal | numer | denom |   p    |   bp   |  gleu  |
+-------+-------+-------+-------+-------+--------+--------+--------+
|   1   |   6   |   0   |   6   |   6   | 100.00 | 100.00 | 100.00 |
|   2   |   5   |   0   |   5   |   5   | 100.00 | 100.00 | 100.00 |
|   3   |   4   |   0   |   4   |   4   | 100.00 | 100.00 | 100.00 |
|   4   |   3   |   0   |   3   |   3   | 100.00 | 100.00 | 100.00 |
| total |   18  |   0   |   18  |   18  | 100.00 | 100.00 | 100.00 |
+-------+-------+-------+-------+-------+--------+--------+--------+
S-1     Keeping the Secret of Genetic Testing
H-1-1   Keeping the Secret of Genetic Testing
R-1-2   Keeping the Secret of Genetic Testing
+-------+-------+-------+-------+-------+--------+--------+--------+
|       | match | penal | numer | denom |   p    |   bp   |  gleu  |
+-------+-------+-------+-------+-------+--------+--------+--------+
|   1   |   6   |   0   |   6   |   6   | 100.00 | 100.00 | 100.00 |
|   2   |   5   |   0   |   5   |   5   | 100.00 | 100.00 | 100.00 |
|   3   |   4   |   0   |   4   |   4   | 100.00 | 100.00 | 100.00 |
|   4   |   3   |   0   |   3   |   3   | 100.00 | 100.00 | 100.00 |
| total |   18  |   0   |   18  |   18  | 100.00 | 100.00 | 100.00 |
+-------+-------+-------+-------+-------+--------+--------+--------+
...
```

```
 $ gleu mean -s INPUT -r REF0 REF1 -o AMU CAMB INPUT REF0
AMU     54.12
CAMB    54.33
INPUT   50.67
REF0    79.63

$ gleu mean -s INPUT -r REF0 REF1 -o AMU CAMB INPUT REF0 -m
AMU     66.70
CAMB    65.55
INPUT   64.53
REF0    100.00
```

