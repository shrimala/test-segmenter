# text-segmenter
A python library for segmenting text with controllable length 

# Negative Binomial Distribution
success: the successful placement of a section boundary within the text.
failures: segments/sentences between section boundaries
$$\\P(X = k) = \binom{k + r - 1}{k} \cdot (1-p)^r \cdot p^k$$
