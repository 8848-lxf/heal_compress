# Importance normalization

The default strategy name is
`coupled_dependency_mean_then_scope_mean_v1`; implementation is in
`pruning/importance/first_order_taylor.py` and
`pruning/importance/normalization.py`.

For dependency member \(m\), root channel \(c\), and its parameter slice
\(P_{m,c}\):

\[
t_{m,c}=\sum_{p\in P_{m,c}}\left|w_p\frac{\partial L}{\partial w_p}\right|.
\]

For the finite scored members \(M_{s,c}\) in dependency scope \(s\):

\[
r_{s,c}=\frac{1}{|M_{s,c}|}\sum_{m\in M_{s,c}}t_{m,c}.
\]

Let \(F_s\) be the channels with finite raw scores. The normalized score is:

\[
\hat r_{s,c}=\frac{r_{s,c}}
{\frac{1}{|F_s|}\sum_{j\in F_s}r_{s,j}+\epsilon},\qquad \epsilon=10^{-12}.
\]

If the finite scope mean is exactly zero, all finite normalized scores are
zero. Missing gradients or unresolved member slices remain infinite and cannot
be selected by lowest-score ranking. This is the verified v10.8 formula; it was
not replaced with parameter-count, kernel-size or fan-in normalization during
formalization. L1, L2 and second-order Fisher remain explicit optional modes.

