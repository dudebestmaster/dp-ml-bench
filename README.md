# dp-ml-bench
Empirical privacy–utility experiments: DP-SGD (Opacus) on Adult dataset. Figures, scripts, and reproducible runs.

Main finding (summary): On the full Adult dataset, test accuracy (~85%) is robust across a wide range of privacy budgets. When the training set is substantially reduced (≈3k samples) strong privacy (low ε) causes a clear accuracy drop.

Results & interpretation

We preprocess the UCI Adult dataset by removing rows with missing fields, one-hot encoding categorical features, and standard scaling. The model is a small MLP (one hidden layer, 64 units, ReLU). For DP experiments we use Opacus’ PrivacyEngine with per-sample gradient clipping and Gaussian noise (DP-SGD). We report ε computed by Opacus (δ = 1e-5) and test accuracy on a held-out test set. Full configs and exact commands are in reproduction.md.


The full dataset shows a privacy–utility plateau: moderate privacy levels (ε in the 1–5 range) produce minimal accuracy loss relative to the non-private baseline. However, when training data is reduced (e.g., ≈3k examples), accuracy degrades significantly under strong privacy, demonstrating the interaction between dataset size and DP noise. See fig_accuracy_vs_epsilon.png and note_dp_adult_1page.pdf for details.

How to cite / contact

If you use this code or figures, please cite the repo and/or contact me for collaboration:
Shaurya Singh — University of Waterloo
Repo: https://github.com/dudebestmaster/dp-ml-bench
Email: shaurya.singh12006@gmail.com

License

MIT(See License)
