# Training scripts from Google Colab notebooks

Seventeen PyTorch training scripts. Each follows a notebook that runs in Google Colab and
trains on Colab's own `/content/sample_data` files. They are here because the generated
scripts in `silent_bug_agent/mutation/` are all written one way, and real notebooks are not.
Notebook code does things the generator never did:
- updates weights by hand with `.data` (no optimizer);
- uses a summed loss with a tiny learning rate on purpose;
- uses `log_softmax` with `NLLLoss`;
- applies `reduction="none"` and then `.mean()`;
- trains a batch-norm net with sigmoid activations at a learning rate of 1.0;
- clips gradients at a learning rate of 5;
- trains on one sentence at a time.

| Script | Follows | Data | Split |
|---|---|---|---|
| colab_01_quickstart_mnist | PyTorch tutorial "Quickstart" (learning rate raised from 1e-3 to 1e-2) | MNIST (sample_data) | train |
| colab_02_training_with_pytorch | PyTorch tutorial "Training with PyTorch" | MNIST | dev |
| colab_03_cifar_tutorial_net | PyTorch tutorial "Training a Classifier" (1 channel, 28x28) | MNIST | test |
| colab_04_nn_tutorial | PyTorch tutorial "What is torch.nn really?" (the CNN) | MNIST | train |
| colab_05_nn_tutorial_from_scratch | the same notebook, raw tensors updated by hand | MNIST | test |
| colab_06_housing_regression | California housing regression, as in Colab's intro notebooks | California housing (sample_data) | train |
| colab_07_polynomial_fit | PyTorch "Learning PyTorch with Examples" (nn, manual update) | sin(x) | dev |
| colab_08_lstm_tagger | PyTorch "Sequence Models and LSTM Networks" (6 sentences) | inline | train |
| colab_09_ngram_language_model | PyTorch "Word Embeddings" (n-gram model, Sonnet 2) | inline | test |
| colab_10_char_rnn_names | PyTorch "Classifying Names with a Character-Level RNN" | inline surnames | test |
| colab_11_text_classification | PyTorch "Text classification" (EmbeddingBag, LR 5, clipping) | synthetic tokens in place of AG_NEWS | dev |
| colab_12_d2l_linear_regression_scratch | d2l.ai "Linear Regression Implementation from Scratch" | synthetic, as in the book | train |
| colab_13_d2l_mlp_concise | d2l.ai "Concise Implementation of MLPs" | MNIST in place of Fashion-MNIST | train |
| colab_14_d2l_dropout | d2l.ai "Dropout" | MNIST | test |
| colab_15_d2l_batchnorm_lenet | d2l.ai "Batch Normalization" (LeNet with BN) | MNIST | test |
| colab_16_autoencoder_mnist | the common Colab MNIST autoencoder | MNIST | train |
| colab_17_housing_binary | binary classification of California housing | California housing | dev |

The scripts are written in the structure of those notebooks (models, loops, hyperparameters,
idioms), adapted to read the sample data and to finish quickly on a CPU. They were not copied
line for line. The PyTorch tutorials are BSD-3-Clause licensed and d2l.ai's code is MIT-0.

**Data.** `fetch_sample_data.py` puts copies of Colab's sample files in
`~/.cache/silent_bug_agent/sample_data` (or `$SBA_SAMPLE_DATA`):
- the California housing CSVs, downloaded from Google's Machine Learning Crash Course, the files Colab ships;
- MNIST, written in Colab's CSV format (label, then 784 pixels) from the mirror torchvision uses: `mnist_train_small.csv` (20,000 rows) and `mnist_test.csv` (10,000 rows).

`variants.py` rewrites `SAMPLE_DATA = "/content/sample_data"` to point at the local copy.
Without the data, the scripts that need it are skipped.

**Splits** are by script, never by version:
- **train:** the classifier learns from these. The mutation operators plant their bugs, and each version is traced with three seeds.
- **dev:** settings are chosen on these. Bugs are planted by hand.
- **test:** only used to report results. Bugs are planted by hand, often in ways the operators never use:
  - labels sorted, or rolled by one;
  - a custom NLL loss that sums;
  - only the bias updated by hand;
  - a one-hot input of 255;
  - the batch-norm net left in training mode for its test batch.

**Label check.** A planted bug must make the run worse than the healthy version of the same
script (`silent_bug_agent/mutation/dataset.py`, `has_effect`), measured on a 300-step run
because some of these scripts barely move in 60. Five dev/test bugs failed that check and are
left out. They really are harmless here:
- a summed loss under gradient clipping, and under Adam;
- unscaled inputs into a batch-norm net;
- a frozen layer and shuffled targets in two scripts that barely learn in 300 steps anyway.
