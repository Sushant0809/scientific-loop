from dataclasses import dataclass, field
from typing import Dict, List, Literal
import random

Split = Literal["warmup", "train", "eval"]


@dataclass
class Paper:
    paper_id: str
    title: str
    abstract: str
    methodology: str                   # What the agent reads — no target values
    target_metrics: Dict[str, float]   # Ground truth — NEVER shown to agent
    metric_weights: Dict[str, float]   # Importance weight per metric (sum = 1.0)
    dataset_hint: str
    difficulty: Literal["warmup", "easy", "medium", "hard"]
    execution_timeout: int             # seconds; keep ≤55s for CPU safety margin
    split: Split                       # "warmup" | "train" | "eval"


PAPER_CORPUS: List[Paper] = [

    # ─────────────────────────────────────────────────────────────────────────
    # WARMUP PAPERS  (3)
    # ─────────────────────────────────────────────────────────────────────────

    Paper(
        paper_id="linear_regression_gd",
        title="Gradient Descent for Linear Regression (Rumelhart et al., 1986)",
        abstract="Gradient descent is an iterative optimization algorithm for finding "
                 "the minimum of a differentiable function by following the negative gradient.",
        methodology=(
            "Generate synthetic data: 200 samples, x ~ Uniform(-1,1), y = 3*x + 2 + noise(std=0.1). "
            "Implement linear regression y_hat = w*x + b from scratch using numpy only (no torch). "
            "Initialize w=0.0, b=0.0. Use gradient descent for 500 steps with lr=0.1. "
            "At each step: compute MSE loss, compute gradients dL/dw and dL/db analytically, update w and b. "
            "After training report: final_mse (MSE on training set), learned_w (should be ~3.0), learned_b (should be ~2.0)."
        ),
        target_metrics={"final_mse": 0.01, "learned_w": 3.0, "learned_b": 2.0},
        metric_weights={"final_mse": 0.5, "learned_w": 0.3, "learned_b": 0.2},
        dataset_hint="Synthetic (numpy only, no downloads)",
        difficulty="warmup",
        execution_timeout=10,
        split="warmup",
    ),

    Paper(
        paper_id="logistic_regression_scratch",
        title="A Training Algorithm for Optimal Margin Classifiers (Boser et al., 1992)",
        abstract="We describe a training algorithm that maximizes the margin of the decision boundary "
                 "for binary classification problems.",
        methodology=(
            "Implement logistic regression from scratch using numpy only (no torch, no sklearn). "
            "Generate synthetic binary data: 300 samples, 2 features, linearly separable with some noise. "
            "Use numpy random seed 42. Class 0: center (-1,-1), Class 1: center (1,1), std=0.5. "
            "Model: p = sigmoid(w^T x + b). Loss: binary cross-entropy. "
            "Train with gradient descent for 200 steps, lr=0.1. "
            "Report: test_accuracy (80/20 split), final_loss, num_steps_trained=200."
        ),
        target_metrics={"test_accuracy": 0.95, "final_loss": 0.15, "num_steps_trained": 200.0},
        metric_weights={"test_accuracy": 0.7, "final_loss": 0.2, "num_steps_trained": 0.1},
        dataset_hint="Synthetic (numpy only, no downloads)",
        difficulty="warmup",
        execution_timeout=10,
        split="warmup",
    ),

    Paper(
        paper_id="perceptron_xor",
        title="Learning Representations by Back-propagating Errors (Rumelhart et al., 1986)",
        abstract="We describe a new learning procedure for networks of neuron-like units that adjusts "
                 "weights by back-propagating errors through the network.",
        methodology=(
            "Implement a 2-layer MLP (no torch.nn, use numpy only) to solve XOR. "
            "Architecture: input(2) -> hidden(4, tanh) -> output(1, sigmoid). "
            "XOR dataset: 4 samples [(0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0]. "
            "Initialize weights with numpy random seed 0, values in [-1, 1]. "
            "Train with backpropagation for 5000 steps, lr=0.5, MSE loss. "
            "Report: final_loss (should be <0.01), xor_accuracy (number of correct predictions out of 4)."
        ),
        target_metrics={"final_loss": 0.005, "xor_accuracy": 4.0},
        metric_weights={"final_loss": 0.5, "xor_accuracy": 0.5},
        dataset_hint="XOR (synthetic, numpy only, no downloads)",
        difficulty="warmup",
        execution_timeout=10,
        split="warmup",
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # TRAIN PAPERS  (10)
    # ─────────────────────────────────────────────────────────────────────────

    Paper(
        paper_id="adam_optimizer",
        title="Adam: A Method for Stochastic Optimization (Kingma & Ba, 2015)",
        abstract="We introduce Adam, an algorithm for first-order gradient-based optimization "
                 "of stochastic objective functions, based on adaptive estimates of lower-order moments.",
        methodology=(
            "Implement the Adam optimizer from scratch using PyTorch tensors but NOT torch.optim.Adam. "
            "Parameters: lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8. "
            "Apply it to train a 2-layer MLP (Linear(784,128)->ReLU->Linear(128,10)) on MNIST. "
            "Load MNIST from torchvision.datasets.MNIST(download=True). "
            "Train for 3 epochs, batch size 256. "
            "Manually implement: m = beta1*m + (1-beta1)*grad, v = beta2*v + (1-beta2)*grad^2, "
            "m_hat = m/(1-beta1^t), v_hat = v/(1-beta2^t), param -= lr * m_hat / (sqrt(v_hat) + eps). "
            "Report: test_accuracy (0-100 scale), final_train_loss."
        ),
        target_metrics={"test_accuracy": 97.0, "final_train_loss": 0.10},
        metric_weights={"test_accuracy": 0.8, "final_train_loss": 0.2},
        dataset_hint="MNIST (torchvision, downloads ~11MB)",
        difficulty="easy",
        execution_timeout=40,
        split="train",
    ),

    Paper(
        paper_id="dropout_regularization",
        title="Dropout: A Simple Way to Prevent Neural Networks from Overfitting (Srivastava et al., 2014)",
        abstract="We describe a technique called dropout for addressing overfitting. "
                 "The key idea is to randomly drop units from the neural network during training.",
        methodology=(
            "Train two 3-layer MLPs on MNIST and compare them. "
            "Model A (with Dropout): Linear(784,256)->ReLU->Dropout(p=0.5)->Linear(256,128)->ReLU->Dropout(p=0.5)->Linear(128,10). "
            "Model B (no Dropout): same architecture, no Dropout layers. "
            "Use the same random seed (42) for both. Adam lr=1e-3, batch size 128, 3 epochs. "
            "Load MNIST from torchvision. Report: "
            "dropout_test_acc (0-100), nodropout_test_acc (0-100), "
            "dropout_improvement = dropout_test_acc - nodropout_test_acc."
        ),
        target_metrics={"dropout_test_acc": 97.8, "nodropout_test_acc": 97.2, "dropout_improvement": 0.6},
        metric_weights={"dropout_test_acc": 0.5, "nodropout_test_acc": 0.3, "dropout_improvement": 0.2},
        dataset_hint="MNIST (torchvision)",
        difficulty="easy",
        execution_timeout=40,
        split="train",
    ),

    Paper(
        paper_id="reinforce_cartpole",
        title="Simple Statistical Gradient-Following Algorithms for Connectionist RL (Williams, 1992)",
        abstract="A general class of associative reinforcement learning algorithms for connectionist "
                 "networks is introduced, and it is shown that the weight updates for the special case "
                 "of the REINFORCE algorithm are proportional to the gradient of expected reinforcement.",
        methodology=(
            "Implement the REINFORCE (Monte Carlo Policy Gradient) algorithm on CartPole-v1. "
            "Policy network: Linear(4,128)->ReLU->Linear(128,2), output passed through softmax. "
            "Each episode: run until done, collect (log_prob, reward) pairs. "
            "Compute discounted returns G_t with gamma=0.99, normalize them (subtract mean, divide by std). "
            "Loss = -sum(log_prob * G_t). Update with Adam lr=1e-3. "
            "Train for 400 episodes. Report: mean_reward_last_50 (mean episode reward over last 50 eps), "
            "solved = 1.0 if mean_reward_last_50 >= 195 else 0.0."
        ),
        target_metrics={"mean_reward_last_50": 180.0, "solved": 0.0},
        metric_weights={"mean_reward_last_50": 0.9, "solved": 0.1},
        dataset_hint="CartPole-v1 (gymnasium)",
        difficulty="easy",
        execution_timeout=40,
        split="train",
    ),

    Paper(
        paper_id="vae_mnist",
        title="Auto-Encoding Variational Bayes (Kingma & Welling, 2014)",
        abstract="We introduce a stochastic variational inference and learning algorithm that scales "
                 "to large datasets and is not limited to models with simple posterior distributions.",
        methodology=(
            "Implement a Variational Autoencoder (VAE) on MNIST. "
            "Encoder: Linear(784,400)->ReLU, then two parallel heads Linear(400,20) for mu and log_var. "
            "Reparameterization: z = mu + eps * exp(0.5 * log_var) where eps ~ N(0,I). "
            "Decoder: Linear(20,400)->ReLU->Linear(400,784)->Sigmoid. "
            "Loss: reconstruction (BCE pixel-wise, sum over pixels) + KL divergence "
            "(-0.5 * sum(1 + log_var - mu^2 - exp(log_var))). "
            "Adam lr=1e-3, batch size 128, train 2 epochs on MNIST. "
            "Report: test_elbo (negative, lower magnitude is better, e.g. -90), "
            "recon_loss (reconstruction part of ELBO on test set)."
        ),
        target_metrics={"test_elbo": -105.0, "recon_loss": 85.0},
        metric_weights={"test_elbo": 0.6, "recon_loss": 0.4},
        dataset_hint="MNIST (torchvision)",
        difficulty="medium",
        execution_timeout=45,
        split="train",
    ),

    Paper(
        paper_id="dqn_cartpole",
        title="Human-level Control Through Deep Reinforcement Learning (Mnih et al., 2015)",
        abstract="We present the first artificial agent to achieve human-level performance across "
                 "a diverse range of Atari 2600 games using only raw pixels and game score.",
        methodology=(
            "Implement DQN with experience replay and a target network on CartPole-v1. "
            "Q-Network: Linear(4,64)->ReLU->Linear(64,64)->ReLU->Linear(64,2). "
            "Replay buffer: deque of capacity 5000. Min buffer size before training: 500. "
            "Batch size: 64. Target network: hard update every 50 episodes. "
            "Epsilon-greedy: start=1.0, end=0.05, decay multiplicative by 0.995 each episode. "
            "Gamma=0.99, Adam lr=5e-4. Train 200 episodes. "
            "Report: mean_reward_last_50 (mean over last 50 episodes), epsilon_final."
        ),
        target_metrics={"mean_reward_last_50": 150.0, "epsilon_final": 0.08},
        metric_weights={"mean_reward_last_50": 0.9, "epsilon_final": 0.1},
        dataset_hint="CartPole-v1 (gymnasium)",
        difficulty="medium",
        execution_timeout=45,
        split="train",
    ),

    Paper(
        paper_id="batch_norm_mnist",
        title="Batch Normalization: Accelerating Deep Network Training (Ioffe & Szegedy, 2015)",
        abstract="Training Deep Neural Networks is complicated by the fact that the distribution of "
                 "each layer's inputs changes during training. We refer to this as internal covariate shift.",
        methodology=(
            "Train two MLPs on MNIST and compare convergence speed and accuracy. "
            "Model A (with BatchNorm): Linear(784,256)->BN(256)->ReLU->Linear(256,128)->BN(128)->ReLU->Linear(128,10). "
            "Model B (baseline): same without BN layers. Same seed (0), SGD lr=0.1, momentum=0.9, batch=128, 3 epochs. "
            "Record test accuracy at each epoch for both. "
            "Report: bn_test_acc (final epoch, 0-100), baseline_test_acc (final epoch, 0-100), "
            "bn_epoch1_acc (accuracy after epoch 1 -- measures faster convergence)."
        ),
        target_metrics={"bn_test_acc": 97.5, "baseline_test_acc": 96.0, "bn_epoch1_acc": 93.0},
        metric_weights={"bn_test_acc": 0.4, "baseline_test_acc": 0.3, "bn_epoch1_acc": 0.3},
        dataset_hint="MNIST (torchvision)",
        difficulty="medium",
        execution_timeout=40,
        split="train",
    ),

    Paper(
        paper_id="word2vec_tiny",
        title="Efficient Estimation of Word Representations in Vector Space (Mikolov et al., 2013)",
        abstract="We propose two novel model architectures for computing continuous vector "
                 "representations of words from very large data sets.",
        methodology=(
            "Implement Skip-gram word2vec with negative sampling from scratch using PyTorch. "
            "Use a fixed tiny corpus: the string 'the quick brown fox jumps over the lazy dog ' repeated 500 times. "
            "Vocabulary: all unique words (9 words). Embedding dim: 10. Window size: 2. Negative samples: 3. "
            "Use nn.Embedding for target and context embeddings. "
            "For each (center, context) pair, sample 3 negatives from vocab (excluding center). "
            "Loss: -log(sigma(v_context . v_center)) - sum(log(sigma(-v_neg . v_center))). "
            "Adam lr=0.01, train for 1000 steps with batch of all positive pairs. "
            "After training, report: fox_dog_sim = cosine similarity between embeddings of 'fox' and 'dog', "
            "the_over_sim = cosine similarity between 'the' and 'over' (should be higher, both function words)."
        ),
        target_metrics={"fox_dog_sim": 0.5, "the_over_sim": 0.6},
        metric_weights={"fox_dog_sim": 0.5, "the_over_sim": 0.5},
        dataset_hint="Fixed tiny corpus (no downloads)",
        difficulty="medium",
        execution_timeout=20,
        split="train",
    ),

    Paper(
        paper_id="pca_reconstruction",
        title="Analysis of a Complex of Statistical Variables into Principal Components (Hotelling, 1933)",
        abstract="A set of complex statistical variables may often be resolved into a smaller number of "
                 "fundamental variables called principal components which account for most of the variance.",
        methodology=(
            "Implement PCA from scratch using numpy (no sklearn) on MNIST. "
            "Load 1000 MNIST training images (flatten to 784-d vectors). Normalize to [0,1]. "
            "Center the data (subtract mean). Compute covariance matrix. "
            "Use numpy.linalg.eigh to get eigenvectors. Sort by descending eigenvalue. "
            "Project data to top-k=50 components, then reconstruct back to 784-d. "
            "Report: reconstruction_mse (mean squared error between original and reconstructed, x1000 for readability), "
            "variance_explained_50 (cumulative variance explained by top-50 components, as percentage 0-100)."
        ),
        target_metrics={"reconstruction_mse": 8.0, "variance_explained_50": 75.0},
        metric_weights={"reconstruction_mse": 0.5, "variance_explained_50": 0.5},
        dataset_hint="MNIST (torchvision, first 1000 samples)",
        difficulty="medium",
        execution_timeout=30,
        split="train",
    ),

    Paper(
        paper_id="attention_toy",
        title="Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2015)",
        abstract="We conjecture that the use of a fixed-length vector is a bottleneck in improving "
                 "the performance of encoder-decoder architectures.",
        methodology=(
            "Implement a simple additive attention mechanism on a toy sequence reversal task using PyTorch. "
            "Task: reverse sequences of length 8 from a vocab of 10 integers. "
            "Generate 2000 training pairs: input = random int sequence, target = its reverse. "
            "Encoder: nn.Embedding(10,16) + nn.GRU(16,32,batch_first=True). "
            "Attention: score_t = tanh(W1 * encoder_out + W2 * decoder_hidden), alpha = softmax(score_t), "
            "context = sum(alpha * encoder_out). "
            "Decoder: GRU taking [embedding(prev_token), context] as input, Linear(32,10) for output. "
            "Cross-entropy loss, Adam lr=1e-3, batch size 64, train 20 epochs. "
            "Report: sequence_accuracy (fraction of fully correct sequences in 200-sample val set), "
            "token_accuracy (fraction of correct individual tokens)."
        ),
        target_metrics={"sequence_accuracy": 0.70, "token_accuracy": 0.90},
        metric_weights={"sequence_accuracy": 0.6, "token_accuracy": 0.4},
        dataset_hint="Synthetic reversal task (no downloads)",
        difficulty="hard",
        execution_timeout=50,
        split="train",
    ),

    Paper(
        paper_id="residual_connections",
        title="Deep Residual Learning for Image Recognition (He et al., 2016)",
        abstract="We present a residual learning framework to ease the training of networks that are "
                 "substantially deeper than those used previously.",
        methodology=(
            "Implement a small ResNet on MNIST (not CIFAR -- use 1-channel 28x28 images). "
            "Build 2 residual blocks: each block is Conv(C,C,3,pad=1)->BN->ReLU->Conv(C,C,3,pad=1)->BN, "
            "plus a skip connection added before final ReLU. Use C=32 channels. "
            "Full network: Conv(1,32,3,pad=1)->BN->ReLU -> ResBlock -> ResBlock -> AdaptiveAvgPool(1) -> Linear(32,10). "
            "Train both this ResNet and an identical network WITHOUT skip connections. "
            "Adam lr=1e-3, batch size 128, 3 epochs on MNIST. "
            "Report: resnet_acc (0-100), plain_acc (0-100), residual_gain = resnet_acc - plain_acc."
        ),
        target_metrics={"resnet_acc": 99.0, "plain_acc": 98.5, "residual_gain": 0.5},
        metric_weights={"resnet_acc": 0.5, "plain_acc": 0.3, "residual_gain": 0.2},
        dataset_hint="MNIST (torchvision)",
        difficulty="hard",
        execution_timeout=50,
        split="train",
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # EVAL PAPERS  (5) — Never sampled during training
    # ─────────────────────────────────────────────────────────────────────────

    Paper(
        paper_id="momentum_sgd",
        title="On the Importance of Initialization and Momentum in Deep Learning (Sutskever et al., 2013)",
        abstract="We examine the role of momentum in deep learning and demonstrate its crucial "
                 "importance for training deep networks effectively.",
        methodology=(
            "Implement SGD with Nesterov momentum from scratch using PyTorch tensors but NOT torch.optim. "
            "Nesterov update: look_ahead = param - lr * momentum * velocity, "
            "gradient at look_ahead, velocity = momentum * velocity + lr * grad_at_lookahead, param -= velocity. "
            "Apply to a 2-layer MLP (Linear(784,256)->ReLU->Linear(256,10)) on MNIST. "
            "Parameters: lr=0.01, momentum=0.9. Train 3 epochs, batch size 128. "
            "Also train same model with vanilla SGD (momentum=0). "
            "Report: nesterov_test_acc (0-100), vanilla_test_acc (0-100), momentum_gain = nesterov - vanilla."
        ),
        target_metrics={"nesterov_test_acc": 97.0, "vanilla_test_acc": 95.5, "momentum_gain": 1.5},
        metric_weights={"nesterov_test_acc": 0.5, "vanilla_test_acc": 0.3, "momentum_gain": 0.2},
        dataset_hint="MNIST (torchvision)",
        difficulty="medium",
        execution_timeout=40,
        split="eval",
    ),

    Paper(
        paper_id="autoencoder_simple",
        title="Reducing the Dimensionality of Data with Neural Networks (Hinton & Salakhutdinov, 2006)",
        abstract="We describe an effective way of initializing the weights that allows deep autoencoder "
                 "networks to learn low-dimensional codes that work much better than PCA.",
        methodology=(
            "Implement a simple deterministic autoencoder on MNIST (NOT a VAE -- no sampling). "
            "Encoder: Linear(784,256)->ReLU->Linear(256,64)->ReLU->Linear(64,16). "
            "Decoder: Linear(16,64)->ReLU->Linear(64,256)->ReLU->Linear(256,784)->Sigmoid. "
            "Loss: MSE between input and reconstruction (treat pixels as 0-1 floats). "
            "Adam lr=1e-3, batch size 256, train 3 epochs. "
            "Report: test_recon_mse (MSE on test set x1000 for readability), "
            "bottleneck_dim=16 (constant -- confirms correct architecture)."
        ),
        target_metrics={"test_recon_mse": 12.0, "bottleneck_dim": 16.0},
        metric_weights={"test_recon_mse": 0.9, "bottleneck_dim": 0.1},
        dataset_hint="MNIST (torchvision)",
        difficulty="medium",
        execution_timeout=40,
        split="eval",
    ),

    Paper(
        paper_id="q_learning_tabular",
        title="Q-Learning (Watkins & Dayan, 1992)",
        abstract="Q-learning is a simple way for agents to learn how to act optimally in "
                 "controlled Markovian domains.",
        methodology=(
            "Implement tabular Q-learning on FrozenLake-v1 (4x4, not slippery). "
            "Q-table: numpy array shape (16, 4), initialized to zeros. "
            "Training: 5000 episodes. Epsilon-greedy: start=1.0, decay by 0.995 each episode, min=0.01. "
            "On each step: Q[s,a] += alpha * (reward + gamma * max(Q[s',:]) - Q[s,a]). "
            "alpha=0.8, gamma=0.95. "
            "Evaluation: run 100 test episodes with greedy policy (epsilon=0). "
            "Report: eval_success_rate (fraction of episodes reaching goal in 100 eval eps), "
            "final_epsilon."
        ),
        target_metrics={"eval_success_rate": 0.90, "final_epsilon": 0.01},
        metric_weights={"eval_success_rate": 0.9, "final_epsilon": 0.1},
        dataset_hint="FrozenLake-v1 (gymnasium, is_slippery=False)",
        difficulty="easy",
        execution_timeout=20,
        split="eval",
    ),

    Paper(
        paper_id="simple_rnn_char",
        title="Generating Sequences with Recurrent Neural Networks (Graves, 2013)",
        abstract="This paper shows how Long Short-term Memory recurrent neural networks can be used to "
                 "generate complex sequences with long-range structure.",
        methodology=(
            "Implement a character-level RNN for next-character prediction using PyTorch. "
            "Corpus: use the string 'hello world ' repeated 200 times (fixed, no download). "
            "Vocab: unique chars in corpus. "
            "Model: nn.Embedding(vocab_size, 16) -> nn.RNN(16, 64, batch_first=True) -> nn.Linear(64, vocab_size). "
            "Prepare sequences: input = chars[i:i+20], target = chars[i+1:i+21]. Stride=1. "
            "CrossEntropyLoss, Adam lr=1e-2, batch size 32, train 50 epochs. "
            "Report: final_train_loss, char_accuracy (fraction of correctly predicted next chars on train set)."
        ),
        target_metrics={"final_train_loss": 0.5, "char_accuracy": 0.85},
        metric_weights={"final_train_loss": 0.4, "char_accuracy": 0.6},
        dataset_hint="Fixed string corpus (no downloads)",
        difficulty="medium",
        execution_timeout=30,
        split="eval",
    ),

    Paper(
        paper_id="l2_regularization",
        title="A Simple Weight Decay Can Improve Generalization (Krogh & Hertz, 1992)",
        abstract="We study the effect of weight decay (L2 regularization) on generalization in neural "
                 "networks and show it can significantly reduce overfitting.",
        methodology=(
            "Demonstrate L2 regularization on a deliberately overfit scenario on MNIST. "
            "Create a small training set: only 200 MNIST samples (promotes overfitting). Use full 10k test set. "
            "Train Model A: Linear(784,512)->ReLU->Linear(512,10), Adam lr=1e-3, NO weight decay, 20 epochs. "
            "Train Model B: same architecture, same lr, weight_decay=0.01 in Adam (L2 regularization), 20 epochs. "
            "Report: l2_test_acc (0-100), nol2_test_acc (0-100), "
            "overfitting_gap_nol2 = nol2_train_acc - nol2_test_acc (measures overfitting), "
            "overfitting_gap_l2 = l2_train_acc - l2_test_acc."
        ),
        target_metrics={"l2_test_acc": 82.0, "nol2_test_acc": 75.0, "overfitting_gap_nol2": 20.0, "overfitting_gap_l2": 8.0},
        metric_weights={"l2_test_acc": 0.3, "nol2_test_acc": 0.2, "overfitting_gap_nol2": 0.25, "overfitting_gap_l2": 0.25},
        dataset_hint="MNIST (torchvision, 200-sample subset)",
        difficulty="medium",
        execution_timeout=35,
        split="eval",
    ),
]


TRAIN_PAPERS = [p for p in PAPER_CORPUS if p.split == "train"]
EVAL_PAPERS = [p for p in PAPER_CORPUS if p.split == "eval"]
WARMUP_PAPERS = [p for p in PAPER_CORPUS if p.split == "warmup"]


def load_paper(paper_id: str) -> Paper:
    for p in PAPER_CORPUS:
        if p.paper_id == paper_id:
            return p
    raise ValueError(f"Paper {paper_id!r} not found in corpus")


def sample_paper(episode_number: int = 0) -> Paper:
    """
    Curriculum sampling:
    - Episodes 0-49:    warmup papers only (extended: model must learn output format first)
    - Episodes 50-130:  easy + medium + hard train papers (weighted 5:3:1)
    - Episodes 131+:    all train papers uniformly
    Eval papers are NEVER sampled here.
    """
    if episode_number <= 49:
        return random.choice(WARMUP_PAPERS)

    train_by_difficulty = {
        "easy":   [p for p in TRAIN_PAPERS if p.difficulty == "easy"],
        "medium": [p for p in TRAIN_PAPERS if p.difficulty == "medium"],
        "hard":   [p for p in TRAIN_PAPERS if p.difficulty == "hard"],
    }

    if episode_number <= 130:
        pool = (
            train_by_difficulty["easy"] * 5 +
            train_by_difficulty["medium"] * 3 +
            train_by_difficulty["hard"] * 1
        )
    else:
        pool = TRAIN_PAPERS

    return random.choice(pool)


def format_paper_for_agent(paper: Paper) -> str:
    """
    Formats paper into the prompt the agent receives.
    target_metrics are NOT included — agent must run the code and measure results.
    """
    return f"""=== PAPER ===
Title: {paper.title}

Abstract:
{paper.abstract}

Methodology (implement this exactly):
{paper.methodology}

Dataset / Environment: {paper.dataset_hint}

=== YOUR TASK ===
Write a complete, executable Python script that implements the methodology above.

STRICT RULES:
1. The script must run in under {paper.execution_timeout} seconds
2. Allowed libraries: torch, torchvision, numpy, gymnasium, dataclasses, collections, json, math, random
3. On the very last line, print your measured metrics in this exact format:
   METRICS: {{"metric_name": value}}
   Example: METRICS: {{"test_accuracy": 97.3, "final_loss": 0.12}}
4. Do NOT import requests, urllib, subprocess, or write to disk outside /tmp
5. Do NOT hardcode expected values -- run the actual computation and measure the result
"""
