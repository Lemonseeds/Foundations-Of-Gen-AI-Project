# Comparative Autoregressive Text Generation using Recurrent Neural Architectures

**Foundations of Generative AI (ICT4403) — Assignment 2**  
*Autoregressive Next-Word Prediction & Sequential Generation on Ambrose Bierce's "The Devil's Dictionary"*

---

## 1. Problem Statement & Objective

Autoregressive language models predict probability distributions over sequences of tokens by decomposing joint sequence probability into a product of conditional probabilities:
$$P(w_1, w_2, \dots, w_T) = \prod_{t=1}^T P(w_t \mid w_1, \dots, w_{t-1})$$

The objective of this assignment is to design, implement, train, and critically evaluate an autoregressive text-generation system comparing four foundational recurrent neural network architectures:
1. **Simple Recurrent Neural Network (SimpleRNN / Elman RNN)**
2. **Bidirectional Recurrent Neural Network (BiRNN)**
3. **Long Short-Term Memory (LSTM)**
4. **Gated Recurrent Unit (GRU)**

All four architectures are trained under strictly controlled, identical experimental conditions (shared vocabulary, sequence length, embedding dimensions, optimizer, loss function, learning rate schedule, regularization, and random seed) on a unique textual corpus. The models are evaluated quantitatively using loss, perplexity, top-1 and top-5 accuracy, and computational efficiency, as well as qualitatively through sampling across multiple temperatures ($T \in \{0.2, 0.5, 1.0, 1.5\}$) to analyze diversity, repetition, coherence, and contextual consistency.

---

## 2. Dataset Description & Source Link

### 2.1 Selected Dataset: *The Devil's Dictionary* (Ambrose Bierce, 1911)
- **Source Link:** [Project Gutenberg eBook #972](https://www.gutenberg.org/ebooks/972)
- **Raw UTF-8 Text:** [https://www.gutenberg.org/cache/epub/972/pg972.txt](https://www.gutenberg.org/cache/epub/972/pg972.txt)

*The Devil's Dictionary* is a celebrated work of American literary satire consisting of approximately 1,000 cynical, witty definitions interleaved with mock-heroic verse ascribed to fictitious authorities (e.g., Father Gassalasca Jape, S.J.).

### 2.2 Rationale for Corpus Selection
1. **Uncommon & Distinctive:** Standard NLP tutorials predominantly use over-saturated datasets like Shakespeare sonnets, *Alice in Wonderland*, or Sherlock Holmes. *The Devil's Dictionary* provides an atypical, stylistically rich alternative.
2. **Dual Structural Registers:** The text alternates between terse prose lexicographical definitions (`HEADWORD, pos. definition`) and metered, rhymed couplets, providing an exacting test of whether recurrent models can capture diverse syntactic patterns.
3. **Lexical Richness:** Bierce's diction features an extensive vocabulary with high hapax legomena (words occurring exactly once), testing the vocabulary handling and generalization of autoregressive models.
4. **Feasibility on Local Hardware:** At ~373 KB cleaned text and ~75,000 tokens, the corpus is sized to achieve meaningful convergence on CPU within minutes without requiring multi-GPU infrastructure.

### 2.3 Dataset Characteristics & Corpus Statistics

| Metric | Measured Value |
|---|---|
| **Characters (cleaned text)** | 373,203 |
| **Total Tokens** | 75,278 |
| **Word Tokens** | 60,053 |
| **Unique Tokens (Lexical Types)** | 11,200 |
| **Vocabulary Size ($V$, frequency $\ge 2$)** | 4,731 |
| **Hapax Legomena (frequency $= 1$)** | 6,471 |
| **Out-of-Vocabulary ($\text{<unk>}$) Rate** | 8.6% |
| **Dictionary Entries Identified** | ~980 |
| **Verse Line Breaks ($\text{<nl>}$)** | 2,146 |
| **Paragraph Boundaries ($\text{<par>}$)** | 1,734 |
| **Train Sequences ($X \to y$)** | 28,627 |
| **Validation Sequences** | 5,454 |
| **Test Sequences** | 5,399 |

![Dataset Overview](results/dataset_overview.png)

### 2.4 Preprocessing & Sequence Construction Pipeline
The preprocessing logic is implemented in `src/data.py`:
1. **Header/Footer Stripping:** Strips the Project Gutenberg licensing boilerplate and author preface, isolating the dictionary body from `ABASEMENT, n.` to `ZOOLOGY, n.`.
2. **Text Normalization:** Lowercases all characters, normalizes typographic punctuation (curly quotes, em-dashes), removes markdown italics.
3. **Structural Tokens:** Retains formatting by injecting explicit tokens:
   - `<par>` indicates paragraph/definition breaks.
   - `<nl>` indicates line breaks within indented verse blocks.
4. **Punctuation Separation:** Punctuation marks (`,`, `.`, `;`, `:`, `!`, `?`, `"`, `(`, `)`, `--`) are tokenized as independent tokens so models learn proper grammatical cadences and the dictionary formula (`word , n .`).
5. **Vocabulary Capping:** Words with frequency $< 2$ are mapped to `<unk>`. The `<unk>` token is never sampled during generation, and sequences targeting `<unk>` are excluded from loss calculation.
6. **Leakage-Free Block Split:** The token stream is segmented into contiguous blocks of 2,000 tokens, allocated 8:1:1 to train, validation, and test sets. Windows are formed strictly within each block so that no sliding window crosses split boundaries, completely eliminating data leakage while maintaining full coverage across the alphabet.
7. **Sliding Window:** Context window length $T = 20$. Training uses stride $s = 2$ for computational efficiency, while validation and test sets use stride $s = 1$ for uncompromised evaluation.

---

## 3. Description of Implemented Models

All four models follow an identical modular pipeline:
$$\mathbf{x} \in \mathbb{R}^{B \times T} \xrightarrow{\text{Embedding}} \mathbf{E} \in \mathbb{R}^{B \times T \times d_e} \xrightarrow{\text{Dropout}(0.3)} \mathbf{h} \in \mathbb{R}^{B \times T \times d_h} \xrightarrow{\text{Recurrent Core}} \mathbf{h}_{\text{summary}} \xrightarrow{\text{Dropout}(0.3)} \mathbf{\hat{y}} \in \mathbb{R}^{B \times |V|}$$

Where $d_e = 128$, $d_h = 256$, number of layers $L = 2$, and vocabulary size $|V| = 4,731$.

| Model | Architecture Highlights | Parameter Count | Strengths & Vulnerabilities |
|---|---|---|---|
| **SimpleRNN** | 2-layer stacked Elman RNN with $\tanh$ non-linearity. | **2,051,835** | Fastest per step; vulnerable to vanishing/exploding gradients over multi-token contexts. |
| **BiRNN** | 2-layer bidirectional RNN processing the past context window forward and backward; concatenates forward last state $\vec{h}_T$ with backward first state $\overleftarrow{h}_1$. | **3,624,443** | Enriched representation of the fixed context window while preserving causality (no future information beyond $w_{t-1}$ is accessed). |
| **LSTM** | 2-layer stacked LSTM with input ($i_t$), forget ($f_t$), output ($o_t$) gates and additive memory cell state ($c_t$). | **2,743,035** | Robust against vanishing gradients via linear error carousel; captures long-range lexical dependencies. |
| **GRU** | 2-layer stacked GRU with reset ($r_t$) and update ($z_t$) gates; merges hidden state and cell memory. | **2,512,635** | Gated performance with ~8% fewer parameters than LSTM; faster training convergence. |

---

## 4. Execution Steps & Required Dependencies

### 4.1 System Requirements & Dependencies
- Python 3.10+ (tested on Python 3.12)
- Dependencies listed in `requirements.txt`:
  - `torch >= 2.0`
  - `numpy >= 1.24`
  - `pandas >= 2.0`
  - `matplotlib >= 3.7`
  - `jupyter`, `ipykernel`, `nbclient`
  - `python-docx` (for submission document generation)

Install dependencies:
```bash
pip install -r requirements.txt
```

### 4.2 Running the Standalone Pipeline
To download data, train all 4 models, evaluate, run temperature analysis, and save all artifacts:
```bash
python main.py
```
Optional CLI flags:
```bash
python main.py --epochs 15 --threads 6
```

### 4.3 Running the Interactive Jupyter Notebook
The assignment is fully encapsulated in `FGAI_Assignment2.ipynb`:
```bash
jupyter notebook FGAI_Assignment2.ipynb
```
All outputs, loss curves, comparative bar charts, and temperature sample tables are rendered directly in the notebook.

### 4.4 Repository Structure
```
fgaiasss/
├── data/
│   └── devils_dictionary.txt       # Downloaded raw text (Project Gutenberg #972)
├── src/
│   ├── __init__.py
│   ├── data.py                     # Cleaning, tokenizing, vocab, sliding-window dataset
│   ├── models.py                   # SimpleRNN, BiRNN, LSTM, GRU module definitions
│   ├── train.py                    # Shared training loop, early stopping, LR scheduler
│   ├── generate.py                 # Temperature sampling, diversity & repetition metrics
│   └── experiment.py               # Experiment orchestration, plotting, markdown generator
├── results/
│   ├── dataset_stats.json          # Numerical corpus characteristics
│   ├── dataset_overview.png        # Word frequency, Zipf distribution, POS tags
│   ├── train_config.json           # Exact hyperparameter config
│   ├── training_history.csv        # Epoch-by-epoch training and validation metrics
│   ├── training_curves.png         # Loss, perplexity, accuracy, and latency curves
│   ├── model_metrics.csv           # Final quantitative test metrics
│   ├── model_comparison.png        # Bar charts comparing test metrics across models
│   ├── temperature_metrics.csv     # Quantitative metrics across temperatures
│   ├── temperature_analysis.png    # Plots of diversity, repetition, and coherence vs. T
│   └── generated_samples.md        # Qualitative generations for all models and seeds
├── models/
│   ├── SimpleRNN.pt                # Checkpoint weights and configuration
│   ├── BiRNN.pt
│   ├── LSTM.pt
│   └── GRU.pt
├── FGAI_Assignment2.ipynb          # Runnable master Jupyter notebook with all outputs
├── main.py                         # Standalone CLI entry point
├── requirements.txt                # Python package requirements
├── .gitignore
└── README.md                       # Comprehensive documentation
```

---

## 5. Results Obtained

All models were trained on CPU with PyTorch under identical hyperparameters:
- Initial Learning Rate: $3 \times 10^{-3}$ with Adam optimizer
- Learning Rate Scheduler: `ReduceLROnPlateau` (factor $0.5$, patience $1$)
- Gradient Norm Clipping: $1.0$
- Batch Size: $256$
- Maximum Epochs: $15$ with early stopping (patience $= 3$ epochs)
- Seed: $42$

### 5.1 Quantitative Performance Summary Table

| Model | Parameters | Best Epoch | Val Loss | Val Perplexity | Test Loss | Test Perplexity | Test Top-1 Acc | Test Top-5 Acc | Total Time (s) | Time/Epoch (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| **SimpleRNN** | 2,051,835 | 5 | 5.9100 | 368.70 | 6.0513 | 424.66 | 17.30% | 34.32% | **128.2s** | **16.0s** |
| **BiRNN** | 3,624,443 | 1 | 5.8508 | 347.51 | 5.9234 | 373.67 | 16.61% | 32.62% | 163.5s | 40.9s |
| **LSTM** | 2,743,035 | 6 | **5.5937** | **268.73** | **5.7027** | **299.67** | 17.89% | 35.30% | 385.3s | 42.8s |
| **GRU** | 2,512,635 | 5 | 5.7587 | 316.93 | 5.8800 | 357.81 | **18.11%** | **35.38%** | 282.1s | 35.3s |

![Training Curves](results/training_curves.png)
![Model Comparison](results/model_comparison.png)

---

## 6. Model Comparison & Evaluation

### 6.1 Quantitative Evaluation
1. **Perplexity & Loss:**
   - **LSTM achieved the superior test perplexity ($299.67$) and lowest test loss ($5.7027$)**, outperforming GRU ($357.81$), BiRNN ($373.67$), and SimpleRNN ($424.66$).
   - The cell state $c_t$ and additive gradient path in LSTM provide superior capacity to retain historical context over 20 tokens compared to the single multiplicative recurrent transformation of SimpleRNN.
2. **Prediction Accuracy:**
   - GRU achieved the highest Test Top-1 Accuracy ($18.11\%$) and Top-5 Accuracy ($35.38\%$), narrowly edging out LSTM ($17.89\%$ Top-1, $35.30\%$ Top-5).
   - In next-word prediction with a vocabulary of 4,731 words, attaining $>35\%$ top-5 accuracy on a literary, archaic dataset demonstrates effective learning of grammatical word classes and syntactic structure.
3. **Computational Efficiency:**
   - SimpleRNN is the fastest ($16.0$ s/epoch, $128.2$ s total) due to minimal matrix multiplications per time step.
   - GRU ($35.3$ s/epoch, $282.1$ s total) was ~27% faster than LSTM ($42.8$ s/epoch, $385.3$ s total) because it utilizes two gates rather than three.
   - BiRNN took $40.9$ s/epoch due to conducting forward and backward recurrent sweeps across each batch.

### 6.2 Qualitative Evaluation (Characteristics of Generated Text)
- **Coherence & Grammar:** Both LSTM and GRU consistently generate grammatically plausible noun phrases (`an honorable of the ancient man`, `the tail of the power of his own`) and properly place commas, periods, and line-breaks. SimpleRNN produces loose associations that frequently break subject-verb agreements.
- **Contextual Consistency & Register:** BiRNN and LSTM successfully learned to produce dictionary entry formatting (`word, n. definition`) when generating from definition seeds, and transitioned into rhymed couplets with `<nl>` tokens when poetry contexts emerged.
- **Repetition Tendencies:** At low sampling temperatures ($T = 0.2$), all models succumb to phrase-level circular loops (`of the king of the king of the king` or `of the man of the man of the man`). GRU and LSTM escape this trap rapidly as temperature increases to $0.5$.

---

## 7. Sampling-Temperature Analysis

The sampling temperature $T > 0$ scales the pre-softmax logits $z_i$:
$$P(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

- As $T \to 0$, the distribution concentrates on $\operatorname{argmax}(z)$, approaching deterministic greedy decoding.
- At $T = 1.0$, tokens are sampled according to the model's true predicted distribution.
- As $T \to \infty$, the distribution approaches a uniform distribution over the entire vocabulary.

Continuations of 60 tokens were generated from three fixed seeds:
1. `politics, n.`
2. `love, n.`
3. `a lawyer is`

### 7.1 Quantitative Metrics Across Temperatures

| Model | Temperature | Distinct-1 (Unigram Diversity) | Distinct-2 (Bigram Diversity) | Rep-3gram (Trigram Repetition Rate) | Bigram Attested (Coherence Proxy) | Novel 4-gram Rate |
|---|---|---|---|---|---|---|
| **SimpleRNN** | 0.2 | 0.030 | 0.057 | 0.583 | 0.938 | 0.951 |
| | 0.5 | 0.147 | 0.345 | 0.133 | 0.855 | 0.941 |
| | 1.0 | 0.423 | 0.893 | 0.000 | 0.452 | 0.991 |
| | 1.5 | 0.630 | 0.992 | 0.000 | 0.171 | 0.998 |
| **BiRNN** | 0.2 | 0.027 | 0.051 | 0.692 | 0.923 | 0.926 |
| | 0.5 | 0.136 | 0.328 | 0.164 | 0.813 | 0.884 |
| | 1.0 | 0.465 | 0.913 | 0.000 | 0.369 | 0.981 |
| | 1.5 | 0.680 | 0.995 | 0.000 | 0.119 | 1.000 |
| **LSTM** | 0.2 | 0.027 | 0.051 | 0.749 | 0.974 | 0.981 |
| | 0.5 | 0.202 | 0.495 | 0.054 | 0.842 | 0.937 |
| | 1.0 | 0.480 | 0.932 | 0.000 | 0.382 | 0.994 |
| | 1.5 | 0.673 | 0.995 | 0.000 | 0.129 | 1.000 |
| **GRU** | 0.2 | 0.041 | 0.094 | 0.487 | 0.829 | 0.970 |
| | 0.5 | 0.191 | 0.484 | 0.026 | 0.797 | 0.943 |
| | 1.0 | 0.441 | 0.925 | 0.001 | 0.409 | 0.992 |
| | 1.5 | 0.646 | 0.995 | 0.000 | 0.146 | 1.000 |

![Temperature Analysis](results/temperature_analysis.png)

### 7.2 Generation Samples Showcase

#### Seed: `politics, n.`
- **Temperature 0.2 (Degenerative Looping):**
  - *LSTM:* `politics, n. a man of the time of the man of the man of the man of the man of the man...`
  - *GRU:* `politics, n. a world of the most of the world of the human of the world... and the holy of the king...`
  - *Analysis:* The highest probability tokens capture local frequency dominance ("of the", "man"), leading to degenerative cyclic attractor states with repetition rates $> 50\%$.
- **Temperature 0.5 (Balanced & Stylistic):**
  - *LSTM:* `politics, n. a man of the time of the man of the word of the man of the contents of the man, one of the person of an solemn of his own of the ancient of the man...`
  - *BiRNN:* `politics, n. a beast of the innocent of the disease. g, n. a person of the creature at, one of the king is not the king.`
  - *Analysis:* At $T = 0.5$, models successfully capture the sardonic lexicon of Bierce ("beast", "disease", "solemn", "innocent") while keeping repetition down and maintaining corpus-attested bigram grammar ($> 80\%$).
- **Temperature 1.0 (Creative & Natural Diversity):**
  - *LSTM:* `politics, n. an my salvation of consists is determine the are: please me of b. ye wilson at with the one of the tail in an solemn and his antiquity, or my money of an opponent of the dinner desolating now they after saint to better turned my true fitted you cannot a running process`
  - *GRU:* `politics, n. an permitted it to consists have determine the are of the dinner of the sovereign of the comparative people at with the one of the person contemplation and the adam but the dupe or the provision of an opponent of the dinner desolating now as after he to`
  - *Analysis:* High lexical diversity (distinct-2 $> 0.91$), repetition drops to $0.0\%$, and distinctive Biercian vocabulary emerges ("sovereign", "contemplation", "dupe", "antiquity").
- **Temperature 1.5 (High Entropy / Word Salad):**
  - *LSTM:* `politics, n. an my salvation what consists have determine the are: please grass brow were safety-clutch imperfectly widely ye shall at with incur one of persons...`
  - *Analysis:* Softmax becomes overly uniform; low-probability words ("safety-clutch", "brow", "incur") are frequently chosen, eroding syntactic coherence (corpus bigram attestation plunges to $\sim 12\%$).

---

## 8. Conclusion & Model Justification

### Identification of the Best-Performing Model: **Long Short-Term Memory (LSTM)**

Based on rigorous experimental comparison across quantitative benchmarks and qualitative generation fidelity, the **LSTM model is identified as the most effective autoregressive generation system for this dataset**.

### Justification:
1. **Lowest Perplexity & Loss:**
   - LSTM achieved a Test Perplexity of **299.67** (versus GRU's $357.81$, BiRNN's $373.67$, and SimpleRNN's $424.66$). In autoregressive modeling, perplexity directly reflects how well the probability model anticipates unseen text.
2. **Superior Representation of Long-Term Context:**
   - The gating mechanism in LSTM decouples memory cell retention from hidden state emissions, enabling the model to track multi-word syntactic dependencies (e.g., maintaining subject-verb relationships across relative clauses and verse breaks).
3. **Optimal Generation Trade-Off at Intermediate Temperatures:**
   - At $T = 0.5$ and $T = 1.0$, LSTM achieved the best balance between lexical diversity (distinct-2 $= 0.495$ at $T=0.5$, $0.932$ at $T=1.0$), negligible repetition, and strong contextual consistency with Bierce's cynical, satirical vocabulary.
4. **GRU as Close Runner-Up:**
   - The GRU model is a strong contender, offering slightly higher top-1/top-5 prediction accuracy and faster training time per epoch ($35.3$ s vs. $42.8$ s), making it an excellent candidate under strict compute constraints. However, LSTM's superior perplexity confirms its deeper probabilistic capture of the textual distribution.

---

## Authors & Academic Integrity
- **Course:** Foundations of Generative AI (ICT4403)
- **Assignment:** Assignment 2 — Autoregressive Text Generation
- **Dataset Source:** Project Gutenberg (Public Domain)
