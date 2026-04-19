# **From Memorization to Reasoning in the Spectrum of Loss Curvature**

**Jack Merullo, Srihita Vatsavaya, Lucius Bushnaq, Owen Lewis** *Goodfire* {jack,siri,lucius,owen}@goodfire.ai  
**arXiv:2510.24256v2 \[cs.CL\] 31 Oct 2025**

## **Abstract**

We characterize how memorization is represented in transformer models and show that it can be disentangled in the weights of both language models (LMs) and vision transformers (ViTs) using a decomposition based on the loss landscape curvature. This insight is based on prior theoretical and empirical work showing that the curvature for memorized training points is much sharper than non-memorized, meaning ordering weight components from high to low curvature can reveal a distinction without explicit labels. This motivates a weight editing procedure that suppresses far more recitation of untargeted memorized data more effectively than a recent unlearning method (BalancedSubnet), while maintaining lower perplexity. Since the basis of curvature has a natural interpretation for shared structure in model weights, we analyze the editing procedure extensively on its effect on downstream tasks in LMs, and find that fact retrieval and arithmetic are specifically and consistently negatively affected, even though open book fact retrieval and general logical reasoning is conserved. We posit these tasks rely heavily on specialized directions in weight space rather than general purpose mechanisms, regardless of whether those individual datapoints are memorized. We support this by showing a correspondence between task data's activation strength with low curvature components that we edit out, and the drop in task performance after the edit. Our work enhances the understanding of memorization in neural networks with practical applications towards removing it, and provides evidence for idiosyncratic, narrowly-used structures involved in solving tasks like math and fact retrieval.

## **1\. Introduction**

To what degree do models generate genuinely new knowledge, as opposed to simply reassembling snippets of data memorized from their training sets? Much discussion about the current utility and future prospects of large neural networks has centered on this question. On the one hand, a growing trophy case of model accomplishments on novel tasks near the frontier of human capabilities argues strongly against memorization strictly construed as an explanation of the full range of model behavior. But on the other hand, recent papers have argued convincingly that models do in fact memorize large volumes of their training data verbatim, and a surprisingly large fraction of naturalistic interactions with language-model chatbots contain significant verbatim recitations (Aerni et al., 2024; Carlini et al., 2022; Stoehr et al., 2024), behaviors which have significant implications for copyright and data privacy (Karamolegkou et al., 2023; Carlini et al., 2019; Shokri et al., 2017).  
Models thus seem to have a significant and frequently used capacity for both memorization and generalization. The question is not whether models generalize or recite, but rather how these capabilities are represented, how they interact and trade off (Nguyen & Reddy, 2025), and how they might be modulated. These are the questions we seek to address in this work. We build on existing work that characterizes memorization in terms of the curvature of the loss landscape as a function of a model's weights (Foret et al., 2021; Hochreiter & Schmidhuber, 1997; LeCun et al., 1989; Hassibi et al., 1993; Keskar et al., 2017; Garg et al., 2024; Ravikumar et al., 2024; Jeon et al., 2024; Kim et al., 2023). This prior work argues theoretically and empirically that the loss landscape has highly curved directions in the neighborhood of memorized points, while generalization corresponds to flatter basins. We exploit this insight while extending it in several ways.  
We study models' behavior in aggregate, rather than for individual examples (Figure 1, left). Are there model structures that account for memorization and generalization across large swaths of training data? We find that there are: in both language and vision models, the eigenbasis of the approximated Hessian of weight matrices uncovers distinct disentanglement of memorization and generalization, in a way that extends across a range of subdistributions of memorized data. In extending from studying per-example to bulk memorization, we propose a novel inversion of the previous interpretation of loss curvature: while individual memorized points are associated with high curvature, the direction of curvature varies across examples, meaning that, averaged across multiple examples, memorization directions are actually flatter than generalizing directions, which maintain a consistent moderate curvature across points.  
We propose an effective recitation-reduction technique based on ablating memorized directions in weight space. We compare our results to a recent supervised memorization removal technique (BalancedSubnet (BSN); Sakarvadia et al. (2025)) and find that our method matches the suppression of the targeted forget set of BSN, and is similarly robust to stress tests (Huang et al., 2024), while removing far more unseen memorized data and achieving lower perplexity (Section 5.2).  
We go beyond pure memorization and generalization and find curvature signatures of intermediate behaviors like fact retrieval and arithmetic (Figure 1, right). Many classical analyses of memorization and generalization have studied classification models, where the distinction between the two is clear. In modern language models, though, there is a much richer spectrum of behaviors between the poles of pure memorization (verbatim recitation of long passages) and pure generalization (de novo reasoning). For example, facts like "Paris is the capital of France" are memorized in the sense that they are specific pieces of information that the model knows, but are general in the sense that they are not tied to specific syntactic instantiations seen in the training data. Similarly, arithmetic and logical reasoning test a model's ability to generate novel inferences, but the inference rules and base axioms may be remembered from training. Going beyond prior work, we extend the loss curvature analysis to cover these reasoning types, situating them on a continuum between memorization and generalization. We find that, besides memorization, arithmetic and factual recall exhibit weight activation with low-curvature weight directions and are sensitive to their removal. On the other hand, logical reasoning, which does not necessarily require precise recall or calculation, is robust to removing flat directions, which in some cases even improving performance. Our general approach is illustrated in Figure 1\.  
For all of our analyses, we measure curvature with the Kronecker-Factored Approximate Curvature (K-FAC) approximation to the model's Hessian, a computational technique that makes curvature analyses tractable at scale. K-FAC has been widely used for Hessian approximation in other settings, particularly for natural gradient optimization, but to our knowledge, our use of it to study memorization and generalization via curvature is novel.

## **2\. Related Work**

Memorization, especially as a special case of overfitting, is a widely studied topic in both modern and classical machine learning. In the modern era of extremely large overparameterized models, there is particular interest in quantifying the ability and tendency of models to use their huge capacity to memorize training data. Recent work has shown that models are indeed able to store large amounts of data exactly (Morris et al., 2025), and that this data can be elicited verbatim in both naturalistic (Aerni et al., 2024\) and adversarial (Carlini et al., 2022; Karamolegkou et al., 2023\) regimes.  
A closely related question is whether memories can be localized in model weights (Maini et al., 2023; Chang et al., 2024; Stoehr et al., 2024; Huang et al., 2024). Aligning with previous work (Hase et al., 2023; Karamolegkou et al., 2023), our work suggests that memorization is hard to pinpoint (and likely highly distributed), but we do find that distinctly loss-curved directions related to recitation of memorized data can be localized to some (early/late) layers. Similar localization work has studied the storage and retrieval of facts (Geva et al., 2021; Gur-Arieh et al., 2025; Meng et al., 2022; Dai et al., 2022; Rajamanoharan et al., 2023; Merullo et al., 2024; Menta et al., 2025), connecting to our analysis of factual recall in Section 6\. Other work has focused on localizing functional components in weight space through other types of decompositions (Baker et al., 2025; Bushnaq et al., 2025).  
Like the present study, previous work has used techniques like SVD to prune directions in weight space to compress models, expose low-rank structure, and understand memorization (Zhao et al., 2024; Jaiswal et al., 2025; Sharma et al., 2023). Relatedly, spectral dynamics has also been used explore memorization and generalization (Yunis et al., 2024).  
Finally, a range of theoretical and empirical work has studied the connection between memorization and loss curvature, connecting high-curvature directions with memorized examples (Foret et al., 2021; Hochreiter & Schmidhuber, 1997; LeCun et al., 1989; Hassibi et al., 1993; Keskar et al., 2017; Garg et al., 2024; Ravikumar et al., 2024; Jeon et al., 2024; Kim et al., 2023). Bushnaq et al. (2024) investigate using the loss landscape for interpretability.

## **3\. Methods**

### **3.1. Finding Memorization Weights with K-FAC**

In this work, we aim to decompose weight matrices in such a way that disentangles weight directions involved in verbatim memorization vs. generalization behavior. To do so, we decompose the MLP weight matrices in a model using the activations and gradients around them (Figure 1, top).  
For a weight matrix $W$, we collect a sample of activations going into $W$ and backpropagate (using the loss over the model's distribution) to collect gradients on the output side of $W$. We then form the covariance matrices $A$ for activations and $G$ for the gradients. Given an eigenvector $u$ from $G$ and $v$ from $A$, the outer product $u v^\\top$ forms a rank-one matrix in the space of $W$. We show that there is a strong ordered relationship between the eigenspectrum of these weight components, and memorized data; the relationship being that the components corresponding to the smallest eigenvalues are more likely to be used for reciting verbatim memorized training data.  
The precise reason for why this construction makes sense to study memorization is because it corresponds to K-FAC (Kronecker-Factored Approximate Curvature; Martens & Grosse (2015)), which estimates the Fisher Information Matrix (FIM) as $F \\approx G \\otimes A$. Therefore, we refer to the weight components we use in this work $u v^\\top$ as K-FAC eigenvectors. Extensive prior work has connected loss curvature to memorization (§2), and K-FAC gives us a way to obtain essentially a dataset-average of curvature with respect to model weights. We detail this relationship in more detail in Section 7\.

### **3.2. Collecting K-FAC Statistics**

We use K-FAC to approximate the Fisher block of each linear projection as a Kronecker product as shown in Equation 1, where $A$ captures input correlations and $G$ captures correlations of output gradients. We collect these factors for the MLP projections (gate\_proj, up\_proj, down\_proj) by streaming \~20M tokens from Dolmino/OLMo mixtures with sequence length 512 under next-token cross-entropy. In the forward pass we buffer pre-activation inputs $x$ (excluding the last position), and in the backward pass we record the corresponding gradients $g$. We accumulate $x x^\\top$ and $g g^\\top$ and normalize by the total number of contributing positions to form $A$ and $G$. For ViT experiments, we collect 10k images from the training split of ImageNet, using only the CLS token to collect activations and gradients.

### **3.3. Models**

In this section, we describe the models we use for analysis, and settings we use when evaluating memorized data.  
**Vision Transformers (ViTs):** Memorization in image classification models has been well studied, and there are simple recipes for producing models that memorize specific images. We train a family of 86M parameter ViT-Base models (Dosovitskiy et al., 2020\) with 16x16 image patches at image resolution 224x224. We follow Dosovitskiy et al. (2020) training recipe on the ILSVRC 2012 ImageNet dataset (Russakovsky et al., 2015). In order to control memorization, we train ViT variants where a subset of training images have randomly assigned 'noised' labels. The only way for a model to reduce the loss on these images is to memorize these input-label pairs exactly. This is a standard setup for evaluating memorization in image classifiers (Zhang et al., 2017). Our default for evaluation is to train with 10% noised labels for 300 epochs. Our model trained with the noised labels achieves a top-1 accuracy on the validation set of 68.7%. When training with no noise, our model achieves 77.2% top-1 accuracy.  
**Language Models (LMs):** We use the OLMo-2 family of models (OLMo et al., 2024), because they have openly accessible pretraining data and high performance on language modeling tasks. We report results for the 7B model. Previous work on evaluating memorization in LMs (Carlini et al., 2019; Huang et al., 2024; Shokri et al., 2017; Carlini et al., 2022\) generally sampled sequences from a model's pretraining data, split each sequence into a prefix $P$ and suffix $S$, and evaluated whether the model produced $S$ under greedy decoding with prompt $P$. We adopt this same methodology, and use prefixes with length $|P|=64$ tokens and suffixes with length $|S|^{\*}=48$ tokens.

## **4\. Disentangling Weights Involved in Memorization**

This section will show that K-FAC is indeed a particularly good candidate to disentangle weights involved in memorized recitation. In simple terms, our procedure is to measure the interaction between memorized and non-memorized (clean) data with different K-FAC components of the weights. Our hypothesis is that if the curvature is a good measure of memorization, then the eigenvectors in different parts of the spectrum of the curvature basis (i.e., the top 10% of eigenvectors, bottom 50%, etc.) will activate differently from each other on memorized or clean data. The way we measure this is through activation ratios: for some hidden activation $x\_{mem}$ stemming from a memorized input, we compare the ratio of its activation with a weight component $C$ to the activation with a clean input $x\_{clean}$. If one weight component has a high $\\|Cx\_{mem}\\| / \\|Cx\_{clean}\\|$ ratio and another component has a very low ratio, then we know this weight matrix distinguishes the two types of data. To summarize the experiment: for a given weight matrix, we would like to know if it has some components that interact more with memorized than non-memorized data, with the hypothesis that the loss curvature basis is a principled way to disentangle these two signals.  
**SVD:** A reasonable idea is that we can disentangle memorization in the basis of singular vectors of a weight matrix, which decomposes into the components of most to least 'importance' for reconstructing the matrix. The top singular vectors may correspond to more general directions in weight space, and the lower ones towards directions used for reciting memorization (see Yunis et al. (2024)). If we compute the product between activations and weights as $Wx$, and the SVD is $U \\Sigma V^\\top x$, then the right singular vectors in $V^\\top$ give us the magnitude with which those singular vectors read from.  
**K-FAC Eigenvectors:** Similar to the above setting with the right singular vectors, we can disentangle memorization in the basis of the activation eigenvectors. Since we do not measure gradient information in this setting, we project incoming activations onto the eigenbasis of $A$. This can also be thought of as projecting onto the uncentered PCA components of activations, however, for the purposes of drawing percentile bands (top 10% vs. bottom 50%, e.g.) we do order the activation eigenvectors by the true FIM order (by the products of all combos of eigenvalues between activations and gradients).  
**Results:** Figure 2 shows our results. In both LMs and ViTs, K-FAC shows a more salient divergence between different parts of the eigenspectrum on memorized and clean data. For example, at the layer 22 MLP input in OLMo-7B, the bottom 50% of eigenvectors has a 23.1% higher activation on memorized data than clean data on average, and the top 10% of eigenvectors has a 26% higher activation on clean data than memorized data. Importantly, the relative strength is sorted according to eigenvector band. That is, in terms of activation with clean data, the strength of the top 10% \> 10-25% \> 25-50% \> bottom 50%. SVD has a strong separation on the last layer (top 10% having 26% higher activation on memorized data), as well as around layer 20 in the gate projection only, but is lacking this sorted property. We therefore suggest that the curvature basis is more interpretable and accurate to describe the spectrum between memorized and non-memorized. In the ViT model we train, the top 10% eigenvectors have over a 2x activation strength on clean over memorized data at the last layer; the SVD for this model shows no such pattern. We train several other variants of the ViT models with varying weight decay, and find that it has a substantial effect on this separation with higher weight decay typically causing more drastic specialization. These results can be found in Appendix H.1.  
These results support our hypothesis that the curvature basis allows us to disentangle weights involved in memorized recitation. As we will show in Section 6, we can use the amount of divergence between the top and bottom parts of the eigenspectrum to predict model behaviors on various tasks. In the following section, we will show that removing weight components at the bottom of the K-FAC spectrum suppresses memorized data while retaining strong performance.

## **5\. Editing Model Weights to Suppress Memorization**

A natural followup to finding distinct patterns of activations for (non-)memorized data across weight components in the curvature basis is whether we can use this discovery to prevent the recitation of memorized data while retaining general capabilities. We propose a novel editing method which projects an MLP weight matrix $W$ into a subspace defined by its Hessian. As discussed in Section 7, the eigendecomposition of this Hessian sorts components of $W$ into directions of highest to lowest curvature in the loss landscape across a dataset. As we have discussed, in the aggregate across a dataset, the top eigenvectors correspond to generalizing directions; a claim that we will directly test here. Therefore, we propose keeping only the top $k\\%$ of eigenvectors as a way to keep this shared structure while removing noisy or generally unimportant weight directions. In words, we define a matrix projection of an MLP weight matrix $W$ that prevents communication through directions of low curvature in the eigenvectors of K-FAC.  
Our method decomposes weight matrices using eigenbases derived from activation and gradient covariance matrices. Rather than truncating the eigenbases directly, we select specific pairs of eigenvectors whose joint contribution to curvature is highest, preserving a targeted fraction of the total curvature mass. Concretely, we start with K-FAC factor matrices $G \\in \\mathbb{R}^{p \\times p}$ (gradient covariance) and $A \\in \\mathbb{R}^{q \\times q}$ (activation covariance). These are decomposed into their eigenspaces as:  
$$G \= U\_G \\operatorname{diag}(\\lambda) U\_G^\\top, \\quad A \= U\_A \\operatorname{diag}(\\mu) U\_A^\\top$$  
with $\\lambda\_0 \\ge \\lambda\_1 \\ge \\dots \\ge 0$ and $\\mu\_0 \\ge \\mu\_1 \\ge \\dots \\ge 0$.  
Given a weight matrix $W \\in \\mathbb{R}^{p \\times q}$, we first express it in terms of these eigenbases as:  
$$C \= U\_G^\\top W U\_A, \\quad \\text{where each coefficient } C\_{ij} \= u\_i^\\top W v\_j$$  
To guide our compression, for each eigenvector pair $(i, j)$ we define a measure of curvature mass, $\\Pi\_{ij} := \\lambda\_i \\mu\_j$. The total curvature mass, summing over all pairs, is then given by:  
$$M\_{tot} := \\sum\_{i,j} \\Pi\_{ij} \= (\\sum\_i \\lambda\_i)(\\sum\_j \\mu\_j)$$  
Our compression strategy then selects a subset $S$ of eigenvector pairs, prioritizing those with the highest curvature mass. Formally, given a threshold parameter $\\rho \\in (0,1\]$, we construct $S$ by including pairs in descending order of $\\Pi\_{ij}$ until the cumulative curvature mass of selected pairs meets or exceeds the fraction $\\rho$ of the total mass: $\\sum\_{(i,j) \\in S} \\Pi\_{ij} \\ge \\rho M\_{tot}$.  
Once the subset $S$ is determined, we define a binary mask matrix $M \\in \\{0,1\\}^{p \\times q}$ where $M\_{ij}=1$ if $(i,j) \\in S$ and 0 otherwise. Finally, we construct the compressed weight matrix by zeroing out coefficients corresponding to pairs not in $S$:  
$$W\_{pairs} \= U\_G (C \\odot M) U\_A^\\top \= \\sum\_{(i,j) \\in S} C\_{ij} u\_i v\_j^\\top$$  
This method selectively preserves those directions in weight space most significant to the model's curvature.  
We also test decomposing and truncating the bottom $k\\%$ of singular values of a matrix $W$, as well. It may be the case that the singular vector spectrum aligns incidentally with directions of high/low curvature, providing a data-free method for separating memorization. This setting also expands experiments on truncation explored in Yunis et al. (2024). Why might this alignment occur? Recall that we are approximating curvature using the eigenvectors of the covariance activations matrix $A$ going into the layer. These eigenvectors are simply the uncentered principal component directions. When we say that the right singular vectors of $W$ align with the top eigenvectors of $A$, we're equivalently saying $W$ places most of its sensitivity along the top input principal components. While we don't directly test this alignment, our results empirically support this interpretation.

### **5.1. Experimental Setup**

We construct two exact-match memorization sets under greedy decoding, one drawn from the pretraining corpus with a prefix and suffix of 48 and 64 respectively, and another of memorized historical quotes which we measure as memorized with a suffix of 8\. Details are in Appendix B.  
We report the metrics described in Section 5.1 separately on the Dolma and Quotes datasets. We primarily rely on strict accuracy to detect exact memorization in the dataset generation. However for evaluation, cases where strict \= 0 but loose \= 1 highlight sequences differing only slightly—semantically or syntactically—from the memorized target, representing partial memorization we also seek to avoid. Thus, loose accuracy complements strict accuracy by capturing near-verbatim memorization. Additionally, Levenshtein distance provides a continuous, threshold-free metric, allowing us to quantify memorization degradation more precisely.  
**Metrics:** We evaluate memorization suppression in ViTs with three metrics: memory reduction (drop in top-1 predictions of memorized/noised labels), ground-truth recovery (accuracy of recovering the true label for images trained with noised labels), and validation accuracy (post-edit validation accuracy to assess impact on core capabilities).  
For LMs, given a prefix-suffix pair $(P, S)$ with $|S|=L$, we prompt with $P$ and greedily generate $L$ tokens to obtain $\\hat{S}$. We compute the token-level Levenshtein distance $d(S, \\hat{S})$ and report: **Strict Accuracy** $\\mathbb{I}\[d(S, \\hat{S})=0\]$; **Loose Accuracy** $\\mathbb{I}\[1 \- d(S, \\hat{S})/L \\ge \\tau\]$ with $\\tau=0.75$; and **Average Normalized Distance** $\\frac{1}{N} \\sum\_{n=1}^N \\frac{d(S\_n, \\hat{S}\_n)}{L\_n}$, where higher values indicate less memorization.  
**Baseline: Balanced Subnet (BSN):** We compare to BSN, a recent memorization unlearning method introduced in Sakarvadia et al. (2025). This method trains a binary mask over individual MLP parameters optimized to maximize loss on a forget set (memorized data), while retaining low loss on a retain set (non-memorized, clean data).  
**Model Settings:**

* **K-FAC:** The full hyperparameter search details can be found in Appendix D. In LMs we edit layers 23, 24, and 25 at 60% energy retained in the up and gate projections in MLPs. In ViTs, we edit layers 0 and 11 to 75% energy on both up and down MLP projections.  
* **BSN:** The best BSN settings for editing the language model were a loss weight of 0.7, 5 epochs, a sparsity ratio of 0.0015, and a learning rate of 0.3.  
* **SVD:** The best SVD settings for editing the language model were pruning ratios of 0.005 (0.5%) for the up and down projections and 0.5 (50%) for the gate projection in layer 21\.

### **5.2. Results**

**K-FAC suppresses the broadest range of memorized text in LMs:** We compare our proposed K-FAC method against the state-of-the-art BSN baseline and SVD in Table 1\. To ensure comparability of model coherence, we matched perplexities closely (K-FAC: 22.84, BSN: 23.59, SVD: 22.49), noting that BSN achieved slightly better nDCG@10 (0.97 vs. 0.91 for both K-FAC and SVD). While BSN required explicit training data, K-FAC and SVD did not—highlighting an important advantage of these approaches. On the Dolma validation set, K-FAC achieved 3.4% strict accuracy, BSN achieved 6.0%, and SVD achieved 3.0%. More notably, on the truly out-of-distribution historical quotes dataset, K-FAC achieved 16.1% strict accuracy, followed by SVD at 17.5%, whereas BSN achieved 60.0%. For completeness, Table 1 includes corresponding experiments on the 1B model, with settings detailed in Appendix F. In addition to perplexity, we include 20 generations from each method in Appendix J. Since it involves gradient ascent, BSN generates mostly nonsense when it detects memorization (and in some cases, for clean text). K-FAC and SVD edits retain very diverse generations; it is known, however, that low rank truncation, like that performed with the SVD edit can lead to unusual text, like dropped function words or incoherent text that don't show up in benchmark numbers (Sharma et al., 2023; Jaiswal et al., 2024). We only see 2 examples of this, but it may be necessary to train further after the edit to regain full expressivity. We don't see this issue with the K-FAC edits, which retain full rank. These results demonstrate our curvature-based pruning approach effectively mitigates memorization the best across both model sizes without requiring supervised training data, achieving notably better generalization to unseen memorized content.  
**Table 1: Comparison of unlearning methods on OLMo-2 7B and 1B models.** *Lower Strict/Loose percentages indicate better memorization suppression.*

| Model | Method | Dolma Val Strict (%) | Dolma Val Loose (%) | Dolma Avg Lev ↑ | Hist. Quotes Strict (%) | Hist. Quotes Loose (%) | Hist. Quotes Avg Lev ↑ | Pile 10k Perplexity |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **7B** | Baseline | 99.9 | 100.0 | 0.002 | 99.9 | 100.0 | 0.001 | 19.04 |
|  | BSN | 6.0 | 11.0 | 0.860 | 60.0 | 79.0 | 0.180 | 23.59 |
|  | K-FAC | 3.4 | 8.8 | 0.704 | 16.1 | 23.8 | 0.625 | 22.84 |
|  | SVD | 3.0 | 6.8 | 0.754 | 17.5 | 30.4 | 0.560 | 22.49 |
| **1B** | Baseline | 98.46 | 99.38 | 0.005 | 98.5 | 98.95 | 0.006 | 23.19 |
|  | BSN | 3.0 | 5.0 | 0.900 | 57.0 | 66.0 | 0.250 | 25.41 |
|  | K-FAC | 2.8 | 7.2 | 0.761 | 27.7 | 39.9 | 0.470 | 26.53 |
|  | SVD | 3.2 | 6.3 | 0.781 | 39.6 | 48.5 | 0.401 | 26.94 |

**ViTs Edited with K-FAC recover more ground truth labels than SVD:** Table 2 shows the results for editing ViT-Base with 10% training noise in various settings. On a per-layer basis, we see that pruning the earliest and latest layers provides the best results across the board. For both methods, we achieve the best performance when we prune MLPs 0 and 11 simultaneously, driving memorization performance down to 3.5% from over 80%. K-FAC also increases the validation accuracy over 4% from 67% to 71.7%, while SVD only increases performance around 1%. If we have successfully targeted memorized features, then we should see that the images that were memorized should switch to predicting their ground truth (GT) labels. K-FAC successfully raises the ground truth accuracy up to 66.5% while SVD reaches 58.9%.  
**Table 2: Comparison of edits on ViT-Base.**

| Method | Memorized Train (%) ↓ | Train GT Accuracy (%) ↑ | Validation Accuracy (%) ↑ |
| :---- | :---- | :---- | :---- |
| Baseline | 81.6 | 10.5 | 67.0 |
| SVD | 3.5 | 58.9 | 67.8 |
| K-FAC | 3.5 | 66.5 | 71.7 |

**Stress tests:** Drawing from the positional perturbation stress tests outlined by Huang et al. (2024), we conducted a similar evaluation comparing K-FAC against BSN. For space we include these in Appendix G, but we find far less sensitivity to positional perturbations in both K-FAC and BSN than the older methods analyzed in prior work.

## **6\. Spectrum of Memorization to Reasoning in Downstream Behaviors in LMs**

In traditional classification models, the distinction between memorization and generalization is stark and exhaustive: a label is either randomly generated (memorized) or inferred based on training (generalized). LMs have a varied landscape between memorization and reasoning. Here, we connect a wide range of LM behaviors to our story about weight space curvature and demonstrate tasks that have varying sensitivity to perturbations in weights. We demonstrate a spectrum of behaviors between pure memorization and pure reasoning that maps to our measurement of sharpness, and notably, that mathematical reasoning is highly brittle (Nikankin et al., 2025), while difficult non-numerical logical reasoning is among the most robust behaviors.  
**Setup:** We use the OLMES evaluation suite (Gu et al., 2025\) to measure performance on edited models across benchmarks. We target benchmarks four main categories of tasks: Closed-book fact retrieval, Open-book fact retrieval, Logical Reasoning, and Math (arithmetic heavy). Open book retrieval involves question answering where there is a source available in context, whereas closed-book requires retrieval of facts directly from parametric knowledge. For example, TriviaQA is typically closed-book, but can be made open-book by including the relevant Wikipedia page (we refer to this as TriviaQA-Open). We include three non-standard datasets: Boar Etruscan (McCoy et al., 2023), which is an in-context constructed fake language like pig-latin. In order to perform well, models must rely solely on reasoning about rules provided in context, Relations (Hernandez et al., 2024), which is a dataset of factual relations such as "capital-of-country" (we use the 26 factual relations), and SimpleMath which is a generated dataset of two digit addition/subtraction problems (fed to the model 5-shot, with no other context).  
**Results:** In Figure 3, we report a subset of benchmarks covering logic, fact recall, math, and our datasets of memorized sequences as a proportion of the unedited models accuracy. We find a mostly smooth drop off from logical reasoning (95-106% retention of baseline), open-book QA (93-99% retention), closed-book QA (74-86% retention), math (66-74%), and memorization (3-16%). Note that outside of the domains of math, and closed book QA, we find that K-FAC edited models perform very well compared to baseline (and often better than BSN), such as on CommonsenseQA, which doesn't cleanly fall into any of these categories. See Appendix I for details. The ranges of degradation we see reflect behavioral brittleness to weight perturbations specific to the domain of the task.  
We can also show that this brittleness is measurable in terms of the magnitude of the activation along a K-FAC eigenvector direction (see §7). Figure 4 shows that interactions with the top and bottom of the curvature eigenbasis are predictive of how brittle they are (where a task falls relative to others in Figure 3). For example, hidden activations from OpenbookQA interact far less with the bottom of the eigenspectrum across layers 23-25 than the memorized data (memorized data interact at \~ 1.6x higher magnitude); therefore, we might expect removing them to affect performance less, which is what we previously saw. The opposite is true for SimpleMath: hidden states interactions with the bottom part of the spectrum skew much more towards this dataset than clean data compared to the top of the spectrum, so we might expect removing them possibly harms performance. Again, this is consistent with the large drop seen in Figure 3\. While we find that arithmetic ability is especially brittle, it's not clear from our results that LMs don't also contain delicate structure to solve it (Kantamneni & Tegmark, 2025). Additional discussion and results on math and factual recall are in Appendices 6.1 and 6.2.

### **6.1. Error Analysis on Math Data**

We find that arithmetic is specifically hurt by the K-FAC edit. This could be because arithmetic problems themselves are memorized (at the 7B scale), or because they require narrowly used directions to do precise calculations. We find it interesting, though, that this is so specifically and negatively affected when seemingly related skills remain intact. Besides the logical reasoning benchmarks included in Figure 3, we find that while MMLU-Pro Math drops to 67% of baseline (23.4% to 15.8%), the computer science subset stays at about the same level (Baseline: 26.1%, K-FAC: 26.3%, BSN: 25.1%). An example from GSM8K also shows that despite generating the exact same chain of thought as the baseline model, it failed specifically at making the calculation (Table 3). Not all generations are as similar to the original, but this demonstrates how targeted the new deficit for arithmetic is. We include an example of essentially the opposite (same generation but fixing an error) in Boolean Expressions; see Table 4\.  
**Table 3: Example highlighting arithmetic reasoning error (GSM8K).** *Comparing Baseline vs K-FAC Edit.*

| Type | Content |
| :---- | :---- |
| **Prompt** | Seattle has 20 sheep. Charleston has 4 times as many sheep as Seattle. Toulouse has twice as many sheep as Charleston. How many sheep do they have together? |
| **Ground Truth** | 260 |
| **Baseline** | Seattle has 20 sheep. Charleston has 4 times as many sheep as Seattle. So Charleston has $4 \\times 20=80$ sheep. Toulouse has twice as many sheep as Charleston. So Toulouse has $2 \\times 80=160$ sheep. Together, they have $20+80+160=260$ sheep. So the answer is 260\. |
| **K-FAC Edit** | Seattle has 20 sheep. Charleston has 4 times as many sheep as Seattle. So Charleston has $4 \\times 20=80$ sheep. Toulouse has twice as many sheep as Charleston. So Toulouse has $2 \\times 80=160$ sheep. Together, they have $20+80+160=220$ sheep. So the answer is 220\. |

### **6.2. Error Analysis on Fact Retrieval**

We explore whether specific types of facts are more brittle to a K-FAC edit. A natural question is whether the frequency of a fact changes the probability that it is not forgotten by an edit. We show that more frequent relations in the Relations dataset are less affected by our K-FAC edit to lower eigenvalues of the Hessian (Figure 5). These relations are sorted according to results for the OLMo-1 model, which is trained on a different dataset, but we assume some similarity. We see that the most frequent relations like country-largest-city or person-band-lead-singer change relatively little, going up or down a few points (or experiencing no change), while the least frequent like Company-CEO drop 78% relative to baseline.  
**Table 4: Example demonstrating improved Boolean reasoning after K-FAC editing.**

| Type | Content |
| :---- | :---- |
| **Prompt** | False or not (True) and False is |
| **Ground Truth** | False |
| **Baseline** | ... We first simplify this expression "Z" as follows: $Z=$ False or not (True) and False \= A or B ... Plugging in A and B, we get: $Z=A$ or $B=$ True or False \= True. So the answer is True. |
| **K-FAC** | ... We first simplify this expression "Z" as follows: $Z=$ False or not (True) and False \= A and B ... Plugging in A and B, we get: $Z=A$ and $B=$ True and False \= False. So the answer is False. |

## **7\. Background on Loss Curvature and K-FAC**

**Memorized individual instances exhibit sharp curvature:** Per-example analyses often find that memorized points are locally sharp, meaning the loss has a high second derivative in some directions (Ravikumar et al., 2024; Garg et al., 2024; Hochreiter & Schmidhuber, 1997; Foret et al., 2021). One way to think about this result is that the model is very brittle for that point: if a model memorized a datapoint exactly, and you were to perturb either the input itself, or weights interacting with it, the loss would spike (since the model can no longer recognize the exact point it memorized). Note that this is a simplified view, and that models using better generalizing mechanisms may be more robust to perturbations and the loss is locally flatter. We can quantify how curved the loss landscape is by measuring how quickly the sharpest direction of the Hessian is (through its top eigenvalue) or how much curvature is present in the Hessian (the trace). This way of measurement establishes that there are directions of high and low curvature that we can use to detect memorization. The remainder of this section will cover the connection between K-FAC and a dataset-average picture of loss curvature, and discuss how this picture inverts this intuition about individual points.  
**K-FAC's relationship to curvature:** Following Martens & Grosse (2015); Foret et al. (2021), we study memorization and generalization through the lens of loss curvature, specifically as a function of the model's weights (Keskar et al., 2017). Mathematically, the curvature of the loss landscape is captured by the Hessian $H \= \\nabla\_{\\theta}^2 L(\\theta)$, where $L$ is the loss function and $\\theta$ is the vector of flattened model weights. Practically, though, $H$ is not tractably computable for any but the smallest models, as its size is quadratic in the number of model weights. Prior work bypasses explicitly computing $H$ by approximating its top eigenvalues and/or trace. For our analyses, though, we need a more complete picture of the whole spectrum of $H$, and to get this picture, we turn to the Kronecker-Factored Approximate Curvature (K-FAC) Martens & Grosse (2015). Originally introduced as an efficient natural-gradient method for optimization, K-FAC approximates the Fisher Information Matrix (FIM) and provides a structured approximation to the loss curvature without forming the full Hessian.  
For a model trained with softmax cross-entropy loss, the relationship of the FIM $F$ to the curvature of parameters is given by:  
$$F \= \\mathbb{E}\_{D}\[\\nabla\_{\\theta} \\log p\_{\\theta}(y|x,\\theta) \\nabla\_{\\theta} \\log p\_{\\theta}(y|x,\\theta)^\\top\] \= \\mathbb{E}\_{D}\[\\nabla\_{\\theta}^2 (-\\log p\_{\\theta}(y|x))\]$$  
Here, $D$ is a dataset consisting of input-label pairs $(x,y)$, and $p\_{\\theta}(y|x)$ is the model's predicted label distribution for input $x$. For an individual matrix $W \\in \\mathbb{R}^{d\_{out} \\times d\_{in}}$ with incoming activations $a$ and backpropagated gradients $g$, K-FAC gives an easily computable approximation to a weight matrix $W$'s block of $F$:  
$$F\_W \\approx G \\otimes A \= \\mathbb{E}\[gg^\\top\] \\otimes \\mathbb{E}\[aa^\\top\]$$  
where $A \\in \\mathbb{R}^{d\_{in} \\times d\_{in}}$ and $G \\in \\mathbb{R}^{d\_{out} \\times d\_{out}}$. In words, this is the Kronecker product of the (uncentered) second-moment matrices of the activations going into the layer and the gradients coming out. When computing the loss to backpropagate into $G$, we sample $y$ from the model's predicted label distribution, rather than taking the ground truth $y$. Not only is this important for the correctness of the FIM (Martens & Grosse, 2015), but it also means we can use this method without any labeled data.  
**Instance- vs. population-level curvature:** Our use of the Fisher/K-FAC differs by averaging curvature across data, which emphasizes directions that are consistently important. Idiosyncratic sharp directions associated with specific examples point in different directions and largely cancel in the average, contributing to a low-curvature background. Directions that implement shared mechanisms (used by many inputs) add coherently and remain high-curvature on average. This explains why retaining high curvature mass preserves general abilities, while removing low-curvature components preferentially suppresses recitation.  
**Decomposing K-FAC:** Throughout this work, we will decompose the FIM of a weight matrix $W$ into distinct components (directions) to analyze their role in reciting memorized data. We can compute the eigendecomposition of the FIM by individually eigendecomposing $A$ and $G$ (see Appendix A). We refer to the eigendecomposition of K-FAC as the curvature basis, since the eigenvalues are sorted in terms of most to least loss curvature. A single eigenvector of K-FAC is the outer product of an activations eigenvector and gradient eigenvector, and thus can be considered a weight component of $W$ (i.e., as a matrix with $W$'s size). Therefore, when we describe the 'activation' of data with a rank-one component $C$, we are describing the matrix vector product $Cx$. The magnitude of this product would be the norm of the resulting vector.

## **8\. Discussion and Limitations**

This work shows that we can decompose weight matrices in the loss-curvature basis in real models to disentangle different types of capabilities. At the population level, high loss curvature corresponds to weight components shared across datapoints, while flat directions are flat and (potentially) high curvature for only a few datapoints. We use this to motivate model editing procedures, using both truncation in the curvature and SVD bases, for suppressing memorization that don't require direct unlearning, but outperform a highly competitive method directly optimized to do so. We show that LM capabilities like logical reasoning, fact recall, and arithmetic interact to differing degrees with weight components of high and low curvature. Arithmetic for example, takes a 'path' through weight components that looks more like the path taken by memorized sequences than non-memorized sequences; the opposite is true for logical reasoning tasks. We are excited by future work extending our analysis, as well as exploring directions connecting the loss curvature of different skills in LMs to, e.g., their ease to learn/improve in finetuning, the effective capacity required to acquire them, and ways to make them more robust. One particularly interesting direction would be in understanding whether and in what circumstances models can be trained smaller in order to specialize for reasoning tasks, as our results could possibly suggest.  
While we have made progress connecting behaviors to the loss curvature spectrum, our work has several limitations. We make no claims about fully removing memories from models, as our methods likely suffer from a tendency for memorized data to resurface after further tuning/perturbations (Lee et al., 2025). In terms of fully explaining why some tasks are more sensitive to perturbations and interact more with lower K-FAC eigenvectors, we have to speculate. For example, 'uncommon' weight directions which do not manifest as high curvature directions in K-FAC could correspond to precise and sophisticated structure (Kantamneni & Tegmark, 2025), rather than memorization or narrowly-useful patterns (Nikankin et al., 2025), necessarily. Our approximation of curvature is not perfect, and when estimating the bottommost eigenvalues, could suffer from numerical stability issues. While this doesn't affect our model edit, it may affect other analyses.

## **9\. Conclusion**

We showed that loss-curvature provides a unifying lens for separating memorization from generalization in Transformers: the K-FAC curvature basis disentangles weight directions that support shared, reusable structure (top of the spectrum) from those that chiefly underwrite recitation and brittle behaviors (bottom of the spectrum). Leveraging this finding, we introduced a weight-editing method that preserves a targeted fraction of curvature mass and, across LMs and ViTs, strongly suppresses untargeted memorization while maintaining model coherence and, in the vision setting, even improving validation accuracy. Compared to a supervised unlearning baseline (BSN), our approach requires no forget set, achieves lower perplexity and markedly stronger generalization to unseen memorized content, and exhibits competitive robustness under stress tests. Extending beyond verbatim recall, our analyses position downstream behaviors along a memorization-reasoning continuum: arithmetic and closed-book fact retrieval rely more on low-curvature directions and are disproportionately impacted by edits, whereas open-book and non-numerical logical reasoning are largely preserved or occasionally improved. These results (i) reconcile instance-level sharpness with population-level flatness, (ii) offer practical tools for recitation-reducing model editing, and (iii) go beyond previous results in finding curvature signatures for a range of model behaviors beyond strict memorization and generalization.

## **10\. Acknowledgments**

We would like to thank Michael Byun, Atticus Geiger, Joshua Batson, Roger Grosse, Christopher Potts, Jing Huang, Ekdeep Singh Lubana, Nathan Rourke, Adam Ball, and Thomas McGrath for feedback on drafts of this work.

## **Appendices**

### **A. A Primer on the Eigendecomposition of A and G**

This section provides background on how to think about the eigenvectors and eigenvalues of the Hessian, as approximated by the K-FAC factorization $F \\approx G \\otimes A$. For a given weight matrix, recall that $A$ is the covariance matrix of the activations going into it, and that $A \\in \\mathbb{R}^{d\_{in} \\times d\_{in}}$. $G$ is the covariance matrix of the gradients on the output side of the matrix, and $G \\in \\mathbb{R}^{d\_{out} \\times d\_{out}}$.  
Notice that we have $d\_{in} \* d\_{out}$ eigenpairs in the Hessian. The approximate eigenvalues of the FIM are the products between each of the eigenvalues of the $G$ and $A$ matrices from K-FAC, and the corresponding eigenvectors are the Kronecker products between the eigenvectors of $G$ and $A$.

### **B. Datasets**

**Dolma:** We mine memorized continuations from Dolma to obtain on-distribution memorization examples. Fixed-length windows \[64 | 48\] are sampled per document and a window is labeled memorized iff the teacher-forced argmax at each of the 48 suffix positions equals the gold token. Positives are aggressively deduplicated to avoid inflation from near-identical suffixes (e.g., templatic code/comments). The resulting 1000 sequences are split evenly: one half trains BSN (unlearning) and sweeps K-FAC, and the other half validates both.  
**Historical Quotes:** For each quote (length $\\ge 9$ tokens), the prefix is all but the last 8 tokens and the suffix is the final 8\. We greedily generate 8 tokens from the prefix and mark memorized on exact match. As quotes have canonical phrasing and high surface regularity, an exact-match is meaningful and less sensitive to trivial paraphrases. This 512 dataset is used strictly for validation to check whether methods preserve non-target knowledge while removing targeted memorization.

### **C. nDCG@10 (Token-Ranking Overlap)**

For each token position $t$, the frozen baseline provides a ranked list of its top-K next-token predictions, $B\_t \= \\{b\_{t,1}, \\dots, b\_{t,K}\\}$. After editing, the model produces its own top-K ranking $\\hat{y}\_{t,1:K}$. We assign graded relevance scores based on the presence and rank order of the edited model's predictions within the baseline set:  
$$rel(r) \= \\begin{cases} K-r+1, & \\text{if } \\hat{y}\_{t,r} \\in B\_t, \\\\ 0, & \\text{otherwise}. \\end{cases}$$  
We then compute the Discounted Cumulative Gain (DCG), normalized by the Ideal DCG (IDCG), resulting in the normalized Discounted Cumulative Gain (nDCG):  
$$DCG\_K \= \\sum\_{r=1}^K \\frac{rel(r)}{\\log\_2(r+1)} \\quad IDCG\_K \= \\sum\_{r=1}^K \\frac{K-r+1}{\\log\_2(r+1)}$$$$nDCG \= \\frac{DCG\_K}{IDCG\_K} \\in \[0,1\]$$  
We compute nDCG@10 on the first 200k tokens of the held-out pile10k dataset. Intuitively, this measures how closely the edited model's token-ranking aligns with the baseline, capturing local preference drift. We specifically chose ranking rather than probabilities to isolate the ordering of high-probability tokens, as these largely determine predictive entropy. We report the mean nDCG@10 across positions (higher is better) as an indication of how faithfully the model preserves the baseline's preference structure post-edit.

### **D. Hyperparameters**

**Model Settings:**

* **K-FAC:** The full hyperparameter search details can be found below. In LMs we edit layers 23, 24, and 25 at 60% energy retained in the up and gate projections in MLPs. In ViTs, we edit layers 0 and 11 to 75% energy on both up and down MLP projections.  
* **BSN:** The best BSN settings for editing the language model were a loss weight of 0.7, 5 epochs, a sparsity ratio of 0.0015, and a learning rate of 0.3.

**D.1 K-FAC Compression Configuration** We systematically explored applying K-FAC compression to selected Transformer MLP layers of the OLMo-2 models. Our experiments focused on two primary hyperparameters:

* **Energy threshold:** Instead of selecting a fixed number of eigenvectors, we retained eigenvectors based on a cumulative "energy" threshold—the fraction of the total eigenvalue sum preserved. We tested thresholds ranging from 60% (stronger compression) to 90% (milder compression), evaluating their effects for gate, up, and down MLP projections.  
* **Layer selection:** We tested subsets from the 32 MLP layers, targeting early, intermediate, and deep parts of the model, both individually and in combinations up to three layers.

**D.2 Balanced Subnet (BSN) Configuration** For Balanced Subnet (BSN), we started from the original authors' implementation, making minimal adjustments necessary to handle OLMo-2's Transformer architecture and optimize performance given its larger parameter set. We began with hyperparameter ranges recommended by the BSN authors, then expanded them based on initial results. Our final hyperparameter search included:

* **Ratio:** Controls mask sparsity. Expanded to \[0.001, 0.05\].  
* **Loss weighting:** Balances clean vs. memorized examples. Expanded to \[0.1, 0.3, 0.5, 0.7, 0.9\].  
* **Epochs:** Expanded to 1-10.  
* **Include gate:** Optionally includes masking the MLP gate projection.

### **E. Perplexity (Clean Text)**

We compute perplexity on clean, held-out text derived from the held-out pile10k dataset, following Balanced Subnet (BSN) evaluation approach. Perplexity is measured both before and after applying memorization edits, serving as a baseline metric to identify unintended deterioration in the model's general language modeling capabilities.

### **F. 1B Model Results**

For the 1B model, we mined memorized sequences in a similar fashion as the 7B model. We split the 650 Dolma sequences into 525 train and 125 validation sequences. For Historical Quotes, we evaluated on all 664 sequences since they were not used during training. The baseline model shows near-perfect memorization on both datasets (98.5% strict accuracy). As shown in Table 5, all three unlearning methods successfully reduce memorization on the Dolma validation set to under 4% strict accuracy. However, their transfer to the Historical Quotes dataset varies significantly.  
**Table 5: Comparison of unlearning methods on OLMo-2 1B model.**

| Method | Dolma Val Strict (%) | Dolma Val Loose (%) | Dolma Avg Lev ↑ | Hist. Quotes Strict (%) | Hist. Quotes Loose (%) | Hist. Quotes Avg Lev ↑ |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Baseline | 98.46 | 99.38 | 0.005 | 98.5 | 98.95 | 0.006 |
| BSN | 3.0 | 5.0 | 0.900 | 57.0 | 66.0 | 0.250 |
| K-FAC | 2.8 | 7.2 | 0.761 | 27.7 | 39.9 | 0.470 |
| SVD | 3.2 | 6.3 | 0.781 | 39.6 | 48.5 | 0.401 |

### **G. Stress Tests**

Huang et al. (2024) reported substantial sensitivity to positional perturbations, with average exact match lengths increasing from 19 to 35 tokens (+16 tokens) for gradient ascent, and from 23 to 36 tokens (+13 tokens) for sparse fine-tuning. By comparison, our K-FAC method showed a smaller absolute increase, from 6.6 to 13.3 tokens (+6.7 tokens), and BSN increased from 4.6 to 10 tokens (+5.4 tokens).  
**Table 6: Effect of positional perturbation stress tests on memorization extraction.**

| Method | Original | Perturbed |
| :---- | :---- | :---- |
| BSN | $4.6 \\pm 10.45$ | $9.79 \\pm 14.36$ |
| K-FAC | $6.64 \\pm 8.6$ | $13.27 \\pm 9.34$ |
| SVD | $6.33 \\pm 8.93$ | $12.78 \\pm 9.19$ |

### **H. ViT Results**

**H.1 The Effect of Weight Decay:** While training ViT models, we observed that there is a strong effect of weight decay on the separability. In fact, something analogous was observed in Yunis et al. (2024), in which the effective rank of the singular value spectrum was observed to decrease as weight decay increased, something that they connect to generalization and memorization.

### **I. Further Benchmark Results on LMs**

**Table 7: Coherence Metrics.**

| Method | nDCG@10 | Perplexity |
| :---- | :---- | :---- |
| Baseline | 1.000 | 19.04 |
| BSN | 0.97 | 23.59 |
| K-FAC | 0.91 | 22.84 |

**Table 8: Benchmark results for OLMo 2 7B.**

| Task | Baseline | BSN | K-FAC |
| :---- | :---- | :---- | :---- |
| TriviaQA | 0.780 | 0.766 | 0.648 |
| Relations | 74.855 | 0.268 | 64.390 |
| PopQA | 0.807 | 0.779 | 0.598 |
| OBQA | 0.804 | 0.790 | 0.800 |
| CSQA | 0.751 | 0.722 | 0.731 |
| TriviaQA-Open | 0.760 | 0.708 | 0.720 |
| OBQA+Fact | 0.888 | 0.884 | 0.894 |
| BoolQ | 0.863 | 0.853 | 0.854 |
| GSM8K | 0.675 | 0.610 | 0.447 |
| Winogrande | 0.772 | 0.761 | 0.755 |
| MMLU-Pro | 0.499 / 0.283 | 0.463 / 0.270 | 0.475 / 0.253 |
| MMLU-Pro Math | 0.234 | 0.226 | 0.158 |
| MMLU-Pro CS | 0.261 | 0.251 | 0.263 |

**Table 9: Individual task results for BigBench-Hard (OLMo 2 7B).** *(Selected tasks)*

| Task | OLMo (%) | KFAC (%) | KFAC Diff | BSN (%) | BSN Diff |
| :---- | :---- | :---- | :---- | :---- | :---- |
| boolean\_expressions | 76.40 | 81.20 | 4.80 | 75.60 | \-0.80 |
| logical\_deduction\_three\_objects | 56.40 | 60.80 | 4.40 | 56.80 | 0.40 |
| web\_of\_lies | 82.40 | 82.80 | 0.40 | 82.00 | \-0.40 |
| sports\_understanding | 84.00 | 83.20 | \-0.80 | 66.40 | \-17.60 |
| object\_counting | 62.40 | 39.60 | \-22.80 | 45.60 | \-16.80 |

**Table 10: Individual evaluations for MMLU-Pro (OLMo 2 7B).**

| Task | OLMo (%) | KFAC (%) | KFAC Diff | BSN (%) | BSN Diff |
| :---- | :---- | :---- | :---- | :---- | :---- |
| psychology | 42.73 | 43.73 | 1.00 | 43.23 | 0.50 |
| computer science | 26.10 | 26.34 | 0.24 | 25.12 | \-0.98 |
| math | 23.39 | 15.84 | \-7.55 | 22.58 | \-0.81 |
| business | 26.11 | 20.41 | \-5.70 | 21.80 | \-4.31 |
| **Average** | **29.92** | **27.32** | **\-2.60** | **28.35** | **\-1.57** |

### **J. Example LM Generations**

**Table 12: OLMo 2 7B Generations (Random / Common Prompts).**

| Prompt | Baseline Output | BSN Output | K-FAC Output | SVD Output |
| :---- | :---- | :---- | :---- | :---- |
| Mr. and Mrs. Dursley, of number four, | Privet Drive, were proud to say that they were pe | Privet Drive, were proud to say that they were pe | Privet Drive, London, were proud to say that they | Privet Drive, were proud to say that they were pe |
| The capital of France is | Paris. It is the largest city in France. It is al | Paris. It is the largest city in France. It is al | Paris. The capital of France is Paris. The capita | Paris. It is the largest city in France. It is al |
| Double, double toil and trouble | Fire burn, and cauldron bubble. Double, double | Fire burn, and cauldron bubble. Double, double | , toil and trouble, toil and trouble, toil and tro | Fire burn, and cauld well, bubbling-bubb In a wi |
| The quick brown fox jumps | over the lazy dog. The quick brown fox jumps ove | over the lazy dog. The quick brown fox jumps over | over the lazy dog. The quick brown fox jumps over | over the lazy dog. The quick brown fox jumps over |

