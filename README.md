# Deep Learning Research Paper Collection 

## Overview

This repository is a collection of IMPORTANT deep learning research papers, organized by research area and implementation. The goal is to provide a structured approach to understanding the evolution and core concepts of deep learning.

## Disclaimer

> [!IMPORTANT]
> This is a personal learning project. The implementations and notes may contain errors or simplifications. Use with caution and always refer to the original papers.

## Inspiration and Credits

Inspired by [adam-maj](https://github.com/adam-maj) and expanded with additional research papers and implementations.

## Project Goals

- Implement approximately 60 important deep learning papers
- Provide scratch implementations for learning and understanding
- Create a comprehensive resource for deep learning research

## Contents

### 1. Foundational Deep Neural Networks

#### Papers
- **DNN** (1987): Learning Internal Representations by Error Propagation [pdf](https://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf)
- **CNN** (1989): Backpropagation Applied to Handwritten Zip Code Recognition [pdf](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf)
- **LeNet** (1998): Gradient-Based Learning Applied to Document Recognition [pdf](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
- **AlexNet** (2012): ImageNet Classification with Deep Convolutional Networks [pdf](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- **U-Net** (2015): Convolutional Networks for Biomedical Image Segmentation [pdf](https://arxiv.org/pdf/1505.04597.pdf)

### 2. Optimization and Regularization Techniques

#### Papers
- **Weight Decay** (1991): A Simple Weight Decay Can Improve Generalization [pdf](https://www.cs.toronto.edu/~hinton/absps/nips93.pdf)
- **ReLU** (2011): Deep Sparse Rectified Neural Networks [pdf](https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf)
- **Residuals** (2015): Deep Residual Learning for Image Recognition [pdf](https://arxiv.org/pdf/1512.03385.pdf)
- **Dropout** (2014): Preventing Neural Networks from Overfitting [pdf](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
- **BatchNorm** (2015): Accelerating Deep Network Training [pdf](https://arxiv.org/pdf/1502.03167.pdf)
- **LayerNorm** (2016): Layer Normalization [pdf](https://arxiv.org/pdf/1607.06450.pdf)
- **GELU** (2016): Gaussian Error Linear Units [pdf](https://arxiv.org/pdf/1606.08415.pdf)
- **Adam** (2014): Stochastic Optimization Method [pdf](https://arxiv.org/pdf/1412.6980.pdf)

### 3. Sequence Modeling

#### Papers
- **RNN** (1989): Continually Running Fully Recurrent Neural Networks [pdf](https://www.bioinf.jku.at/publications/older/2604.pdf)
- **LSTM** (1997): Long-Short Term Memory [pdf](https://www.bioinf.jku.at/publications/older/2308.pdf)
- **Learning to Forget** (2000): Continual Prediction with LSTM [pdf](https://www.researchgate.net/publication/221601044_Learning_to_Forget_Continual_Prediction_with_LSTM)
- **Word2Vec** (2013): Word Representations in Vector Space [pdf](https://arxiv.org/pdf/1301.3781.pdf)
- **Phrase2Vec** (2013): Distributed Representations of Words and Phrases [pdf](https://arxiv.org/pdf/1310.4546.pdf)
- **Encoder-Decoder** (2014): RNN Encoder-Decoder for Machine Translation [pdf](https://arxiv.org/pdf/1406.1078.pdf)
- **Seq2Seq** (2014): Sequence to Sequence Learning [pdf](https://arxiv.org/pdf/1409.3215.pdf)
- **Attention** (2014): Neural Machine Translation with Alignment [pdf](https://arxiv.org/pdf/1409.0473.pdf)
- **Mixture of Experts** (2017): Sparsely-Gated Neural Networks [pdf](https://arxiv.org/pdf/1701.06538.pdf)

### 4. Language Modeling

#### Papers
- **Transformer** (2017): Attention Is All You Need [pdf](https://arxiv.org/pdf/1706.03762.pdf)
- **BERT** (2018): Bidirectional Transformers for Language Understanding [pdf](https://arxiv.org/pdf/1810.04805.pdf)
- **RoBERTa** (2019): Robustly Optimized BERT Pretraining [pdf](https://arxiv.org/pdf/1907.11692.pdf)
- **T5** (2019): Unified Text-to-Text Transformer [pdf](https://arxiv.org/pdf/1910.10683.pdf)
- **GPT Series**:
  - GPT (2018): Generative Pre-Training [pdf](https://arxiv.org/pdf/1810.04805.pdf)
  - GPT-2 (2018): Unsupervised Multitask Learning [pdf](https://arxiv.org/pdf/1902.01082.pdf)
  - GPT-3 (2020): Few-Shot Learning [pdf](https://arxiv.org/pdf/2005.14165.pdf)
  - GPT-4 (2023): Advanced Language Model [pdf](https://arxiv.org/pdf/2303.08774.pdf)
- **LoRA** (2021): Low-Rank Adaptation of Large Language Models [pdf](https://arxiv.org/pdf/2106.09685.pdf)
- **RLHF** (2019): Fine-Tuning from Human Preferences [pdf](https://arxiv.org/pdf/1909.08593.pdf)
- **InstructGPT** (2022): Following Instructions with Human Feedback [pdf](https://arxiv.org/pdf/2203.02155.pdf)
- **Vision Transformer** (2020): Image Recognition with Transformers [pdf](https://arxiv.org/pdf/2010.11929.pdf)
- **ELECTRA** (2020): Discriminative Pre-training [pdf](https://arxiv.org/pdf/2003.10555.pdf)

### 5. Image Generative Modeling

#### Papers
- **GAN** (2014): Generative Adversarial Networks [pdf](https://arxiv.org/pdf/1406.2661.pdf)
- **VAE** (2013): Auto-Encoding Variational Bayes [pdf](https://arxiv.org/pdf/1312.6114.pdf)
- **VQ VAE** (2017): Neural Discrete Representation Learning [pdf](https://arxiv.org/pdf/1711.00937.pdf)
- **Diffusion Models**:
  - Initial Diffusion (2015): Nonequilibrium Thermodynamics [pdf](https://arxiv.org/pdf/1503.03585.pdf)
  - Denoising Diffusion (2020): Probabilistic Models [pdf](https://arxiv.org/pdf/2006.11239.pdf)
  - Improved Denoising Diffusion (2021) [pdf](https://arxiv.org/pdf/2102.09672.pdf)
- **CLIP** (2021): Visual Models from Natural Language Supervision [pdf](https://arxiv.org/pdf/2103.00020.pdf)
- **DALL-E** (2021-2022): Text-to-Image Generation [pdf](https://arxiv.org/pdf/2102.12092.pdf)
- **SimCLR** (2020): Contrastive Learning of Visual Representations [pdf](https://arxiv.org/pdf/2002.05709.pdf)

### 6. Deep Reinforcement Learning

#### Papers
- **Deep Reinforcement Learning** (2017): Mastering Chess and Shogi [pdf](https://arxiv.org/pdf/1712.01815.pdf)
- **Deep Q-Learning** (2013): Playing Atari Games [pdf](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
- **AlphaGo** (2016): Mastering the Game of Go [pdf](https://www.nature.com/articles/nature16961.pdf)
- **AlphaFold** (2021): Protein Structure Prediction [pdf](https://www.nature.com/articles/s41586-021-03819-2.pdf)

### 7. Additional Influential Papers

- **Deep Learning Survey** (2015): By LeCun, Bengio, and Hinton [pdf](https://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf)
- **BigGAN** (2018): Large Scale GAN Training [pdf](https://arxiv.org/pdf/1809.11096.pdf)
- **WaveNet** (2016): Generative Model for Raw Audio [pdf](https://arxiv.org/pdf/1609.03499.pdf)
- **BERTology** (2020): Survey of BERT Use Cases [pdf](https://arxiv.org/pdf/2002.10063.pdf)

#### Scaling and Model Optimization
- **Scaling Laws for Neural Language Models** (2020): Predicting Model Performance [pdf](https://arxiv.org/pdf/2001.08361.pdf)
- **Chinchilla** (2022): Training Compute-Optimal Large Language Models [pdf](https://arxiv.org/pdf/2203.15556.pdf)
- **Gopher** (2022): Scaling Language Models with Massive Compute [pdf](https://arxiv.org/pdf/2112.11446.pdf)

#### Fine-tuning and Adaptation
- **P-Tuning** (2021): Prompt Tuning with Soft Prompts [pdf](https://arxiv.org/pdf/2103.10385.pdf)
- **Prefix-Tuning** (2021): Optimizing Continuous Prompts [pdf](https://arxiv.org/pdf/2101.00190.pdf)
- **AdaLoRA** (2023): Adaptive Low-Rank Adaptation [pdf](https://arxiv.org/pdf/2303.10512.pdf)
- **QLoRA** (2023): Efficient Fine-Tuning of Quantized Models [pdf](https://arxiv.org/pdf/2305.14314.pdf)

#### Inference and Optimization Techniques
- **FlashAttention** (2022): Fast and Memory-Efficient Attention [pdf](https://arxiv.org/pdf/2205.14135.pdf)
- **FlashAttention-2** (2023): Faster Attention Mechanism [pdf](https://arxiv.org/pdf/2307.08691.pdf)
- **Direct Preference Optimization (DPO)** (2023): Aligning Language Models with Human Preferences [pdf](https://arxiv.org/pdf/2305.18046.pdf)
- **LoRA** (2021): Low-Rank Adaptation of Large Language Models [pdf](https://arxiv.org/pdf/2106.09685.pdf)

#### Pre-training and Model Architecture
- **Mixture of Experts (MoE)** (2022): Scaling Language Models with Sparse Experts [pdf](https://arxiv.org/pdf/2201.05596.pdf)
- **GLaM** (2021): Efficient Scaling with Mixture of Experts [pdf](https://arxiv.org/pdf/2112.06905.pdf)
- **Switch Transformers** (2022): Scaling to Trillion Parameter Models [pdf](https://arxiv.org/pdf/2101.03961.pdf)

#### Reasoning and Capabilities
- **Chain of Thought Prompting** (2022): Reasoning with Language Models [pdf](https://arxiv.org/pdf/2201.11903.pdf)
- **Self-Consistency** (2022): Improving Language Model Reasoning [pdf](https://arxiv.org/pdf/2203.11171.pdf)
- **Tree of Thoughts** (2023): Deliberate Problem Solving [pdf](https://arxiv.org/pdf/2305.10601.pdf)

#### Efficiency and Compression
- **DistilBERT** (2019): Distilled Version of BERT [pdf](https://arxiv.org/pdf/1910.01108.pdf)
- **Knowledge Distillation** (2022): Comprehensive Survey [pdf](https://arxiv.org/pdf/2006.05525.pdf)
- **Pruning and Quantization Techniques** (2022): Model Compression Survey [pdf](https://arxiv.org/pdf/2102.06322.pdf)