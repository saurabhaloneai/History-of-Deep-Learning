
# speedrun implemntation of history-of-deep-learning 🐳

- here i am trying to implement around 60 imp DL papers. (it will be completed between aug-sep..hope so : )

- why am i doing this? because i'm retarded, so my notes and code might be retarded sometimes. so, be careful.

- inspired by **adam-maj** - I added few more papers and few sections.

- three stage of implemntation : from Scrath, in pytorch and in jax(not all but some).

- my approach is to first gather all resource learn and also i will be keep updating the repo.

- this repo is strictly my personal ml notes. thank u : ) 

- my ml resource stack : [link](https://whimsical.com/current-goals-NP2xuDwNCMhKmZyWLDw4ch)


## Totalcount : (9/60)

## 01-deep-neural-networks

| Concept       | Complete |
|---------------|-------|
| BackPropagation | ✅   |
| CNN           | ✅   |
| AlexNet       | ✅   |
| U-net         | ✅   |

## 02-optimization-and-regularization

| Concept         | Complete |
|-----------------|-------|
| weights-decay   |  ✅   |
| relu            |  ✅   |
| residuals       |       |
| dropout         |  ✅   |
| batch-norm      |       |
| layer-norm      |       |
| gelu            |  ✅   |
| adam            |       |
| early-stopping  |   ✅   |

## 03-sequence-modeling

| Concept           | Complete |
|-------------------|-------|
| rnn               |    |
| lstm              |    |
| learning-to-forget|    |
| word2vec          |    |
| seq2seq           |    |
| attention         |    |
| mixture-of-experts|    |

## 04-transformer

| Concept            | Complete |
|--------------------|-------|
| transformer        |    |
| bert               |    |
| t5                 |    |
| gpt                |    |
| lora               |    |
| rlhf               |    |
| vision-transformer |    |

## 05-image-generation

| Concept         | Complete |
|-----------------|-------|
| gans            |    |
| vae             |    |
| diffusion       |    |
| clip            |    |
| dall-e          |    |

## 06-reinforcement-learning

| Concept         | Complete |
|-----------------|-------|
| Q-learning       |    |

## 07-machine-learning 

| Algorithm                 | Complete    |
|---------------------------|-------------|
| Linear Regression         |             |
| Logistic Regression       |             |
| Decision Trees            |             |
| Random Forest             |             |
| Support Vector Machines   |             |
| K-Nearest Neighbors       |             |
| K-Means Clustering        |             |
| Naive Bayes               |             |
| PCA                       |             |
| Perceptron                |             |



---
# Papers 

- [x]  **DNN** - Learning Internal Representations by Error Propagation (1987), D. E. Rumelhart et al. [[PDF]](https://www.notion.so/Papers-587fcad411304657b7ef990db5299e65?pvs=21)
- [x]  **CNN** - Backpropagation Applied to Handwritten Zip Code Recognition (1989), Y. Lecun et al. [[PDF]](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf)
- [x]  **LeNet** - Gradient-Based Learning Applied to Document Recognition (1998), Y. Lecun et al. [[PDF]](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
- [x]  **AlexNet** - ImageNet Classification with Deep Convolutional Networks (2012), A. Krizhevsky et al. [[PDF]](https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- [x]  **U-Net** - U-Net: Convolutional Networks for Biomedical Image Segmentation (2015), O. Ronneberger et al. [[PDF]](https://arxiv.org/abs/1505.04597)
- [x]  **Weight Decay** - A Simple Weight Decay Can Improve Generalization (1991), A. Krogh and J. Hertz [[PDF]](https://proceedings.neurips.cc/paper/1991/file/8eefcfdf5990e441f0fb6f3fad709e21-Paper.pdf)
- [x]  **ReLU** - Deep Sparse Rectified Neural Networks (2011), X. Glorot et al. [[PDF]](https://www.researchgate.net/publication/215616967_Deep_Sparse_Rectifier_Neural_Networks)
- [x]  **Residuals** - Deep Residual Learning for Image Recognition (2015), K. He et al. [[PDF]](https://arxiv.org/pdf/1512.03385)
- [x]  **Dropout** - Dropout: A Simple Way to Prevent Neural Networks from Overfitting (2014), N. Strivastava et al. [[PDF]](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
- [x]  **BatchNorm** - Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (2015), S. Ioffe and C. Szegedy [[PDF]](https://arxiv.org/pdf/1502.03167)
- [x]  **LayerNorm** - Layer Normalization (2016), J. Lei Ba et al. [[PDF]](https://arxiv.org/pdf/1607.06450)
- [x]  **GELU** - Gaussian Error Linear Units (GELUs) (2016), D. Hendrycks and K. Gimpel [[PDF]](https://arxiv.org/pdf/1606.08415)
- [x]  **Adam** - Adam: A Method for Stochastic Optimization (2014), D. P. Kingma and J. Ba [[PDF]](https://arxiv.org/pdf/1412.6980)
- [ ]  **RNN** - A Learning Algorithm for Continually Running Fully Recurrent Neural Networks (1989), R. J. Williams [[PDF]](https://gwern.net/doc/ai/nn/rnn/1989-williams-2.pdf)
- [ ]  **LSTM** - Long-Short Term Memory (1997), S. Hochreiter and J. Schmidhuber [[PDF]](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [ ]  **Learning to Forget** - Learning to Forget: Continual Prediction with LSTM (2000), F. A. Gers et al. [[PDF]](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=e10f98b86797ebf6c8caea6f54cacbc5a50e8b34)
- [ ]  **Word2Vec** - Efficient Estimation of Word Representations in Vector Space (2013), T. Mikolov et al. [[PDF]](https://arxiv.org/pdf/1301.3781)
- [ ]  **Phrase2Vec** - Distributed Representations of Words and Phrases and their Compositionality (2013), T. Mikolov et al. [[PDF]](https://arxiv.org/pdf/1310.4546)
- [ ]  **Encoder-Decoder** - Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation (2014), K. Cho et al. [[PDF]](https://arxiv.org/pdf/1406.1078)
- [ ]  **Seq2Seq** - Sequence to Sequence Learning with Neural Networks (2014), I. Sutskever et al. [[PDF]](https://arxiv.org/pdf/1409.3215)
- [ ]  **Attention** - Neural Machine Translation by Jointly Learning to Align and Translate (2014), D. Bahdanau et al. [[PDF]](https://arxiv.org/pdf/1409.0473)
- [ ]  **Mixture of Experts** - Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (2017), N. Shazeer et al. [[PDF]](https://arxiv.org/pdf/1701.06538)
- [ ]  **Transformer** - Attention Is All You Need (2017), A. Vaswani et al. [[PDF]](https://arxiv.org/pdf/1706.03762)
- [ ]  **BERT** - BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018), J. Devlin et al. [[PDF]](https://arxiv.org/pdf/1810.04805)
- [ ]  **RoBERTa** - RoBERTa: A Robustly Optimized BERT Pretraining Approach (2019), Y. Liu et al. [[PDF]](https://arxiv.org/pdf/1907.11692)
- [ ]  **T5** - Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (2019), C. Raffel et al. [[PDF]](https://arxiv.org/pdf/1910.10683)
- [ ]  **GPT-2** - Language Models are Unsupervised Multitask Learners (2018), A. Radford et al. [[PDF]](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [ ]  **GPT-3** - Language Models are Few-Shot Learners (2020) T. B. Brown et al. [[PDF]](https://arxiv.org/pdf/2005.14165)
- [ ]  **LoRA -** LoRA: Low-Rank Adaptation of Large Language Models (2021), E. J. Hu et al. [[PDF]](https://arxiv.org/pdf/2106.09685)
- [ ]  **RLHF** - Fine-Tuning Language Models From Human Preferences (2019), D. Ziegler et al. [[PDF]](https://arxiv.org/pdf/1909.08593)
- [ ]  **PPO** - Proximal Policy Optimization Algorithms (2017), J. Schulman et al. [[PDF]](https://arxiv.org/pdf/1707.06347)
- [ ]  **InstructGPT** - Training language models to follow instructions with human feedback (2022), L. Ouyang et al. [[PDF]](https://arxiv.org/pdf/2203.02155)
- [ ]  **Helpful & Harmless** - Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback (2022), Y. Bai et al. [[PDF]](https://arxiv.org/pdf/2204.05862)
- [ ]  **Vision Transformer** - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (2020), A. Dosovitskiy et al. [[PDF]](https://arxiv.org/pdf/2010.11929)
- [ ]  **GAN** - Generative Adversarial Networks (2014), I. J. Goodfellow et al. [[PDF]](https://arxiv.org/pdf/1406.2661)
- [ ]  **VAE** - Auto-Encoding Variational Bayes (2013), D. Kingma and M. Welling [[PDF]](https://arxiv.org/pdf/1312.6114)
- [ ]  **VQ VAE** - Neural Discrete Representation Learning (2017), A. Oord et al. [[PDF]](https://arxiv.org/pdf/1711.00937)
- [ ]  **VQ VAE 2** - Generating Diverse High-Fidelity Images with VQ-VAE-2 (2019), A. Razavi et al. [[PDF]](https://arxiv.org/pdf/1906.00446)
- [ ]  **Diffusion** - Deep Unsupervised Learning using Nonequilibrium Thermodynamics (2015), J. Sohl-Dickstein et al. [[PDF]](https://arxiv.org/pdf/1503.03585)
- [ ]  **Denoising Diffusion** - Denoising Diffusion Probabilistic Models (2020), J. Ho. et al. [[PDF]](https://arxiv.org/pdf/2006.11239)
- [ ]  **Denoising Diffusion 2** - Improved Denoising Diffusion Probabilistic Models (2021), A. Nichol and P. Dhariwal [[PDF]](https://arxiv.org/pdf/2102.09672)
- [ ]  **Diffusion Beats GANs** - Diffusion Models Beat GANs on Image Synthesis, P. Dhariwal and A. Nichol [[PDF]](https://arxiv.org/pdf/2105.05233)
- [ ]  **CLIP** - Learning Transferable Visual Models From Natural Language Supervision (2021), A. Radford et al. [[PDF]](https://arxiv.org/pdf/2103.00020)
- [ ]  **DALL E** - Zero-Shot Text-to-Image Generation (2021), A. Ramesh et al. [[PDF]](https://arxiv.org/pdf/2102.12092)
- [ ]  **DALL E 2** - Hierarchical Text-Conditional Image Generation with CLIP Latents (2022), A. Ramesh et al. [[PDF]](https://arxiv.org/pdf/2204.06125)
- [ ]  **Deep Learning** - Deep Learning (2015), Y. LeCun, Y. Bengio, and G. Hinton [[PDF]](https://www.nature.com/articles/nature14539.pdf)
- [ ]  **GAN** - Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (2016), A. Radford et al. [[PDF]](https://arxiv.org/pdf/1511.06434)
- [ ]  **DCGAN** - Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (2016), A. Radford et al. [[PDF]](https://arxiv.org/pdf/1511.06434)
- [ ]  **BigGAN** - Large Scale GAN Training for High Fidelity Natural Image Synthesis (2018), A. Brock et al. [[PDF]](https://arxiv.org/pdf/1809.11096)
- [ ]  **WaveNet** - WaveNet: A Generative Model for Raw Audio (2016), A. van den Oord et al. [[PDF]](https://arxiv.org/pdf/1609.03499)
- [ ]  **BERTology** - A Survey of BERT Use Cases (2020), R. Rogers et al. [[PDF]](https://arxiv.org/pdf/2002.12327)
- [ ]  **GPT** - Improving Language Understanding by Generative Pre-Training (2018), A. Radford et al. [[PDF]](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [ ]  **GPT-4** - GPT-4 Technical Report (2023), OpenAI [[PDF]](https://arxiv.org/pdf/2303.08774)
- [ ]  **Deep Reinforcement Learning** - Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (2017), D. Silver et al. [[PDF]](https://arxiv.org/pdf/1712.01815)
- [ ]  **Deep Q-Learning** - Playing Atari with Deep Reinforcement Learning (2013), V. Mnih et al. [[PDF]](https://arxiv.org/pdf/1312.5602)
- [ ]  **AlphaGo** - Mastering the Game of Go with Deep Neural Networks and Tree Search (2016), D. Silver et al. [[PDF]](https://www.nature.com/articles/nature16961)
- [ ]  **AlphaFold** - Highly accurate protein structure prediction with AlphaFold (2021), J. Jumper et al. [[PDF]](https://www.nature.com/articles/s41586-021-03819-2)
- [ ]  **T5** - Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (2019), C. Raffel et al. [[PDF]](https://arxiv.org/pdf/1910.10683)
- [ ]  **ELECTRA** - ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators (2020), K. Clark et al. [[PDF]](https://arxiv.org/pdf/2003.10555)
- [ ]  **SimCLR** - A Simple Framework for Contrastive Learning of Visual Representations (2020), T. Chen et al. [[PDF]](https://arxiv.org/pdf/2002.05709)
