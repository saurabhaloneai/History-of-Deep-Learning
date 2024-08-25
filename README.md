
# Speedrun Implemntation Of History Of Deep Learning 🐳

> [!NOTE]
>
> ... hey hi 
>
> ... i'm attempting to implement around 60 important DL papers here.(almost from scratch)
>
> ... i'm doing this because i'm retarded and my notes and code might be retarded sometimes. so, be careful.
>
> ... inspired by [**adam-maj**](https://github.com/adam-maj) -> I added few more papers and few sections.
>
> ... my ml resource stack : [link](https://whimsical.com/current-goals-NP2xuDwNCMhKmZyWLDw4ch)

> [!IMPORTANT]
>
> ... as part of increasing my depth of understanding, i have build the image-captioning project with 
> `resnet+attention+lstms`.
>
> ... you can find the project [here](https://github.com/saurabhaloneai/image-cap.git)

--------

# Contents

[Deep Neural Networks](#deep-neural-networks)

- [DNN](01-deep-neural-networks/01-dnn)
- [CNN](01-deep-neural-networks/02-cnn)
- [LeNet](01-deep-neural-networks/03-lenet)
- [AlexNet](01-deep-neural-networks/04-alexnet)
- [U-Net](01-deep-neural-networks/05-unet)

[Optimization and Regularization](#optimization-and-regularization)

- [Weight Decay](02-optimization-and-regularization/01-weight-decay)
- [ReLU](02-optimization-and-regularization/02-relu)
- [Residuals](02-optimization-and-regularization/03-residuals)
- [Dropout](02-optimization-and-regularization/04-dropout)
- [BatchNorm](02-optimization-and-regularization/05-batchnorm)
- [LayerNorm](02-optimization-and-regularization/06-layernorm)
- [GELU](02-optimization-and-regularization/07-gelu)
- [Adam](02-optimization-and-regularization/08-adam)

[Sequence Modeling](#sequence-modeling)

- [RNN](03-sequence-modeling/01-rnn)
- [LSTM](03-sequence-modeling/02-lstm)
- [Learning to Forget](03-sequence-modeling/03-learning-to-forget)
- [Word2Vec](03-sequence-modeling/04-word2vec)
- [Phrase2Vec](03-sequence-modeling/05-phrase2vec)
- [Encoder-Decoder](03-sequence-modeling/06-encoder-decoder)
- [Seq2Seq](03-sequence-modeling/07-seq2seq)
- [Attention](03-sequence-modeling/08-attention)
- [Mixture of Experts](03-sequence-modeling/09-mixture-of-experts)

[Language Modeling](#language-modeling)

- [Transformer](04-language-modeling/01-transformer)
- [BERT](04-language-modeling/02-bert)
- [RoBERTa](04-language-modeling/03-roberta)
- [T5](04-language-modeling/04-t5)
- [GPT](04-language-modeling/05-gpt)
- [GPT-4](04-language-modeling/06-gpt-4)
- [GPT-2](04-language-modeling/07-gpt-2)
- [GPT-3](04-language-modeling/08-gpt-3)
- [LoRA](04-language-modeling/09-lora)
- [RLHF](04-language-modeling/10-rlhf)
- [PPO](04-language-modeling/11-ppo)
- [InstructGPT](04-language-modeling/12-instructgpt)
- [Helpful & Harmless](04-language-modeling/13-helpful-and-harmless)
- [Vision Transformer](04-language-modeling/14-vision-transformer)
- [ELECTRA](04-language-modeling/15-electra)

[Image Generative Modeling](#image-generative-modeling)

- [GAN](05-image-generative-modeling/01-gan)
- [VAE](05-image-generative-modeling/02-vae)
- [VQ VAE](05-image-generative-modeling/03-vq-vae)
- [VQ VAE 2](05-image-generative-modeling/04-vq-vae-2)
- [Diffusion](05-image-generative-modeling/05-diffusion)
- [Denoising Diffusion](05-image-generative-modeling/06-denoising-diffusion)
- [Denoising Diffusion 2](05-image-generative-modeling/07-denoising-diffusion-2)
- [Diffusion Beats GANs](05-image-generative-modeling/08-diffusion-beats-gans)
- [CLIP](05-image-generative-modeling/09-clip)
- [DALL E](05-image-generative-modeling/10-dall-e)
- [DALL E 2](05-image-generative-modeling/11-dall-e-2)
- [SimCLR](05-image-generative-modeling/12-simclr)

[Deep Reinforcement Learning](#deep-reinforcement-learning)

- [Deep Reinforcement Learning](06-deep-reinforcement-learning/01-deep-reinforcement-learning)
- [Deep Q-Learning](06-deep-reinforcement-learning/02-deep-q-learning)
- [AlphaGo](06-deep-reinforcement-learning/03-alphago)
- [AlphaFold](06-deep-reinforcement-learning/04-alphafold)

[Machine Learning](#machine-learning)

- [Linear Regression](07-machine-learning/01-linear-regression) 
- [Logistic Regression](07-machine-learning/02-logistic-regression)
- [Decision Trees](07-machine-learning/03-decision-trees)
- [Random Forest](07-machine-learning/04-random-forest)
- [SVM](07-machine-learning/05-svm)
- [K-Nearest Neighbors](07-machine-learning/06-k-nearest-neighbors)
- [K-Means](07-machine-learning/07-k-means)
- [PCA](07-machine-learning/08-pca) 
- [Perceptron](07-machine-learning/09-perceptron)

# Papers 

## Deep Neural Networks

- [x]  **DNN** - Learning Internal Representations by Error Propagation (1987), D. E. Rumelhart et al. [[PDF]](https://www.notion.so/Papers-587fcad411304657b7ef990db5299e65?pvs=21)
- [x]  **CNN** - Backpropagation Applied to Handwritten Zip Code Recognition (1989), Y. Lecun et al. [[PDF]](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf)
- [x]  **LeNet** - Gradient-Based Learning Applied to Document Recognition (1998), Y. Lecun et al. [[PDF]](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
- [x]  **AlexNet** - ImageNet Classification with Deep Convolutional Networks (2012), A. Krizhevsky et al. [[PDF]](https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- [x]  **U-Net** - U-Net: Convolutional Networks for Biomedical Image Segmentation (2015), O. Ronneberger et al. [[PDF]](https://arxiv.org/abs/1505.04597)

## Optimization and Regularization

- [x]  **Weight Decay** - A Simple Weight Decay Can Improve Generalization (1991), A. Krogh and J. Hertz [[PDF]](https://proceedings.neurips.cc/paper/1991/file/8eefcfdf5990e441f0fb6f3fad709e21-Paper.pdf)
- [x]  **ReLU** - Deep Sparse Rectified Neural Networks (2011), X. Glorot et al. [[PDF]](https://www.researchgate.net/publication/215616967_Deep_Sparse_Rectifier_Neural_Networks)
- [x]  **Residuals** - Deep Residual Learning for Image Recognition (2015), K. He et al. [[PDF]](https://arxiv.org/pdf/1512.03385)
- [x]  **Dropout** - Dropout: A Simple Way to Prevent Neural Networks from Overfitting (2014), N. Strivastava et al. [[PDF]](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
- [x]  **BatchNorm** - Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (2015), S. Ioffe and C. Szegedy [[PDF]](https://arxiv.org/pdf/1502.03167)
- [x]  **LayerNorm** - Layer Normalization (2016), J. Lei Ba et al. [[PDF]](https://arxiv.org/pdf/1607.06450)
- [x]  **GELU** - Gaussian Error Linear Units (GELUs) (2016), D. Hendrycks and K. Gimpel [[PDF]](https://arxiv.org/pdf/1606.08415)
- [x]  **Adam** - Adam: A Method for Stochastic Optimization (2014), D. P. Kingma and J. Ba [[PDF]](https://arxiv.org/pdf/1412.6980)

## Sequence Modeling

- [ ]  **RNN** - A Learning Algorithm for Continually Running Fully Recurrent Neural Networks (1989), R. J. Williams [[PDF]](https://gwern.net/doc/ai/nn/rnn/1989-williams-2.pdf)
- [ ]  **LSTM** - Long-Short Term Memory (1997), S. Hochreiter and J. Schmidhuber [[PDF]](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [ ]  **Learning to Forget** - Learning to Forget: Continual Prediction with LSTM (2000), F. A. Gers et al. [[PDF]](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=e10f98b86797ebf6c8caea6f54cacbc5a50e8b34)
- [ ]  **Word2Vec** - Efficient Estimation of Word Representations in Vector Space (2013), T. Mikolov et al. [[PDF]](https://arxiv.org/pdf/1301.3781)
- [ ]  **Phrase2Vec** - Distributed Representations of Words and Phrases and their Compositionality (2013), T. Mikolov et al. [[PDF]](https://arxiv.org/pdf/1310.4546)
- [ ]  **Encoder-Decoder** - Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation (2014), K. Cho et al. [[PDF]](https://arxiv.org/pdf/1406.1078)
- [ ]  **Seq2Seq** - Sequence to Sequence Learning with Neural Networks (2014), I. Sutskever et al. [[PDF]](https://arxiv.org/pdf/1409.3215)
- [ ]  **Attention** - Neural Machine Translation by Jointly Learning to Align and Translate (2014), D. Bahdanau et al. [[PDF]](https://arxiv.org/pdf/1409.0473)
- [ ]  **Mixture of Experts** - Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (2017), N. Shazeer et al. [[PDF]](https://arxiv.org/pdf/1701.06538)

## Language Modeling

- [ ]  **Transformer** - Attention Is All You Need (2017), A. Vaswani et al. [[PDF]](https://arxiv.org/pdf/1706.03762)
- [ ]  **BERT** - BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018), J. Devlin et al. [[PDF]](https://arxiv.org/pdf/1810.04805)
- [ ]  **RoBERTa** - RoBERTa: A Robustly Optimized BERT Pretraining Approach (2019), Y. Liu et al. [[PDF]](https://arxiv.org/pdf/1907.11692)
- [ ]  **T5** - Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (2019), C. Raffel et al. [[PDF]](https://arxiv.org/pdf/1910.10683)
- [ ]  **GPT** - Improving Language Understanding by Generative Pre-Training (2018), A. Radford et al. [[PDF]](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [ ]  **GPT-4** - GPT-4 Technical Report (2023), OpenAI [[PDF]](https://arxiv.org/pdf/2303.08774)
- [ ]  **GPT-2** - Language Models are Unsupervised Multitask Learners (2018), A. Radford et al. [[PDF]](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [ ]  **GPT-3** - Language Models are Few-Shot Learners (2020) T. B. Brown et al. [[PDF]](https://arxiv.org/pdf/2005.14165)
- [ ]  **LoRA -** LoRA: Low-Rank Adaptation of Large Language Models (2021), E. J. Hu et al. [[PDF]](https://arxiv.org/pdf/2106.09685)
- [ ]  **RLHF** - Fine-Tuning Language Models From Human Preferences (2019), D. Ziegler et al. [[PDF]](https://arxiv.org/pdf/1909.08593)
- [ ]  **PPO** - Proximal Policy Optimization Algorithms (2017), J. Schulman et al. [[PDF]](https://arxiv.org/pdf/1707.06347)
- [ ]  **InstructGPT** - Training language models to follow instructions with human feedback (2022), L. Ouyang et al. [[PDF]](https://arxiv.org/pdf/2203.02155)
- [ ]  **Helpful & Harmless** - Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback (2022), Y. Bai et al. [[PDF]](https://arxiv.org/pdf/2204.05862)
- [ ]  **Vision Transformer** - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (2020), A. Dosovitskiy et al. [[PDF]](https://arxiv.org/pdf/2010.11929)
- [ ]  **ELECTRA** - ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators (2020), K. Clark et al. [[PDF]](https://arxiv.org/pdf/2003.10555)


## Image Generative Modeling

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
- [ ]  **SimCLR** - A Simple Framework for Contrastive Learning of Visual Representations (2020), T. Chen et al. [[PDF]](https://arxiv.org/pdf/2002.05709)

## Deep Reinforcement Learning

- [ ]  **Deep Reinforcement Learning** - Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (2017), D. Silver et al. [[PDF]](https://arxiv.org/pdf/1712.01815)
- [ ]  **Deep Q-Learning** - Playing Atari with Deep Reinforcement Learning (2013), V. Mnih et al. [[PDF]](https://arxiv.org/pdf/1312.5602)
- [ ]  **AlphaGo** - Mastering the Game of Go with Deep Neural Networks and Tree Search (2016), D. Silver et al. [[PDF]](https://www.nature.com/articles/nature16961)
- [ ]  **AlphaFold** - Highly accurate protein structure prediction with AlphaFold (2021), J. Jumper et al. [[PDF]](https://www.nature.com/articles/s41586-021-03819-2)

## Extraa

- [ ]  **Deep Learning** - Deep Learning (2015), Y. LeCun, Y. Bengio, and G. Hinton [[PDF]](https://www.nature.com/articles/nature14539.pdf)
- [ ]  **GAN** - Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (2016), A. Radford et al. [[PDF]](https://arxiv.org/pdf/1511.06434)
- [ ]  **DCGAN** - Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (2016), A. Radford et al. [[PDF]](https://arxiv.org/pdf/1511.06434)
- [ ]  **BigGAN** - Large Scale GAN Training for High Fidelity Natural Image Synthesis (2018), A. Brock et al. [[PDF]](https://arxiv.org/pdf/1809.11096)
- [ ]  **WaveNet** - WaveNet: A Generative Model for Raw Audio (2016), A. van den Oord et al. [[PDF]](https://arxiv.org/pdf/1609.03499)
- [ ]  **BERTology** - A Survey of BERT Use Cases (2020), R. Rogers et al. [[PDF]](https://arxiv.org/pdf/2002.12327)