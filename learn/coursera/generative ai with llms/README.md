### Week 1

Generative AI Foundational Models: GPT, LLaMa, BERT, PaLM, BLOOM, FLAN-T5.

![foundational models](./images/foundational_models.png)

The more parameters, the more memory and language understanding and the more sophisticated tasks it can perform.

Prompt (Context Window) -> Inference -> Completion

**Next word prediction** can be used for summarization, translation, coding, information retrieval, named entity recognition...

You don't always need a large model, smaller models can be fine-tuned to specific focused tasks. 

Previous generation models used Recurrent Neural Networks (RNNs). RNNs while powerful for their time, were limited by the amount of compute and memory needed to perform well at generative tasks.

**Transformer architecture** started with paper "Attention is all you need", in 2017. This novel approach unlocked the progress in generative AI that we see today. It can be scaled efficiently to use multi-core GPUs, it can parallel process input data, making use of much larger training datasets, and crucially, it's able to learn to pay attention to the meaning of the words it's processing.

### Transformers

The power of the transformer architecture lies in its ability to learn the relevance and context of all of the words in a sentence. Not just to each word next to its neighbor, but to every other word in a sentence. To apply attention weights to those relationships so that the model learns the relevance of each word to each other words no matter where they are in the input. 

![self-attention](./images/self_attention.png)

![complete transformer architecture](./images/complete_transformer_architecture.png)

![transformer architecture](./images/transformer_architecture.png)

Words are tokenized before fed to the model since a Machine Learning model works with numbers and not words. A token can represent the whole word or a part of the word. You then have to use the same tokenizer to train and predict/generate. The token is then embedded, mapped to a multidimensional vector, that encodes the meaning and context of individual tokens in the input sequence. Positional embeddings are also added to preserve the ordering of words in the input sentence.

![word embeddings](./images/word_embeddings.png)

![positional embeddings](./images/positional_embeddings.png)

The embeddings are passed into the self-attention layers, which learn the importance of every word to every other word in the input sequence. It has **Multi-headed self-attention**. This means that multiple sets of self-attention weights or heads are learned in parallel independently of each other. The number of attention heads included in the attention layer varies from model to model, but numbers in the range of 12-100 are common. The intuition here is that each self-attention head will learn a different aspect of language. For example, one head may see the relationship between the people entities in our sentence. Whilst another head may focus on the activity of the sentence. Whilst yet another head may focus on some other properties such as if the words rhyme. We don't dictate ahead of time which aspect of language each head will learn, the weights of each head are randomly initialized.

![multi-headed self-attention](./images/multiheaded_self_attention.png)

The output is then processed through a fully-connected feed-forward network. The output of this layer is a vector of logits proportional to the probability score for each and every token in the tokenizer dictionary. You can then pass these logits to a final softmax layer, where they are normalized into a probability score for each word.
One single token will have a score higher than the rest. This is the most likely predicted token.

![feed forward network](./images/feed_forward_network.png)

Example: translation

![translation task](./images/transformer_task.png)

Encoder: encodes inputs ("prompts") with contextual understanding and produces one vector per input token.

Decoder: Accepts input tokens and generates new tokens.

There are encoder only models, encoder decoders models and decoder only models. Decoder only models are the most popular today.

**"Attention is All You Need"**

"Attention is All You Need" is a research paper published in 2017 by Google researchers, which introduced the Transformer model, a novel architecture that revolutionized the field of natural language processing (NLP) and became the basis for the LLMs we  now know - such as GPT, PaLM and others. The paper proposes a neural network architecture that replaces traditional recurrent neural networks (RNNs) and convolutional neural networks (CNNs) with an entirely attention-based mechanism. 

The Transformer model uses self-attention to compute representations of input sequences, which allows it to capture long-term dependencies and parallelize computation effectively. The authors demonstrate that their model achieves state-of-the-art performance on several machine translation tasks and outperforms previous models that rely on RNNs or CNNs.

The Transformer architecture consists of an encoder and a decoder, each of which is composed of several layers. Each layer consists of two sub-layers: a multi-head self-attention mechanism and a feed-forward neural network. The multi-head self-attention mechanism allows the model to attend to different parts of the input sequence, while the feed-forward network applies a point-wise fully connected layer to each position separately and identically. 

The Transformer model also uses residual connections and layer normalization to facilitate training and prevent overfitting. In addition, the authors introduce a positional encoding scheme that encodes the position of each token in the input sequence, enabling the model to capture the order of the sequence without the need for recurrent or convolutional operations.

**Prompt engineering**

- **In-context learning** - Include examples in the context window to help the model learn the task. Zero / one / few shot inference. If more than 5-6 examples don't increase performance, fine-tuning should be preferred. 

The larger the model, the more performant it is at zero shot inference. Smaller models are generally good at a few tasks, similar to the ones they were trained on.

![](./images/in_context_learning.png)

Inference configuration parameters - temperature, max new tokens, topK, topP.

Greedy vs random sampling - greedy will always choose the word with the highest probability but words can be repeated and the answer gets less natural and less reasonable so this method is good for short generation. Random sampling will select based on random-weighted strategy across all probabilities so this generates more natural and creative text.

topK selects an output using the top-K results after applying random-weighted strategy using the probabilities. 

topP selects using the random-weighted strategy with the top-ranked consecutive results by probability and with a cumulative probability <= p.

Temperature influences the shape of the probability distribution. The higher the temperature, the higher the randomness. The temperature actually alters the predictions the model will make, unlike topK or topP.

![temperature configuration](./images/temperature_config.png)
![AI project lifecycle](./images/project_lifecycle.png)

LLM pre-training: learning from large amounts of unstructured textual data, learning patterns and structures in the language. Involves a lot of compute and GPUs. Often needed are data filtering, to improve quality, reduce bias and remove harmful content. Only 1-3% of the original tokens are kept for training!

![LLM pre-training](./images/llm_pretraining.png)

Encoder only models are trained on masked language modeling tasks, where some randomly chosen tokens are masked and the model has to predict them. These models build bidirectional representations of the input sequence, understanding the full context of each token in the sequence. They are good for tasks that benefit from understanding the full context of the input (bidirectional context): sentiment analysis, named entity recognition, word classification. e.g. BERT, ROBERTA.

![encoder only models](./images/autoencoding_models.png)

Decoder only models are trained on causal language modeling tasks, where the model has to predict the next token in the sequence. The context used is unidirectional, only previous tokens are seen by the model during training. These models are good for tasks like text generation. e.g. GPT, BLOOM.

![decoder only models](./images/autoregressive_models.png)

The exact details of the pre-training objective of Encoder-decoder models vary from model to model. A popular sequence-to-sequence model T5, pre-trains based on span corruption tasks. These models are good for translation, summarization, question answering. e.g. BART, T5.

![encoder-decoder models](./images/sequence_to_sequence_models.png)

![all models](./images/all_models.png)

In general, the more parameters, the more capable a model is at any given task. It may be infeasible to just keep increasing and training larger and larger models, given the cost, though.

![model evolution](./images/model_evolution.png)

Computational challenges: CUDA runs out of memory. e.g. 24GB of GPU RAM to train a 1B parameter model @ 32-bit precision. This is to store the model weights, activations, gradients, optimizer states, etc.

![memory to train models](./images/memory_to_train.png)

To address this, 16-bit precision can be used (quantization), which lowers precision but is acceptable for most cases.

![quantization](./images/quantization.png)

BFLOAT16 is another option, used for newer deep learning models, like flan-T5. It significantly helps with pre-training stability. It's known as truncated 32-bit float. Not well suited for integer calculations but these are relatively rare in deep learning calculations.

![bfloat16](./images/bfloat16_quantization.png)

INT8 will incur a huge precision loss. 

![int8](./images/int8.png)

Quantization aware training learns the quantization scaling factors during training.

![quantization summary](./images/quantization_summary.png)

![quantization savings](./images/quantization_savings.png)

As models scale, multiple GPUs for training are needed. 

![super large models](./images/super_large_models.png)

Data parallelism is a strategy that splits the training data across multiple GPUs. Each GPU processes a different subset of the data simultaneously, which can greatly speed up the overall training time.

Distribute Data Parallel from pytorch - Distribute datasets across GPUs which train a copy of the model, and then synchronization step combines the results. Requires that the model fits in a single GPU.

![ddp](./images/ddp.png)

Fully sharded data parallel - Zero redundancy optimizer distributes models across GPUs. This is suitable for models that cannot fit in a single GPU.

![ZeRO](./images/zero.png)

![FSDP](./images/fsdp.png)

Researchers have explored the trade-offs between training dataset size, model size and compute budget. 

![scaling models](./images/scaling_models.png)

1 petaflop/s-day is approximately equivalent to 8 NVIDIA V100 chips operating at full efficiency for one day, in the context of training transformers. This is equivalent to 2 NVIDIA A100 GPUs. Huge amounts of compute are required to train the bigger models...


![compute_budget](./images/compute_budget.png)

![compute requirements](./images/compute_requirements.png)

![scaling laws](./images/scaling_laws.png)

Chinchilla paper suggests that some large models may be over parametrized and under trained. So smaller models are starting to come up, trained more optimally.

![chinchilla](./images/chinchilla.png)

Pre-training for domain adaptation - when very specific language are common but unlikely to have come up extensively in the training dataset. e.g. BloombergGPT

BloombergGPT, developed by Bloomberg, is a large Decoder-only language model. It underwent pre-training using an extensive financial dataset comprising news articles, reports, and market data, to increase its understanding of finance and enabling it to generate finance-related natural language text. The datasets are shown in the image above.

During the training of BloombergGPT, the authors used the Chinchilla Scaling Laws to guide the number of parameters in the model and the volume of training data, measured in tokens. The recommendations of Chinchilla are represented by the lines Chinchilla-1, Chinchilla-2 and Chinchilla-3 in the image, and we can see that BloombergGPT is close to it. 

While the recommended configuration for the team’s available training compute budget was 50 billion parameters and 1.4 trillion tokens, acquiring 1.4 trillion tokens of training data in the finance domain proved challenging. Consequently, they constructed a dataset containing just 700 billion tokens, less than the compute-optimal value. Furthermore, due to early stopping, the training process terminated after processing 569 billion tokens.

The BloombergGPT project is a good illustration of pre-training a model for increased domain-specificity, and the challenges that may force trade-offs against compute-optimal model and training configurations.

![bloombergGPT](./images/bloombergGPT.png)

In contrast to pre-training, where you train the LLM using vast amounts of unstructured textual data via self-supervised learning, fine-tuning is a supervised learning process where you use a data set of labeled examples to update the weights of the LLM. The labeled examples are prompt completion pairs, the fine-tuning process extends the training of the model to improve its ability to generate good completions for a specific task.

**Instruction fine-tuning** trains the model using examples that demonstrate how it should respond to a specific instruction.

![fine tuning](./images/fine_tuning.png)

![instruction fine tuning](./images/instruction_fine_tuning.png)

![instruction fine tuning process](./images/instruction_fine_tuning_process.png)

You can fine-tune a pre-trained model to improve performance on only the task that is of interest to you. For example, summarization using a dataset of examples for that task. Interestingly, good results can be achieved with relatively few examples. Often just 500-1,000 examples can result in good performance in contrast to the billions of pieces of texts that the model saw during pre-training. However, there is a potential downside to fine-tuning on a single task. The process may lead to a phenomenon called **catastrophic forgetting**. Catastrophic forgetting happens because the full fine-tuning process modifies the weights of the original LLM. While this leads to great performance on the single fine-tuning task, it can degrade performance on other tasks. If you do want or need the model to maintain its multitask generalized capabilities, you can perform fine-tuning on multiple tasks at one time. Good multitask fine-tuning may require 50-100,000 examples across many tasks, and so will require more data and compute to train. Our second option is to perform parameter efficient fine-tuning, or PEFT for short instead of full fine-tuning. PEFT is a set of techniques that preserves the weights of the original LLM and trains only a small number of task-specific adapter layers and parameters. PEFT shows greater robustness to catastrophic forgetting since most of the pre-trained weights are left unchanged.

One way to mitigate catastrophic forgetting is by using regularization techniques to limit the amount of change that can be made to the weights of the model during training. This can help to preserve the information learned during earlier training phases and prevent overfitting to the new data.

![single task fine tuning](./images/single_task_fine_tuning.png)

