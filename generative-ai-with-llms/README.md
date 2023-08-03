https://www.coursera.org/learn/generative-ai-with-llms

# Introduction to LLMs and the generative AI project lifecycle

## Contributors

**Ehsan Kamalinejad, Ph.D.** is a Machine Learning Applied Scientist working on NLP developments at Amazon. Previously he co-founded Visual One, a YCombinator startup in computer vision. Before that, he was a Tech-Lead Machine Learning Engineer at Apple, working on projects such as Memories. EK is also an Associate Professor of Mathematics at California State University East Bay.

**Nashlie Sephus, Ph.D.** is a Principal Technology Evangelist for Amazon AI at AWS. In this role, she focuses on fairness and accuracy as well as identifying and mitigating potential biases in artificial intelligence. She formerly led the Amazon Visual Search team as an Applied Scientist in Atlanta which launched the visual search for replacement parts on the Amazon Shopping app.

**Heiko Hotz** is a Senior Solutions Architect for AI & Machine Learning with a special focus on natural language processing (NLP), large language models (LLMs), and generative AI. Prior to this role, he was the Head of Data Science for Amazon’s EU Customer Service. Heiko helps companies be successful in their AI/ML journey on AWS and has worked with organizations in many industries, including insurance, financial services, media and entertainment, healthcare, utilities, and manufacturing. In his spare time, Heiko travels as much as possible.

**Philipp Schmid** is a Technical Lead at Hugging Face with the mission to democratize good machine learning through open source and open science. Philipp is passionate about productionizing cutting-edge and generative AI machine learning models.


## Introduction

Primeira pergunta: eu devo pegar um modelo e fazer um zero shot ou preciso treinar? Qual modelo eu posso usar?

## Use cases & model lifecycle

Principais modelos de geração de linguagem com seus tamanhos comparativos:

![LLMs comparative sizes](images/llm_sizes.png)

Entrada de um modelo: prompt

Modelo tenta prever a continuação do prompt. Caso a entrada seja uma pergunta, a saída será uma resposta

Casos de uso

- Q&A
- Sumarização
- Tradução
- Geração de código
- Extração de entidades
- _Augmented LLMs_: conexão com outras bases e APIs para prover dados não fornecidos no pré treino. Exemplo: perguntar a um modelo se um voo está atrasado ou não

### Text generation before transformers

Modelos usando RNNs escalam exponencialmente em escala para compreender mais tokens de uma amostra ao mesmo tempo (uma janela maior de entrada)

O idioma é complexo e ambíguo, compalavras homônimas e afins

**Attention Is All You Need** foi o primeiro artigo explicando o mecanismo de atenção dos **Transformers**

### Transformers Architecture

The transformer has an mecanism of self-attention that allows it to learn the relevance of each word to each other word, no matter the position in the phrase

![Self-attention mecanism](images/self-attention.png)

The simplified model of the Transformer can be seen below

![Simplified transformers](images/simplified-transformers.png)

Before entering the Transformer, the words are tokenized through the tokenizer, that can tokenize both words or parts of the word

![Alt text](images/tokenizer-parts.png)

The **embedding** is a high dimentional vectorizer, where each word token will be replaced by a vector. The embedding learns how to vectorize each word representing the meaning and relationships with each other word

After the embedding, there is a **positional encoding**

Inside the Encoder and Decoder, there is a **multi-headed self-attention** layer, making each word visible to each other. It happens in parallel, independently

- One head can learn activity
- Other can learn rhyme and so on

At the end of the encoder and decoder, there is a **feed foward network**, witch generates the probability score for each possible word

At the end a **softmax output** chooses one word and also sends it back to the **embedding input of the decoder** branch

## Generating text with transformers

The output of the **encoder** is a deep representation of meaning and sequence of each word in the phrase used to influentiate the self-attention mecanism of the **decoder**

**Encoder Only Models** are used to model generation only and classification tasks. Example: **BERT**

**Encoder Decoder Models** performs well on Seq2Seq. These models can also be trained to general text generation. Example: BART

**Decoder Only Models** can be generalized to a bunch of tasks. Example: BLOOM, LLAMA and so on

## Generative configuration

Notes on [Generating Text](../wild-notes/generating-text.md)

In summary:
- Top k controls the number of candidates by the absolute number of considered candidates $k$. "Just uses the k-est candidates"
- Top p controls the number of candidates by adding their probabilities and maintaining these below the choosen $p$. "Considers just the top candidates that the sum of their probabilites are just below $p$"
- Temperature

## Project lifecycle

![Project lifecycle](images/llm-project-lifecycle.png)