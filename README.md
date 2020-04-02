# Vector Quantized Contrastive Predictive Coding for Template-based Music Generation
Gaëtan Hadjeres, Sony CSL, Paris, France (gaetan.hadjeres@sony.com)\
Léopold Crestel, Sony CSL, Paris, France (leopold.crestel@sony.com)

This is the companion github of the paper 
[Vector Quantized Contrastive Predictive Coding for Template-based Music Generation](LIEN SITE).
In this paper, we proposed a flexible method for generating variations of discrete sequences 
in which tokens can be grouped into basic units, like sentences in a text or bars in music.
More precisely, given a template sequence, we aim at producing novel sequences sharing perceptible similarities 
with the original template without relying on any annotation.
We introduce 
 - a *self-supervised encoding* technique, named *Vector Quantized Contrastive Predictive Coding* (*VQCPC*), 
which allows to learn a meaningful assignment of the basic units over a discrete set of codes,
together with  mechanisms allowing to control the information content of these learnt discrete representations.
- a *decoder* architecture which can generate sequences from the compressed representations learned by the encoder.
In particular, it can be used to generate variations of a template sequence.
 
Our experiments on the corpus of J.S. Bach chorales can be reproduced using this repository. 


## Installation


## Experiments
Parameters
- Negative sampling scheme + student
- Quantization bottleneck
- Type of downsampling

Decoding is done with a relative transformer (**DESCRIBED IN PAPER**)

**ON MET DES SAMPLES DANS LES CASES**

|       |  LSTM downscaler | Transformer downscaler  
| :--- |:---:| :---:
| Random      | |
| Same sequence | |
| Student | X |
| Random and no quantization | |
| Same sequence and no quantization |  |
| Student and no quantization| X | 

Quoi montrer
- clusters
    - 
- petits fragments original et variations
- rewritting of a whole chorale
