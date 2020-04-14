<!--
<script src="http://vjs.zencdn.net/4.0/video.js"></script>
-->

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS
-MML_HTMLorMML" type="text/javascript"></script>

<script type="text/javascript"> 
      // Show button
      function look(type){ 
      param=document.getElementById(type); 
      if(param.style.display == "none") param.style.display = "block"; 
      else param.style.display = "none" 
      } 
</script> 

This is the companion website of the paper 
[Vector Quantized Contrastive Predictive Coding for Template-based Music Generation](www.google.com).
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
 
We applied our technique on the corpus of J.S. Bach chorales to derive a generative model with high-level controls.
In particular, it is particularly well-suited for generating variations of a given input chorale.
Our experiments can be reproduced using the following repository: [https://github.com/SonyCSLParis/vqcpc-bach](https://github.com/SonyCSLParis/vqcpc-bac)

The results of our experiments are presented in the following sections
  * [Clusters](#clusters)
  * [Examples in the paper](#examples-in-the-paper)
  * [Variations of a source piece](#variations-of-a-source-piece)
  
## Clusters
The encoder learns to map atomic structuring elements of a time-series to a label belonging to a discrete alphabet.
In other words, **an encoder defines a clustering of the space formed by structuring elements**.
This clustering is learned in a self-supervised manner, by optimising a contrastive objective.

In our experiment, we considered Bach chorales.
We chose to define a structuring element as **one beat of a four voices chorale**. 
Note that there is no restriction in using fixed length structuring elements, 
and variable lengths could be used in other applications such as natural language processing.

In the following animated pictures, **each frame represents one cluster**,
and **each bar represents one structuring element** belonging to that cluster. 
A limited number of clusters and elements are diplayed on this site. 
More examples can be downloaded here [clusters.zip](exemples/clusters/clusters.zip).

In our article, we explored three different self-supervised training objectives:
*VQCPC* with random negative sampling,
*VQCPC* with same sequence negative sampling,
and Distilled *VQ-VAE*.
Each of them led to a different type of clustering which we display below: 

### *VQCPC* with random negative sampling
The negative examples in the contrastive objective are sampled randomly among all chorales.
Since chorales have been written in all possible key signatures and we used transposition as a data augmentation,
an easy way to discriminate the positive examples from the negatives is to look at the alterations.
Hence, the clusters are often composed by elements which can lie in the same or a related key.

<img class="recimg" src="exemples/clusters/clusters_random.gif">

### *VQCPC* with same sequence negative sampling
The negative examples in the contrastive objective are sampled in the same-sequence as the positive example, 
but at different location (either before or after the position of the positive).
In that case, the key is no longer a discriminative feature of the positive example. 
On the contrary, the harmonic function is an informative indicator of the position of a chord in a phrase.
Hence, clusters tend to contain elements which could share similar harmonic functions. 
 
<img class="recimg" src="exemples/clusters/clusters_sameSeq.gif">

### Distilled *VQ-VAE*

  
## Examples in the paper

## Variations of a source piece
<table>
<caption><b> Source </b></caption>
  <tr>
    <td style="text-align: center; vertical-align: middle;"><b>Score</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>Audio Rendering</b></td>
  </tr>
  
  <tr>
    <td><img class="recimg" src="https://anonymous0505.github.io/VQCPC/figures/..."></td>
    <td style="text-align: center; vertical-align: middle;">
      <audio controls>
      <source src="https://anonymous0505.github.io/VQCPC/sounds/...">
      </audio>
    </td>
  </tr>
</table>


<table>
<caption><b> Variations with method 1 </b></caption>
  <tr>
    <td style="text-align: center; vertical-align: middle;"><b>Score</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>Audio Rendering</b></td>
  </tr>
  
  <tr>
    <td><img class="recimg" src="https://anonymous0505.github.io/VQCPC/figures/..."></td>
    <td style="text-align: center; vertical-align: middle;">
      <audio controls>
      <source src="https://anonymous0505.github.io/VQCPC/sounds/...">
      </audio>
    </td>
  </tr>
  
  <tr>
    <td><img class="recimg" src="https://anonymous0505.github.io/VQCPC/figures/..."></td>
    <td style="text-align: center; vertical-align: middle;">
      <audio controls>
      <source src="https://anonymous0505.github.io/VQCPC/sounds/...">
      </audio>
    </td>
  </tr>

  <tr>
    <td><img class="recimg" src="https://anonymous0505.github.io/VQCPC/figures/..."></td>
    <td style="text-align: center; vertical-align: middle;">
      <audio controls>
      <source src="https://anonymous0505.github.io/VQCPC/sounds/...">
      </audio>
    </td>
  </tr>
</table>




