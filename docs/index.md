<!--
<script src="http://vjs.zencdn.net/4.0/video.js"></script>
-->

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<script type="text/javascript"> 
      // Show button
      function look(type){ 
      param=document.getElementById(type); 
      if(param.style.display == "none") param.style.display = "block"; 
      else param.style.display = "none" 
      } 
</script> 

# Vector Quantized Contrastive Predictive Coding for Template-based Music

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
 
Our experiments on the corpus of J.S. Bach chorales can be reproduced using this repository.

Here, we directly embed the exposed elements
  * [Clusters](#clusters)
    * [Random VQCPC](#Random VQCPC)
    * [Same-sequence VQCPC](#Same-sequence VQCPC)
    * [Student](#Student)
  * [Examples in the paper](#examples-in-the-paper)
  * [Variations of a source piece](#variations-of-a-source-piece)
  * [Code](#code)
  
## Clusters
Few interesting clusters obtained with different models.
### Random VQCPC
<table>
  <tr>
    <td style="text-align: center; vertical-align: middle;"><b>Score</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>Audio Rendering</b></td>
  </tr>
  
  <tr>
    <td><img class="recimg" src="exemples/test.png"></td>
    <td style="text-align: center; vertical-align: middle;">
      <audio controls>
      <source src="exemples/test.wav">
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

### Same-sequence

  
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




