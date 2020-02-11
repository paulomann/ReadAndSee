# ReadOrSee

This project contains code for detection of high depressive symptoms of academic students from Universidade Federal Fluminense (UFF), in Brazil. The study aims to detect depression using social media (Instagram) posts from students using a multimodal approach. In other words, we use both images and captions of Instagram posts to create Machine Learning models capable of detecting students with high depressive symptoms.

We have sent many invitations through official social media pages of UFF, students' groups on Facebook, and UFF email lists, where students could participate voluntarily. The invitation process, questionnaire, and methodology were under the approval of the UFF ethical committee, with number CAAE:89859418.1.0000.5243.

It's important to note that we **do not** claim that we *predict* depression. This subtle difference is significant because of many psychologists who find problematic to state that a patient *is* depressed or not. They majorly treat their patients with a continuous diagnosis, without labeling them. Here, in this study, we focus on uncovering students (possibly any person) conditions, where we can further guide them to have adequate treatment.

# Paper

The publication is available on [arXiv](https://arxiv.org/abs/1912.01131) ("See and Read: Detecting Depression Symptoms in Higher Education Students Using Multimodal Social Media Data"). The paper was accepted on the 15 November 2019 and will appear in the proceedings of [ICWSM 2020](https://www.icwsm.org/2020)

# How to use the code

Unfortunately, we can not provide access to datasets due to ethical constraints; but all neural networks weights and code will be made available. However, we can give access to embeddings for both images and texts if there are interested researchers. Just get in touch with me!

Install the module in the root repository with `pip install -e .`, and then you can use any function/class with the following code:

```Python
from readorsee import *
```
For any doubts or concerns, get in touch with me through my email, creating an issue here (or anything easier to you).
