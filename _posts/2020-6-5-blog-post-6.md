---
title: 'Learning with Invariance using Tangent Distance Kernel'
date: 2020-06-05
permalink: /posts/2020/06/blog-post-6/
tags:
  - support vector machines
  - kernel trick
  - machine learning
---

In kernel SVM, most kernel functions relies on some distance metric. A commonly used distance metric is the Euclidean distance. However in fields like pattern recognition, Euclidean distance is not robust to simple transformations that don't change the class label: rotation, translation and scaling. For example:

<br/><img src='/images/blog_post_images/euclidean_dist_nonrobust.png'>

According to Euclidean distance B is closer than A. A better distance metric would argue that A is closer than B. Distance metrics that are robust to invariances that don't alter class labels are extremely important to distance-based classifiers like nearest neighbor classifier and support vector machines.

When pattern $P$ is transformed with a transformation $s$ that depends on a parameter $\alpha$ (e.g. rotation of the angle, step of the translation, of the magnitude of the scaling), the set of all transformed points $S_P = \{ x \vert \forall \vec{\alpha} such that x = s(\vec{\alpha}, P)\}$ where $s(\vec{\alpha}, P)$ defines a differentiable tranformation, form a manifold in vector space of the inputs. Note that $s(0, x) = x$.

Therefore, an invariant distance metric should measure the minimum distance between the two manifolds formed by their respective set of all transformed points $S_P$. However, the minimum distance between two manifolds is hard to calculate, therefore we will approximate that distance using the tangent distance. To summerize: we will approximate the distance betwenn two manifolds induced by point $x$ and $x'$ using x's distance to the tangent line of the manifold induced by $x'$ at x'. To illustrate this in picture:

<br/><img src='/images/blog_post_images/tangent_dist.png'>

The dash line is the tangent distance. The illustrated distance, called a one sided tangent distance, is obtained through the following optimization problem:

$$d_{1S}(x, x') := \min_{\vec{\alpha}} \lVert x+\sum_{i=1}^l \alpha_i L_i - x' \rVert$$

where $L$ is the tangent line of the manifold induced by $x$ at point $x$ and is defined mathematically as: $L = \left.\frac{\partial s(\vec{\alpha}, x)}{\partial \vec{\alpha}}\right \vert_{\vec{\alpha}=0}$


















