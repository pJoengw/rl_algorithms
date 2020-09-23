<div align='center'><font size='70'>Policy Search Survey</font></div>

Policy search methods aim to directly find policies by means
of gradient-free or gradient-based methods. Prior to the current
surge of interest in DRL, several successful methods in DRL
eschewed the commonly used backpropagation algorithm in
favour of evolutionary algorithms [37, 23, 64], which are
gradient-free policy search algorithms. Evolutionary methodsrely on evaluating the performance of a population of agents.
Hence, they are expensive for large populations or agents with
many parameters. However, as black-box optimisation methods they can be used to optimise arbitrary, non-differentiable
models and naturally allow for more exploration in parameter
space. In combination with a compressed representation of
neural network weights, evolutionary algorithms can even be
used to train large networks; such a technique resulted in the
first deep neural network to learn an RL task, straight from
high-dimensional visual inputs [64]. Recent work has reignited
interest in evolutionary methods for RL as they can potentially
be distributed at larger scales than techniques that rely on
gradients [116].  
## Backpropagation through Stochastic Functions
The workhorse of DRL, however, remains backpropagation
[162, 111]. The previously discussed REINFORCE rule [164]
allows neural networks to learn stochastic policies in a taskdependent manner, such as deciding where to look in an
image to track [120], classify [83] or caption objects [166].
In these cases, the stochastic variable would determine the
coordinates of a small crop of the image, and hence reduce
the amount of computation needed. This usage of RL to make
discrete, stochastic decisions over inputs is known in the deep
learning literature as hard attention, and is one of the more
compelling uses of basic policy search methods in recent years,
having many applications outside of traditional RL domains.
More generally, the ability to backpropagate through stochastic
functions, using techniques such as REINFORCE [164] or the
“reparameterisation trick” [61, 108], allows neural networks
to be treated as stochastic computation graphs that can be
optimised over [121], which is a key concept in algorithms
such as stochastic value gradients (SVGs) [46].  
## Compounding Errors
Searching directly for a policy represented by a neural
network with very many parameters can be difficult and can
suffer from severe local minima. One way around this is to
use guided policy search (GPS), which takes a few sequences
of actions from another controller (which could be constructed
using a separate method, such as optimal control). GPS learns
from them by using supervised learning in combination with
importance sampling, which corrects for off-policy samples
[73]. This approach effectively biases the search towards a
good (local) optimum. GPS works in a loop, by optimising
policies to match sampled trajectories, and optimising trajectory distributions to match the policy and minimise costs.
Initially, GPS was used to train neural networks on simulated
continuous RL problems [72], but was later utilised to train
a policy for a real robot based on visual inputs [74]. This
research by Levine et al. [74] showed that it was possible
to train visuomotor policies for a robot “end-to-end”, straight
from the RGB pixels of the camera to motor torques, and,
hence, is one of the seminal works in DRL.  
A more commonly used method is to use a trust region, in
which optimisation steps are restricted to lie within a region
where the approximation of the true cost function still holds.
By preventing updated policies from deviating too wildly
from previous policies, the chance of a catastrophically bad update is lessened, and many algorithms that use trust regions
guarantee or practically result in monotonic improvement in
policy performance. The idea of constraining each policy
gradient update, as measured by the Kullback-Leibler (KL)
divergence between the current and proposed policy, has a long
history in RL [57, 4, 59, 103]. One of the newer algorithms in
this line of work, trust region policy optimisation (TRPO),
has been shown to be relatively robust and applicable to
domains with high-dimensional inputs [122]. To achieve this,
TRPO optimises a surrogate objective function—specifically,
it optimises an (importance sampled) advantage estimate, constrained using a quadratic approximation of the KL divergence.
Whilst TRPO can be used as a pure policy gradient method
with a simple baseline, later work by Schulman et al. [123]
introduced generalised advantage estimation (GAE),which
proposed several, more advanced variance reduction baselines.
The combination of TRPO and GAE remains one of the stateof-the-art RL techniques in continuous control. However, the
constrained optimisation of TRPO requires calculating secondorder gradients, limiting its applicability. In contrast, the
newer proximal policy optimisation (PPO) algorithm performs
unconstrained optimisation, requiring only first-order gradient
information [1, 47, 125]. The two main variants include an
adaptive penalty on the KL divergence, and a heuristic clipped
objective which is independent of the KL divergence [125].
Being less expensive whilst retaining the performance of
TRPO means that PPO (with or without GAE) is gaining
popularity for a range of RL tasks [47, 125].
## Actor-Critic Methods
Instead of utilising the average of several Monte Carlo
returns as the baseline for policy gradient methods, actorcritic approaches have grown in popularity as an effective
means of combining the benefits of policy search methods
with learned value functions, which are able to learn from full
returns and/or TD errors. They can benefit from improvements
in both policy gradient methods, such as GAE [123], and value
function methods, such as target networks [84]. In the last few
years, DRL actor-critic methods have been scaled up from
learning simulated physics tasks [46, 79] to real robotic visual
navigation tasks [167], directly from image pixels.  
One recent development in the context of actor-critic algorithms are deterministic policy gradients (DPGs) [127], which
extend the standard policy gradient theorems for stochastic
policies [164] to deterministic policies. One of the major
advantages of DPGs is that, whilst stochastic policy gradients integrate over both state and action spaces, DPGs only
integrate over the state space, requiring fewer samples in
problems with large action spaces. In the initial work on
DPGs, Silver et al. [127] introduced and demonstrated an
off-policy actor-critic algorithm that vastly improved upon
a stochastic policy gradient equivalent in high-dimensional
continuous control problems. Later work introduced deep DPG
(DDPG), which utilised neural networks to operate on highdimensional, visual state spaces [79]. In the same vein as
DPGs, Heess et al. [46] devised a method for calculating
gradients to optimise stochastic policies, by “reparameterising”
[61, 108] the stochasticity away from the network, thereby allowing standard gradients to be used (instead of the highvariance REINFORCE estimator [164]). The resulting SVG
methods are flexible, and can be used both with (SVG(0) and
SVG(1)) and without (SVG(∞)) value function critics, and
with (SVG(∞) and SVG(1)) and without (SVG(0)) models.
Later work proceeded to integrate DPGs and SVGs with
RNNs, allowing them to solve continuous control problems
in POMDPs, learning directly from pixels [45].  
Value functions introduce a broadly applicable benefit in
actor-critic methods—the ability to use off-policy data. Onpolicy methods can be more stable, whilst off-policy methods
can be more data efficient, and hence there have been several
attempts to merge the two [158, 94, 41, 39, 42]. Earlier
work has either utilised a mix of on-policy and off-policy
gradient updates [158, 94, 39], or used the off-policy data
to train a value function in order to reduce the variance of
on-policy gradient updates [41]. The more recent work by
Gu et al. [42] unified these methods under interpolated policy
gradients (IPGs), resulting in one of the newest state-of-theart continuous DRL algorithms, and also providing insights for
future research in this area. Together, the ideas behind IPGs
and SVGs (of which DPGs can be considered a special case)
form algorithmic approaches for improving learning efficiency
in DRL.  
An orthogonal approach to speeding up learning is to
exploit parallel computation. In particular, methods for training
networks through asynchronous gradient updates have been
developed for use on both single machines [107] and distributed systems [25]. By keeping a canonical set of parameters
that are read by and updated in an asynchronous fashion
by multiple copies of a single network, computation can be
efficiently distributed over both processing cores in a single
CPU, and across CPUs in a cluster of machines. Using a
distributed system, Nair et al. [91] developed a framework
for training multiple DQNs in parallel, achieving both better
performance and a reduction in training time. However, the
simpler asynchronous advantage actor-critic (A3C) algorithm
[85], developed for both single and distributed machine settings, has become one of the most popular DRL techniques
in recent times. A3C combines advantage updates with the
actor-critic formulation, and relies on asynchronously updated
policy and value function networks trained in parallel over
several processing threads. The use of multiple agents, situated
in their own, independent environments, not only stabilises
improvements in the parameters, but conveys an additional
benefit in allowing for more exploration to occur. A3C has
been used as a standard starting point in many subsequent
works, including the work of Zhu et al. [167], who applied it
to robotic navigation in the real world through visual inputs.
For simplicity, the underlying algorithm may be used with
just one agent, termed advantage actor-critic (A2C) [156].
Alternatively, segments from the trajectories of multiple agents
can be collected and processed together in a batch, with
batch processing more efficiently enabled by GPUs; this
synchronous version also goes by the name of A2C [125].  
There have been several major advancements on the original
A3C algorithm that reflect various motivations in the field of
DRL. The first is actor-critic with experience replay [158, 39],which adds Retrace(λ) off-policy bias correction [88] to a
Q-value-based A3C, allowing it to use experience replay in
order to improve sample complexity. Others have attempted to
bridge the gap between value and policy-based RL, utilising
theoretical advancements to improve upon the original A3C
[89, 94, 124]. Finally, there is a growing trend towards exploiting auxiliary tasks to improve the representations learned
by DRL agents, and, hence, improve both the learning speed
and final performance of these agents [77, 54, 82].