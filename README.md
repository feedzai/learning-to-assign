# Learning to Assign (L2A)

> **Note**
> The paper related to this work *"Learning to Assign in Human-AI Collaboration Under Fairness and Capacity Constraints"* is currently under review in KDD 2023. Link to the paper [here](https://openreview.net/pdf?id=c0zxqaXvIW).

## Abstract
Human-AI collaboration has been proposed as an alternative to fully automated decision-making, with the objective of improving overall system performance by exploiting the complementary strengths of humans and AI. Simultaneously, due to ethical concerns, human intervention in high-stakes decision-making systems is becoming more prevalent. Nevertheless, the state-of-the-art method to manage assignments in these systems — *Learning to Defer* (L2D) — presents structural limitations that inhibit its adoption in real-world scenarios: L2D requires concurrent human predictions for every instance of the dataset in training, and, in its current formulation, is incapable of dealing with human capacity constraints. This is further aggravated by a lack of well-established realistic benchmarks for this setting. In this work, we propose *Learning to Assign* (L2A), a novel method for performing assignments under capacity and fairness constraints. L2A leverages supervised learning to model the probability of error of humans without requiring extra data, and employs linear programming to globally minimize a cost-sensitive loss subject to capacity constraints. Additionally, we create a new public dataset for testing learning to assign in a fraud detection setting with realistic capacity constraints on a team of 50 synthetic fraud analysts. We show that L2A significantly improves performance over the baselines and that it is effective at mitigating unfairness in the decisions of the human-AI system.
## Installation

To install the repository, simply run with pip in an appropriate environment (python >= 3.7).

```bash
$ pip install . 
```

