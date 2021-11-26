# Toward feasible machine learning model updates in network-based intrusion detection

#### Authors: Pedro Horchulhack and Eduardo K. Viegas and Altair O. Santin
Over the last years, several works have proposed highly accurate machine learning (ML) techniques for network-based intrusion detection systems (NIDS), that are hardly used in production environments. In practice, current intrusion detection schemes cannot easily handle network traffic’s changing behavior over time, requiring frequent and complex model updates to be periodically performed. As a result, unfeasible amounts of labeled training data must be provided for model updates as time passes, making such proposals unfeasible for the real world. This paper proposes a new intrusion detection model based on stream learning with delayed model updates to make the model update task feasible with a twofold implementation. First, our model maintains the intrusion detection accuracy through a classification assessment approach, even with outdated underlying ML models. The classification with a reject option rationale also allows suppressing potential misclassifications caused by new network traffic behavior. Second, the rejected instances are stored for long periods and used for incremental model updates. As an insight, old rejected instances can be easily labeled through publicly available attack repositories without human assistance. Experiments conducted in a novel dataset containing a year of real network traffic with over 2.6 TB of data have shown that current techniques for intrusion detection cannot cope with the network traffic’s evolving behavior, significantly degrading their accuracy over time if no model updates are performed. In contrast, the proposed model can maintain its classification accuracy for long periods without model updates, even improving the false-positive rates by up to 12% while rejecting only 8% of the instances. If periodic model updates are conducted, our proposal can improve the detection accuracy by up to 6% while rejecting only 2% of network events. In addition, the proposed model can perform model updates without human assistance, waiting up to 3 months for the proper event label to be provided without impact on accuracy, while demanding only 3.2% of the computational time and 2% of new instances to be labeled as time passes, making model updates in NIDS a feasible task.

### Dataset
- [MAWILab Dataset](https://secplab.ppgia.pucpr.br/?q=idsovertime)

### BibTeX
```bibtex
@article{HORCHULHACK2021108618,
    title = {Toward feasible machine learning model updates in network-based intrusion detection},
    journal = {Computer Networks},
    pages = {108618},
    year = {2021},
    issn = {1389-1286},
    doi = {https://doi.org/10.1016/j.comnet.2021.108618},
    url = {https://www.sciencedirect.com/science/article/pii/S1389128621005120},
    author = {Pedro Horchulhack and Eduardo K. Viegas and Altair O. Santin},
    keywords = {Intrusion detection, Stream learning, Reject option},
}
```
