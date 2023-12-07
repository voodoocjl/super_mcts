# Q-MCTS
This repository contains the code for paper, Recognizing Good Variational Quantum Circuits with Monte Carlo Tree Search.
```
Dependencies we use in experiments:

numpy == 1.21.5
scikit-learn == 1.1.1
pytorch == 2.0.1
pennylane == 0.30.0

Different package versions may produce inconsistent results.
```
To reproduce the results of the discovered hybrid fusion network which achieves MAE of 1.138 on CMU-MOSI dataset, run ```schemes.py``` in the folder ```best_vqc```.

The file ```Multibench.py``` contains the benchmark from <a href="https://github.com/pliang279/MultiBench/blob/main/examples/Multibench_Example_Usage_Colab.ipynb">MultiBench example</a>, where the dataset is changed to ours for fair comparison.

To start a search for hybrid architectures for multimodal task of sentiment analysis, simply run ```MCTS.py```.