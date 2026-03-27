# Predicting Ground-level Ozone Concentrations Using Deep Learning Networks  
This repository contains Python code designed to evaluate six prominent deep learning (DL) architectures for predicting ground-level ozone concentrations. The architectures under examination include the Fully Connected Network (FCN), commonly referred to as the Multi-Layer Perceptron (MLP), and several variants Recurrent Neural Networks (RNNs), specifically the Long Short-Term Memory (LSTM) and Bidirectional LSTM (Bi-LSTM) models. Additionally, the  Convolutional Neural Network (CNN) and a Transformer-based network are also involved. Moreover, to provide a comprehensive comparison, a conventional Machine Learning (ML) model—LightGBM is included. This allows for an evaluation of how DL approaches stack up against established ML methods, offering a robust framework for assessing the effectiveness of various architectures in predicting ozone concentrations. The implementation of DL models is based on Keras (version 2.6.0) with Tensorflow backend (version 2.6.0).
## Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Raghava-SPLTech-Git/DeepLearningApproachesForOzonePrediction.git
   cd DeepLearningApproachesForOzonePrediction
   ```

2. **Create a virtual environment (`.venv`):**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment:**
   - On **Windows**:
     ```cmd
     .\.venv\Scripts\activate.bat
     ```
   - On **macOS/Linux**:
     ```bash
     source .venv/bin/activate
     ```

4. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Run The Code
After setting up the environment, you can run the model scripts. For example:
```
python OzonePrediction_2-Layer_Conv1d.py
```
For the complete dataset, please contact yfchi@fjsmu.edu.cn.
