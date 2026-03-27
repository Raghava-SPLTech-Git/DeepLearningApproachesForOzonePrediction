# Ozone Prediction Deep Learning Framework: Project Overview

## 🎯 **Goal of the Project**

### 1. The Real-World Problem
Ground-level Ozone ($O_3$) is a harmful air pollutant. Predicting exactly how much ozone will be in the air on any given day is extremely difficult because it depends on a chaotic mix of hundreds of different things happening at once. 

The primary goal of this repository is to build AI models capable of looking at huge amounts of environmental data—106 individual factors like weather (temperature, wind), geography (elevation, land use), human activity (population density), and other pollutants—and instantly **predicting the exact concentration of ozone in the air**. 

### 2. The Science "Tournament"
Because predicting atmospheric pollution is incredibly complex, scientists still don't agree on exactly which type of Artificial Intelligence is "best" at solving this specific problem. 

To solve this, this project isn't just one program—it is designed as a **scientific tournament**. 

The secondary goal is to let **seven entirely different AI architectures compete against each other** on the exact same dataset to definitively prove which one handles environmental prediction the best. 

Instead of relying on just a single AI perspective, this framework lines up:
- **Traditional Machine Learning** (LightGBM) as the baseline control.
- **Deep Neural Networks** (FCN) to simulate raw mathematical complexity.
- **Spatial Networks** (CNN) to look for local patterns in the weather data as if studying an image.
- **Memory-based Networks** (LSTM & Bi-LSTM) to analyze how the air pollution fluctuates sequentially "over time".
- **Transformer Networks** (the technology powering modern LLMs) to simply direct its advanced "Attention" at whatever atmospheric variables it thinks matter the most.

By forcing all these vastly different models to solve the exact same dataset "homework" (measuring their Mean Absolute Error and $R^2$ accuracy scores cleanly), the project mathematically declares exactly which architecture is superior for real-world environmental tracking!

---

## 🗂️ **Directory Map & Model Significance**
Every `.py` file enclosed in the repository represents a distinct neural or machine-learning methodology being tested. Here is a rundown of each implementation script and why it exists:

### 1. `OzonePrediction_7-Layer_DNN.py` *(The FCN/MLP Benchmark)*
- **Architecture**: A very deep **7-Layer Fully Connected Network** (Dense Neural Network). 
- **Significance**: This acts as the standard Deep Learning baseline. It treats all 106 input features independently without any built-in logic for recognizing time sequences or spatial dependencies, proving whether sheer mathematical "depth" is enough to learn Ozone prediction.

### 2. `OzonePrediction_2-Layer_Conv1d.py` *(The Spatial Feature Model)*
- **Architecture**: A 2-Layer **1D Convolutional Neural Network (CNN)**.
- **Significance**: Normally used for images, this applies an overlapping sliding window across the 1D array of environmental features. It attempts to learn localized, interconnected spatial patterns across adjacent atmospheric inputs to see if features grouped near each other imply a heavier influence on Ozone levels.

### 3. `OzonePrediction_2-layer-LSTM.py` *(The Temporal Model)*
- **Architecture**: A 2-Layer **Long Short-Term Memory** (LSTM) network.
- **Significance**: Unlike static models, recurrent networks like LSTM contain internal "memory". This forces the dataset predictions to take sequential context (time-steps/history) into account, preventing it from forgetting important long-term trend data behind the ozone fluctuations.

### 4. `OzonePrediction_2-layer-Bi-LSTM.py` *(The Bidirectional Temporal Model)*
- **Architecture**: A 2-Layer **Bidirectional LSTM**, reading data sequences forwards *and backwards*.
- **Significance**: Extends the standard LSTM by giving the network context of what happens *after* a particular reading sequence as well as before it, testing if dual-context history leads to sharper prediction logic.

### 5. `OzonePrediction_1-Layer_CNN-LSTM.py` *(The Hybrid Model)*
- **Architecture**: A combined **CNN-LSTM** architecture.
- **Significance**: Evaluates a powerful hybrid approach where the 1D Convolutional layers first extract distinct spatial/feature relationships from the metrics, before passing those refined, compressed patterns straight into an LSTM wrapper to assess them hierarchically through time.

### 6. `OzonePrediction_Transformer-2-block.py` *(The Attention Engine)*
- **Architecture**: A modern **Transformer** model utilizing dual multi-head attention blocks.
- **Significance**: Tests the absolute bleeding-edge mapping technique (originally designed for translating languages). Using "Attention", the network decides entirely for itself exactly which out of the 106 environmental features are fundamentally "worth paying attention to", ignoring standard sequential rules.

### 7. `OzonePrediction_LightGBM_Baseline.py` *(The Base ML Control)*
- **Architecture**: **LightGBM** (Light Gradient-Boosting Machine), a decision-tree based classical ML system.
- **Significance**: Serves as the ultimate mathematical "control group". Evaluates exactly how Deep Learning fares mathematically against a much simpler, incredibly fast, and traditionally trusted algorithm to see if all the neural network complexity is actually fundamentally worth the computational cost.
