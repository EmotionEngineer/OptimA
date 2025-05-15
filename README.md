# Optimal Activation Function (OptimA)

**OptimA** and its linearized variant, **OptimALinear**, are custom parametric activation functions designed to capture complex patterns in data by combining different nonlinear components. They aim to address key challenges in neural network training, such as gradient stability and expressive flexibility, by incorporating adjustable parameters that allow them to optimize their shape during training.

This repository contains the implementation of OptimA, OptimALinear, and a series of benchmark experiments comparing their performance against standard and other adaptive activation functions across various tasks.

## 1. Mathematical Formulation

### 1.1. OptimA (Optimal Activation)

The OptimA activation function is defined as:

$$
f_{\text{OptimA}}(x) = \alpha \cdot \tanh(\beta \cdot x) + \gamma \cdot \text{softplus}(\delta \cdot x) \cdot \text{sigmoid}(\lambda \cdot x)
$$

where:
- $x$ is the input value.
- $\alpha, \beta, \gamma, \delta, \lambda$ are trainable scalar parameters.
  - **Initialization (as used in experiments):** $\alpha=1, \beta=0.5, \gamma=1, \delta=0.5, \lambda=1$.
- Each component is chosen to introduce unique nonlinear transformations:
    - The **$\alpha \cdot \tanh(\beta \cdot x)$** term introduces an odd symmetry, effective for handling gradients and adapting to different data scales.
    - The **$\gamma \cdot \text{softplus}(\delta \cdot x) \cdot \text{sigmoid}(\lambda \cdot x)$** term combines a smooth, positive response (softplus) with a gating mechanism (sigmoid), allowing for complex, bounded nonlinear behaviors.

### Advantages of OptimA
1.  **Flexibility**: Trainable parameters allow OptimA to adapt its shape to specific tasks and datasets.
2.  **Gradient Stability**: The use of `tanh` and `sigmoid` (within bounded components) can help mitigate issues like vanishing or exploding gradients.
3.  **Expressive Power**: The combination of symmetric and gated asymmetric components enables the function to model a wide range of data patterns.

### 1.2. OptimALinear: Linear Approximation of OptimA

To explore a potentially less computationally intensive variant while retaining adaptability, **OptimALinear** is defined using piecewise linear approximations:

$$
f_{\text{linear}}(x) = \alpha \cdot \text{clip}(\beta \cdot x, -1, 1) + \gamma \cdot (\text{ReLU}(\delta \cdot x) + \epsilon) \cdot \left(0.5 + 0.25 \cdot \lambda \cdot x\right)
$$

where:
- **ReLU** replaces **softplus**, and the **clip** operation approximates **tanh** within the range $\[-1, 1]\$.
- The linear sigmoid approximation $\(0.5 + 0.25 \cdot \lambda \cdot x\)$ provides a close response to the original nonlinear sigmoid function for small inputs.

## 2. Implementation

OptimA and OptimALinear are implemented as custom Keras layers. The code can be found directly in the benchmark notebooks and is provided below for convenience:

```python
import tensorflow as tf
from tensorflow.keras.layers import Layer

# --- OptimA Definition ---
class OptimA(Layer):
    def __init__(self, **kwargs):
        super(OptimA, self).__init__(**kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight(name='alpha', shape=(), initializer='ones', trainable=True)
        self.beta = self.add_weight(name='beta', shape=(), initializer=tf.keras.initializers.Constant(0.5), trainable=True)
        self.gamma = self.add_weight(name='gamma', shape=(), initializer='ones', trainable=True)
        self.delta = self.add_weight(name='delta', shape=(), initializer=tf.keras.initializers.Constant(0.5), trainable=True)
        self.lambda_ = self.add_weight(name='lambda', shape=(), initializer='ones', trainable=True)
        super(OptimA, self).build(input_shape)

    def call(self, x):
        term1 = self.alpha * tf.math.tanh(self.beta * x)
        term2 = self.gamma * tf.math.softplus(self.delta * x) * tf.math.sigmoid(self.lambda_ * x)
        return term1 + term2

    def get_config(self):
        config = super(OptimA, self).get_config()
        return config

# --- OptimALinear Definition ---
class OptimALinear(Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super(OptimALinear, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.alpha = self.add_weight(name='alpha', shape=(), initializer='ones', trainable=True)
        self.beta = self.add_weight(name='beta', shape=(), initializer=tf.keras.initializers.Constant(0.5), trainable=True)
        self.gamma = self.add_weight(name='gamma', shape=(), initializer='ones', trainable=True)
        self.delta = self.add_weight(name='delta', shape=(), initializer=tf.keras.initializers.Constant(0.5), trainable=True)
        self.lambda_ = self.add_weight(name='lambda', shape=(), initializer='ones', trainable=True)
        super(OptimALinear, self).build(input_shape)

    def call(self, x):
        term1 = self.alpha * tf.clip_by_value(self.beta * x, -1, 1)
        term2 = self.gamma * (tf.maximum(0.0, self.delta * x) + self.epsilon) * (0.5 + 0.25 * self.lambda_ * x)
        return term1 + term2

    def get_config(self):
        config = super(OptimALinear, self).get_config()
        config.update({'epsilon': self.epsilon})
        return config

# Example Usage:
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# model = Sequential()
# model.add(Dense(64)) # Example Dense layer
# model.add(OptimA())  # Apply OptimA after a Dense layer
# # or model.add(Dense(64, activation=OptimA())) if OptimA is registered or passed as an object
# ...
```

## 3. Experiments and Results

We have conducted a comprehensive suite of benchmark experiments to evaluate the performance of OptimA and OptimALinear. These functions were compared against several standard and other adaptive activation functions across various domains: tabular data, Natural Language Processing (NLP), and Computer Vision (CV).

**All experimental setups, code, and detailed results logs can be found in the Jupyter Notebooks within this repository.**

### 3.1. Quick Benchmark (Tabular Data, Single Seed)
- **Notebook:** [`fast-benchmark.ipynb`](fast-benchmark.ipynb)
- **Tasks:** Regression (Boston Housing, Diabetes, California Housing) and Classification (Iris, Wine).
- **Metrics:** MSE for regression, Accuracy for classification.
- **Summary Table (from notebook):**

| Activation Function | Boston Housing (MSE) | Iris (Accuracy) | Wine (Accuracy) | Diabetes (MSE) | California Housing (MSE) |
|---------------------|----------------------|-----------------|-----------------|----------------|--------------------------|
| **OptimA**          | **20.36**            | **1.00**        | **1.00**        | **3021.60**    | **0.294**                |
| **OptimALinear**    | 26.59                | **1.00**        | **1.00**        | 3271.58        | 0.745                    |
| **ReLU**            | 24.59                | 0.97            | 0.97            | 5695.40        | 0.324                    |
| **ELU**             | 22.70                | **1.00**        | **1.00**        | 9833.83        | 0.343                    |
| **Swish**           | 24.96                | 0.97            | **1.00**        | 8055.66        | 0.350                    |
| **GeLU**            | 24.37                | **1.00**        | 0.97            | 6356.98        | 0.323                    |

### 3.2. Extended Tabular Data Benchmarks (Multiple Seeds & Batch Sizes)
- **Notebooks:**
    - Comparison with other adaptive activations: [`Sandbox/AdaptiveBenchmark.ipynb`](Sandbox/AdaptiveBenchmark.ipynb)
    - Comparison with standard activations: [`Sandbox/BaseBenchmark.ipynb`](Sandbox/BaseBenchmark.ipynb)
- **Metrics:** MAE for regression (lower is better), Accuracy for classification (higher is better).
- **Representative Results (Mean over 5 seeds, AdamW optimizer, Batch Size 16):**

*Table 3.2.1: Tabular Data - Regression (MAE)*
| Dataset            | OptimA   | OptimALinear | Best Standard (Act) | Best Other Adaptive (Act) |
|--------------------|----------|--------------|-----------------------|-----------------------------|
| Boston Housing     | ~2.97    | **~2.34**    | ReLU (~2.61)          | AdaptiveMish (~2.49)        |
| Diabetes           | ~43.6    | **~42.5**    | ELU (~42.6)           | AdaptiveSwish (~43.5)       |
| California Housing | **~0.361** | ~0.363       | ReLU (~0.370)         | AdaptiveSwish (~0.359)      |

*Table 3.2.2: Tabular Data - Classification (Accuracy)*
| Dataset | OptimA   | OptimALinear | Best Standard (Act) | Best Other Adaptive (Act) |
|---------|----------|--------------|-----------------------|-----------------------------|
| Iris    | **~0.853** | ~0.807       | ELU (~0.807)          | AdaptiveMish (~0.807)       |
| Wine    | ~0.933   | **~0.972**   | GeLU (~0.967)         | AdaptiveMish (~0.961)       |

_Note: "Best Standard" and "Best Other Adaptive" refer to the best performing function from those respective categories within the specific benchmark run._

### 3.3. Natural Language Processing (NLP) Benchmarks
- **Tasks:** Text classification on IMDB dataset.
- **Models:** Fine-tuned BERT (`bert-base-uncased`) and a custom LSTM model.
- **Notebooks:**
    - BERT: [`NLP/BERT_1.ipynb`](NLP/BERT_1.ipynb), [`NLP/BERT_2.ipynb`](NLP/BERT_2.ipynb)
    - LSTM: [`NLP/LSTM.ipynb`](NLP/LSTM.ipynb)
- **Results (Accuracy on test set, single run):**

*Table 3.3.1: NLP - IMDB Classification (Accuracy)*
| Model | Activation    | Accuracy | Approx. Training Time |
|-------|---------------|----------|-----------------------|
| BERT  | **OptimA**    | **0.8910** | ~5168s                |
| BERT  | OptimALinear  | 0.8896   | ~3886s                |
| BERT  | GeLU (baseline) | 0.8906   | ~4634s                |
| BERT  | Swish         | 0.8880   | ~2350s                |
| LSTM  | **OptimA**    | **0.8419** | ~1906s                |
| LSTM  | OptimALinear  | 0.8352   | ~2858s                |
| LSTM  | Swish         | 0.8410   | ~1921s                |
| LSTM  | Tanh          | 0.8409   | ~2208s                |

### 3.4. Computer Vision (CV) Benchmarks
- **Models:** Simple CNN for MNIST, CIFAR-10, Rotated MNIST; ResNet50 for UTKFace.
- **Notebooks:**
    - Simple CNN tasks: [`CNN/ImageTasksBenchmark.ipynb`](CNN/ImageTasksBenchmark.ipynb)
    - UTKFace (ResNet50): [`CNN/UTKFaceBenchmark.ipynb`](CNN/UTKFaceBenchmark.ipynb)
      - Kaggle Mirror for UTKFace: [OptimA Activation - UTKFace Benchmark](https://www.kaggle.com/code/saicourse/optima-activation-utkface-benchmark)
- **Results (Accuracy for classification, MAE for regression; single run):**

*Table 3.4.1: CV - Image Classification (Accuracy, Simple CNN, AdamW optimizer)*
| Dataset  | OptimA   | OptimALinear | ReLU     | GeLU     |
|----------|----------|--------------|----------|----------|
| MNIST    | ~0.9890  | ~0.9894      | **~0.9907** | ~0.9902  |
| CIFAR-10 | ~0.6414  | ~0.6777      | **~0.6890** | ~0.6507  |

*Table 3.4.2: CV - Image Regression (MAE, Simple CNN with Adam / ResNet50 with AdamW)*
| Dataset        | Model      | OptimA   | OptimALinear | ReLU     | ELU      |
|----------------|------------|----------|--------------|----------|----------|
| RotatedMNIST   | Simple CNN | ~0.1015  | **~0.0868**  | ~0.0869  | ~0.1122  |
| UTKFaceAge     | ResNet50   | **~5.489** | ~5.759       | ~5.599   | ~5.579   |

### 3.5. Key Observations Across All Benchmarks
1.  **No Single Best Activation:** The performance is highly dependent on the dataset, model architecture, and even the optimizer.
2.  **OptimA's Potential:** Often shines in more complex setups or tasks requiring nuanced non-linearities (e.g., BERT, LSTM, ResNet50 for age regression). Its adaptability allows it to find effective functional forms.
3.  **OptimALinear's Niche:** Frequently performs well in regression tasks (especially with simpler CNNs and tabular data) and some classification tasks, offering a good trade-off.
4.  **Comparison with Baselines:** Both OptimA and OptimALinear are competitive and, in several instances, outperform standard activations like ReLU, ELU, Swish, and GeLU, as well as other adaptive functions.
5.  **Interpretability and Customization:** The learned parameters of OptimA/OptimALinear can provide insights into what kind of non-linearity is beneficial for a task. Furthermore, these learned parameters can be "frozen" post-training to create a new, static, task-specialized activation function.
6.  **Training Time:** Parametric activations like OptimA can sometimes lead to longer training times per epoch due to the overhead of learning additional parameters, as observed in the BERT experiments.

## 4. Conclusion

OptimA and OptimALinear are versatile parametric activation functions that demonstrate strong potential for enhancing the performance of deep learning models across a variety of domains. Their ability to adapt their functional form during training makes them valuable tools for tackling complex data patterns.

While no activation function is universally superior, OptimA and OptimALinear offer compelling alternatives to standard fixed activations, particularly when fine-grained adaptation to the task is beneficial. The choice between OptimA and OptimALinear may depend on the specific problem, with OptimA often excelling in scenarios demanding high expressiveness and OptimALinear providing a robust, simpler alternative.

The insights from the learned parameters and the possibility of deriving specialized static functions further underscore their utility for both practical applications and research into the nature of neural network learning.

## 5. Future Work
- More extensive benchmarking with multiple random seeds for all experiments to ensure statistical significance.
- Deeper theoretical analysis of the approximation capabilities and gradient dynamics of OptimA and OptimALinear.
- Exploration of optimal initialization strategies for the trainable parameters $\alpha, \beta, \gamma, \delta, \lambda$.
- Application in other deep learning domains such as Reinforcement Learning, Generative Adversarial Networks (GANs), and Graph Neural Networks (GNNs).
- Investigating layer-specific or channel-specific parameterization for even greater adaptability.

## 6. License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
