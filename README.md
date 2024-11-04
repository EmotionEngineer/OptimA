# Optimal Activation Function (OptimA)

**OptimA**, is a custom activation function designed to capture complex patterns in data by combining different nonlinear components. **OptimA** aims to address key challenges in neural network training, such as gradient stability and expressive flexibility. The design of **OptimA** incorporates adjustable parameters to optimize performance across a wide range of tasks.

## 1. Mathematical Formulation of OptimA

The OptimA activation function is defined as:

$$
f(x) = \alpha \cdot \tanh(\beta \cdot x) + \gamma \cdot \text{softplus}(\delta \cdot x) \cdot \text{sigmoid}(\lambda \cdot x)
$$

where:
- $\alpha$, $\beta$, $\gamma$, $\delta$, and $\lambda$ are trainable parameters.
- Each component is carefully chosen to introduce unique nonlinear transformations, creating a balanced and expressive activation function.

### Explanation of Each Component

- **$\alpha \cdot \tanh(\beta \cdot x)$**:
  - This term introduces an **odd symmetry**, making the function responsive to both positive and negative values.
  - The **$\tanh$** nonlinearity is effective for handling gradients, reducing the likelihood of gradient vanishing or explosion issues in deep networks.
  - The parameters $\alpha$ and $\beta$ control the amplitude and sensitivity of this component, allowing it to adapt to different data scales.

- **$\gamma \cdot \text{softplus}(\delta \cdot x) \cdot \text{sigmoid}(\lambda \cdot x)$**:
  - This term combines **softplus** and **sigmoid** functions to create a smooth, bounded response to input values.
  - **Softplus** allows for a smooth transition above zero, while **sigmoid** controls saturation at extreme values.
  - Parameters $\gamma$, $\delta$, and $\lambda$ govern the scaling and flexibility of this component, enabling a wide range of nonlinear behaviors that can suit various tasks.

### Advantages of OptimA

1. **Flexibility**: The trainable parameters provide a means to adapt the activation function to various tasks, which can lead to improved performance on diverse datasets.
2. **Stability**: The combination of `tanh` and `sigmoid` helps mitigate gradient saturation and explosion issues, making it suitable for both shallow and deep architectures.
3. **Expressive Power**: With both symmetric and asymmetric components, the function can capture a broad range of patterns, aiding in complex feature extraction.

## 2. OptimALinear: Linear Approximation of OptimA

To reduce computational complexity, we also define **OptimALinear**, a linear approximation of OptimA that replaces nonlinear functions with linear transformations. OptimALinear is expressed as:

$$
f_{\text{linear}}(x) = \alpha \cdot \text{clip}(\beta \cdot x, -1, 1) + \gamma \cdot (\text{ReLU}(\delta \cdot x) + \epsilon) \cdot \left(0.5 + 0.25 \cdot \lambda \cdot x\right)
$$

where:
- **ReLU** replaces **softplus**, and the **clip** operation approximates **tanh** within the range $\[-1, 1]\$.
- The linear sigmoid approximation $\(0.5 + 0.25 \cdot \lambda \cdot x\)$ provides a close response to the original nonlinear sigmoid function for small inputs.
  
## 3. Experiment and Results

We evaluated **OptimA** and **OptimALinear** alongside other popular activation functions (ReLU, ELU, Swish, and GeLU) on multiple tasks to understand the effectiveness of these activation functions. The tasks include both regression and classification challenges on structured data. For a detailed implementation, refer to the [benchmark notebook](benchmark.ipynb) in the repository.

### Results Summary

The following table presents the evaluation results for each activation function on different datasets:

| Activation Function | Boston Housing (Regression) | Iris (Classification) | Wine (Multiclass Classification) | Diabetes (Regression) | California Housing (Regression) |
|---------------------|-----------------------------|-----------------------|----------------------------------|------------------------|---------------------------------|
| **OptimA**          | 20.36                       | 1.00                  | 1.00                             | 3021.60                | 0.294                           |
| **OptimALinear**    | 26.59                       | 1.00                  | 1.00                             | 3271.58                | 0.745                           |
| **ReLU**            | 24.59                       | 0.97                  | 0.97                             | 5695.40                | 0.324                           |
| **ELU**             | 22.70                       | 1.00                  | 1.00                             | 9833.83                | 0.343                           |
| **Swish**           | 24.96                       | 0.97                  | 1.00                             | 8055.66                | 0.350                           |
| **GeLU**            | 24.37                       | 1.00                  | 0.97                             | 6356.98                | 0.323                           |

### Observations

- **OptimA** achieved superior performance across several tasks, particularly for regression problems, where it demonstrated lower mean absolute error (MAE).
- **OptimALinear** performs well for classification tasks, but with a slight drop in performance for regression tasks due to the linear approximations.
- Standard activation functions like **ReLU** and **Swish** perform consistently, but **OptimA** outperforms them on complex, nonlinear tasks.

## Conclusion

The **OptimA** activation function, with its trainable parameters and dual-component design, exhibits flexibility and stability, proving to be effective on a variety of machine learning tasks. This function can serve as an adaptable alternative to traditional activation functions in scenarios where complex patterns and interactions need to be learned.
