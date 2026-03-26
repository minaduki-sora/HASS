# HASS



HASS ([Learning Harmonized Representations for Speculative Sampling](https://arxiv.org/abs/2408.15766), ICLR 2025) is a speculative sampling method improved over EAGLE-2 by harmonizing the draft model's objective and context between its training and decoding stages, achieving significant improvements on speedup ratio and acceptance length.



## HASS Performance



### DeepSeek-R1

|             |                  | Datasets   |                |            |                |            |                 |            |                 |            |                |               |                 |            |                |
| ----------- | ---------------- | ---------- | -------------- | ---------- | -------------- | ---------- | --------------- | ---------- | --------------- | ---------- | -------------- | ------------- | --------------- | ---------- | -------------- |
|             |                  | MT-bench   |                | HumanEval  |                | GSM8K      |                 | Alpaca     |                 | QA         |                | Summarization |                 | Mean       |                |
| Temperature | Method           | Throughput | Speedup        | Throughput | Speedup        | Throughput | Speedup         | Throughput | Speedup         | Throughput | Speedup        | Throughput    | Speedup         | Throughput | Speedup        |
| T=0         | DeepSeek Vanilla | 35.6709    | 1.00x          | 35.5978    | 1.00x          | 35.6736    | 1.00x           | 35.6902    | 1.00x           | 35.6730    | 1.00x          | 34.4039       | 1.00x           | 35.4516    | 1.00x          |
|             | DeepSeek MTP     | 62.2485    | 1.75x          | 67.3626    | 1.89x          | 66.3277    | 1.86x           | 61.0297    | 1.71x           | 62.2504    | 1.75x          | 55.5675       | 1.62x           | 62.4644    | 1.76x          |
|             | HASS             | 66.2789    | 1.86x (+6.47%) | 73.6944    | 2.07x (+9.40%) | 71.4170    | 2.00x (+7.67%)  | 65.8383    | 1.84x (+7.88%)  | 67.4284    | 1.89x (+8.32%) | 61.8689       | 1.80x (+11.34%) | 67.7543    | 1.91x (+8.47%) |
| T=1         | DeepSeek Vanilla | 35.6798    | 1.00x          | 34.9864    | 1.00x          | 35.0866    | 1.00x           | 35.1235    | 1.00x           | 35.7631    | 1.00x          | 35.0862       | 1.00x           | 35.2876    | 1.00x          |
|             | DeepSeek MTP     | 54.5895    | 1.53x          | 58.6345    | 1.68x          | 58.9265    | 1.68x           | 49.0787    | 1.40x           | 53.5941    | 1.50x          | 46.1118       | 1.31x           | 53.4892    | 1.52x          |
|             | HASS             | 58.8619    | 1.65x (+7.83%) | 62.0878    | 1.77x (+5.89%) | 65.5968    | 1.87x (+11.32%) | 55.0553    | 1.57x (+12.18%) | 56.0470    | 1.57x (+4.58%) | 49.5268       | 1.41x (+7.41%)  | 57.8626    | 1.64x (+8.18%) |

- Evaluations of DeepSeek-R1 are under the [SGLang](https://github.com/sgl-project/sglang) framework, where the batch size is set as 1. DeepSeek Vanilla, DeepSeek MTP, and HASS represent the auto-regressive decoding, speculative sampling with the official MTP of DeepSeek-R1, and speculative sampling with the MTP continually trained with HASS, respectively. Throughput denotes output token throughput and is evaluated by token/s.
- Here, we continually train the MTP with HASS for 2 epochs on the ShareGPT dataset, where the training data is extremely less than the official MTP and the data distribution is inconsistent with that of DeepSeek-R1.
- On 2$\times$8 H800 GPUs, HASS achieves **1.41x-2.07x** speedup ratio compared to the auto-regressive decoding, surpassing DeepSeek-R1's official MTP by **8.47% / 8.18%** under temperature=0 / 1.



### DeepSeek-R1-Distill-Qwen-32B

|             |        | Datasets |        |           |        |         |        |         |        |         |        |               |        |         |        |
| ----------- | ------ | -------- | ------ | --------- | ------ | ------- | ------ | ------- | ------ | ------- | ------ | ------------- | ------ | ------- | ------ |
|             |        | MT-bench |        | HumanEval |        | GSM8K   |        | Alpaca  |        | QA      |        | Summarization |        | Mean    |        |
| Temperature | Method | Speedup  | $\tau$ | Speedup   | $\tau$ | Speedup | $\tau$ | Speedup | $\tau$ | Speedup | $\tau$ | Speedup       | $\tau$ | Speedup | $\tau$ |
| T=0         | HASS   | 3.4115x  | 4.1052 | 3.2300x   | 4.2911 | 3.4758x | 4.9809 | 3.1918x | 3.7738 | 2.7392x | 3.4779 | 2.9798x       | 3.7536 | 3.1714x | 4.0638 |
| T=1         | HASS   | 2.8712x  | 3.7402 | 2.9311x   | 3.8658 | 3.4341x | 4.7770 | 2.7634x | 3.5367 | 2.5586x | 3.2654 | 2.8616x       | 3.4244 | 2.9033x | 3.7683 |

- $\tau$ denotes the acceptance length.
- On 2 H800 GPUs, HASS achieves **2.56x-3.48x** speedup ratio compared to the auto-regressive decoding.



### LLaMA-series models

- On H800 GPU, HASS achieves **2.81x-4.05x** speedup ratio compared to the auto-regressive decoding, surpassing EAGLE-2 by **8%-20%**.
- Please refer to **Tables 1&2** in the paper.



## HASS Weights



| Base Model         | HASS Weights                                                 | Base Model                   | HASS Weights                                                 |
| ------------------ | ------------------------------------------------------------ | ---------------------------- | ------------------------------------------------------------ |
| DeepSeek-R1        | [HArmonizedSS/HASS-DeepSeek-R1](https://huggingface.co/HArmonizedSS/HASS-DeepSeek-R1) | DeepSeek-R1-Distill-Qwen-32B | [HArmonizedSS/HASS-DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/HArmonizedSS/HASS-DeepSeek-R1-Distill-Qwen-32B) |
| LLaMA3-Instruct-8B | [HArmonizedSS/HASS-LLaMA3-Instruct-8B](https://huggingface.co/HArmonizedSS/HASS-LLaMA3-Instruct-8B) | LLaMA3-Instruct-70B          | [HArmonizedSS/HASS-LLaMA3-Instruct-70B](https://huggingface.co/HArmonizedSS/HASS-LLaMA3-Instruct-70B) |
| LLaMA2-Chat-7B     | [HArmonizedSS/HASS-LLaMA2-Chat-7B](https://huggingface.co/HArmonizedSS/HASS-LLaMA2-Chat-7B) | LLaMA2-Chat-13B              | [HArmonizedSS/HASS-LLaMA2-Chat-13B](https://huggingface.co/HArmonizedSS/HASS-LLaMA2-Chat-13B) |



## Reference



```
@inproceedings{zhang2025learning,
  title={Learning Harmonized Representations for Speculative Sampling},
  author={Zhang, Lefan and Wang, Xiaodan and Huang, Yanhua and Xu, Ruiwen},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
```



## Acknowledgements



This project has been influenced by many excellent projects in the LLM community, such as [EAGLE](https://github.com/SafeAILab/EAGLE) and others.