# fcc-gpt-course

## 简介

这个部分主要跟着一个课程复现了GPT2的模型，并在课程的基础上进行一些改进

- YouTube: https://www.youtube.com/watch?v=UU1WVnMk4E8
- bilibili: https://www.bilibili.com/video/BV1vp421o7Ys/?spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=9247330f59ccaad7678f6728d1cd2714

### 一.数据下载和预处理

在视频中，作者关于数据的下载和处理，特别是下载部分稍微有些模糊，对于国内的朋友可能不太友好，因此我编辑了一段代码用来下载数据，并按照需求对数据进行处理。

### 二.模型输出的效果

我个人使用的计算机的配置为 4060Ti + 32G 内存，训练的配置如下：

```python
batch_size = 8
block_size = 128

max_iters = 10000
learning_rate = 3e-4
eval_iters = 100
n_embd = 384
n_head = 8
n_layer = 8
dropout = 0.2
```

最终得到的`loss=3.173016309738159`。虽然已经可以进行chat，但回答没有什么逻辑性。因此我将会在后面的更新中尝试用一些更好的数据集进行微调，以实现有一定准确度的对话。

### 三.模型的部署

在模型开发完之后，我会做一些简单的项目，以实现调用模型进行对话，项目可能是web或软件。
