# tf-bert-pretraining-pattern

This repository implements pretraining logic to estimate BERT model parameters only with masked language model(MLM) task. Especially, this project doesn't rely on huggingface framework because of my personal view as junior data scientist(only with 2 year field experiences). 

Although it is common practice to rely on huggingface framework even when implementing real-world system whenever it has to leverage power of transformer-family model, I personally think that level of abstraction it provides is too high(i.e. too high-level interface). This is because it tends to implement every case that BERT model can handle to increase versatility of the huggingface software. For example, it contains API for full encoder-decoder architecture support for certain task like neural machine translation(NMT). This gives unnecessary burden for user who wish to understand how code works even if the user needs to make use of encoder part only. 

In such situation, common choice is to ignore the detailed part of implementation for 'high developer productivity' and use API as instructed. As a result, overall technical debt of the system accumulates, and this makes real-world application more prone to cause unexpected errors. To minimize such risk, I believe that an organization should make effort to reduce reliance on black-box code explained above. 

Of course, it can put good amount of effort to comprehend the implemented code. However, this project takes alternative approach by implementing end-to-end BERT pretraining logic only with Tensorflow API. It is true that corresponding huggingface implementation is referenced; key point is to compose the system with comprehensible code components.

## Execute

These codes are implemented and executed on Apple silicon macOS environment within conda environment. Given conda distribution installed and directory changed to the cloned repository, required environment can be composed with  typical conda command below. Then, since environment name is set to *tbpp*(acronym for this repository), composed environment can be activated with following command.

```shell
conda env create
conda activate tbpp
```

During preprocess step, program fetches IMDb dataset from tensorflow datasets. Then, using predefined text preprocessing logic to replace `<br>` tag to single space and lower every alphabet in each document, it saves corpus file into configured local directory. This directory can be customized by fixing `LOCAL_DIR` variable in `config.py` file. This step can be executed by following command.

```shell
python main.py preprocess
```

During training step, parameters of BERT model are estimated through masked language model(MLM) task on corpus saved in preprocess step. Of course, size of the corpus is not enough to fully train the parameters, so model is likely to be undertrained. Nevertheless, it doesn't matter since purpose of this project is to implement the whole BERT pretraining process, not generating nicely performing model with well-prepared rich data. By entering following command, machine will start BERT model pretraining.

```shell
python main.py train
```

Since this implementation leverages keras API, corresponding progress log is printed on the console. For example, it might look like:

```
Epoch 1/3
782/782 [=====================] - 302s 384ms/step - loss: 7.2235
Epoch 2/3
782/782 [=====================] - 304s 389ms/step - loss: 6.8903
Epoch 3/3
782/782 [=====================] - 322s 412ms/step - loss: 6.7830
```

Finally, model is fine-tuned to perform sentiment analysis job. Specifically, it is trained to perform binary classification task to tell whether given review states positive or negative view on the reviewed movie. Enter following command on the console to start the fine-tuning process. 

```shell
python main.py evaluate
```

As in train step, progress log will be printed on the console. It might look like following message. Note that model achieves far inferior level of accuracy compared to expected performance presented in many benchmark results. This implies that trained model is far undertrained, as noted in train step paragraph.

```
Epoch 1/3
782/782 [=====================] - 309s 393ms/step - val_accuracy: 0.7813
Epoch 2/3
782/782 [=====================] - 307s 392ms/step - val_accuracy: 0.8124
Epoch 3/3
782/782 [=====================] - 308s 393ms/step - val_accuracy: 0.8165
```

So, for explorers who succeed to reach this project: be sure to use this code just for self-studying. To use implemented logic in real world use-case, prepare richer corpus to feed in the train step. 