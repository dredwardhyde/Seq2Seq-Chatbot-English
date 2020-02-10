# Seq2Seq Chatbot (English version)

## Requirements:
**Python 3.6.8+**  
**TensorFlow 2.1+**  
[**THIS MODEL CAN NOT BE TRAINED ON NVIDIA GPU BECAUSE OF THIS ISSUE**](https://github.com/tensorflow/tensorflow/issues/33148)  
[**THIS MODEL CAN NOT BE TRAINED ON RADEON GPU BECAUSE OF THIS ISSUE**](https://github.com/plaidml/plaidml/issues/142)
## Model:  
```
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, None)]       0                                            
__________________________________________________________________________________________________
input_2 (InputLayer)            [(None, None)]       0                                            
__________________________________________________________________________________________________
embedding (Embedding)           (None, None, 200)    1582000     input_1[0][0]                    
__________________________________________________________________________________________________
embedding_1 (Embedding)         (None, None, 200)    1582000     input_2[0][0]                    
__________________________________________________________________________________________________
lstm (LSTM)                     [(None, 200), (None, 320800      embedding[0][0]                  
__________________________________________________________________________________________________
lstm_1 (LSTM)                   [(None, None, 200),  320800      embedding_1[0][0]                
                                                                 lstm[0][1]                       
                                                                 lstm[0][2]                       
__________________________________________________________________________________________________
dense (Dense)                   (None, None, 7910)   1589910     lstm_1[0][0]                     
==================================================================================================
Total params: 5,395,510
Trainable params: 5,395,510
Non-trainable params: 0
```
## Small bot:

#### Source: [chatbot_small.py](https://github.com/dredwardhyde/seq2seq-chatbot-english/blob/master/chatbot_small.py)  
#### Dataset: [chatbot_nlp.zip](https://www.kaggle.com/hassanamin/chatbot-nlp)
#### Results:  
<img src="https://raw.githubusercontent.com/dredwardhyde/seq2seq-chatbot-english/master/chatbot_small_demo.PNG" width="900"/>

## Bigger bot:

#### Source: [chatbot_big.py](https://github.com/dredwardhyde/seq2seq-chatbot-english/blob/master/chatbot_big.py)  
#### Dataset: [Cornell Movie Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
#### Results:  
<img src="https://raw.githubusercontent.com/dredwardhyde/seq2seq-chatbot-english/master/chatbot_big_demo.PNG" width="900"/>

**Size of dataset for bigger chatbot is limited by you RAM,  
for example first 10000 samples take up to 45GB (total with OS):**  
<img src="https://raw.githubusercontent.com/dredwardhyde/seq2seq-chatbot-english/master/chatbot_load_memory.PNG" width="900"/>
