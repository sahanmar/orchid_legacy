# Orchid Project 
![vanilla orchid](https://github.com/sahanmar/orchid/blob/imgs/imgs/orchid.png)

Orchid or in other words Vanilla Orchid project is aimed on a development of a new-state-of-the-art coreference resolution algorithm(s*)

## Structure ⚙

We build this project on the top of mega fancy state-of-the-art language understanding transformers models e.g. [SpanBERT](https://arxiv.org/abs/1907.10529). 

The most important things about the building of the project locally:
1. [config.json](https://github.com/sahanmar/orchid/blob/main/config/config.json) file where you can change all the dependencies in one place and then run your code right away! 
2. If you want to know how the whole pipeline works, just simply go to [pipeline.py](https://github.com/sahanmar/orchid/blob/main/pipeline.py) and check both comments and code.

**N.B.** The project is WIP and is going to grow <s>daily</s> weekly! 

## Data 💽

All state of the art algorithms are trained and evaluated on [OntoNotes 5.0 dataset](https://catalog.ldc.upenn.edu/LDC2013T19).
Due to the licencing, bureaucracy and other uninteresting stuff, these data must be processed to `*._conll` format. The guidelines for doing that can be found [here](https://cemantix.org/conll/2012/data.html). 

### Important Note ⬇️ 

If you are not a genius same as me, you can use a [Marvelous Blogpost](https://medium.com/huggingface/how-to-train-a-neural-coreference-model-neuralcoref-2-7bb30c1abdfe) from [Hugging Face 🤗](https://huggingface.co)   

### Data preprocessing cookbook 👨‍🍳

1. In order to get [OntoNotes 5.0 dataset](https://catalog.ldc.upenn.edu/LDC2013T19) you have to: 
   * Register
   * Get your account verified
   * Create a data Request
   * Finally download the data.
2. Download Train data, Test data and additional scripts [here](https://cemantix.org/conll/2012/data.html). These are the so called `skeleton` data that must be combined with the original OntoNotes data that you must have been already downloaded. Enter gently each folder and go deeper till you find `train`, `test` and `scripts` folders in three above mentioned directories. Choose one of the directories where you will copy the rest. All in all you must have `train`, `test` and `scripts` all together. The guidelines for using tge scripts are in the same link as the above mentioned data from this paragraph. 
   * **NB! The original folder that will be used for the data extraction must have conll-2012 name or you will get errors**.
3. You can change all the scripts provided by [Conll-2012](https://cemantix.org/conll/2012/data.html) to `Python 3` by changing the exception `except ErrorMsg, e` -> `except ErrorMsg as e` and all prints from `print "smth"` -> `print("smth")`. Or you can have an environment (e.g conda env) with `Python 2.7`. My personal choice is having a separate `2.7` environment because its either faster to switch to a different env and sometimes you just need it to work with some archaic scripts. This step will create that mysterious `*.v4_gold_conll` files in `conll-2012` directory. 
   * **NB! The whole process will take around 90 minutes. At least my poor MacPro 2017 was struggling that exact amount of time. Moreover, there is no progress bar so be patient Morty!**. 
4. I'm following HuggingFace guidelines. They tell to assemble the appropriate files into one large file each for training, development and testing. 
   * `my_lang` can be `english`, `arabic` or `chinese`
   * `cat conll-2012/v4/data/train/data/my_lang/annotations/*/*/*/*.v4_gold_conll >> train.my_lang.v4_gold_conll`
   * `cat conll-2012/v4/data/development/data/my_lang/annotations/*/*/*/*.v4_gold_conll >> dev.my_lang.v4_gold_conll`
   * `cat conll-2012/v4/data/test/data/my_lang/annotations/*/*/*/*.v4_auto_conll >> test.my_lang.v4_auto_conll`
    
    Put all these data to separate folders `train`, `test` and `dev` and you are all set!