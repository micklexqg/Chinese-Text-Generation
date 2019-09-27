import os
import re
import random
import logging
import numpy as np
import keras
from keras import layers


class Params:
    data_path = "Fan Ren Xiu Xian Chuan [ Geng Xin Zhi  Di - Wang Yu.txt"
    max_len = 20
    batch_size = 2048


with open(Params.data_path, encoding='utf-8') as f:
    text = f.read()

text = re.sub("------章节内容开始-------", "", text)
text = re.sub("[(（].{2,}?[)）]", "", text)
print("len(text): {}".format(len(text)))

chars = list(set(text))
Params.chars = chars
print("len(chars): {}".format(len(chars)))

char_indices = dict((char, chars.index(char)) for char in chars) 
Params.char_indices = char_indices


def data_generator(text, batch_size):

    while 1:

        sentences = []
        next_chars = []
        for _ in range(batch_size):
            start_idx = random.randint(0, len(text)-Params.max_len-1)
            sentence = text[start_idx:start_idx+Params.max_len]
            next_char = text[start_idx+Params.max_len]

            sentences.append(sentence)
            next_chars.append(next_char)

        x = np.zeros((batch_size, Params.max_len, len(Params.char_indices)), dtype=np.bool)
        y = np.zeros((batch_size, len(Params.char_indices)), dtype=np.bool)

        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, char_indices[char]] = 1
                y[i, char_indices[next_chars[i]]] = 1

        yield x,y


model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(Params.max_len, len(Params.char_indices)), return_sequences=True))
model.add(layers.LSTM(128))
model.add(layers.Dense(len(Params.char_indices), activation='softmax'))

model.summary()

optimizer = keras.optimizers.RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


model.fit_generator(
    generator = data_generator(text, Params.batch_size),
    steps_per_epoch = 100,
    epochs = 30,
)


# model.save_weights("12.h5")



def sample(preds, temperature=1.0):
    '''sample the next character given the model's predictions
    softmax temperature that characterizes the entropy of the probability distribution 
    used for sampling: it characterizes how surprising or predictable the choice of next 
    character will be.
    
    low temperature = more deterministic
    high temperature = more random
    
    '''
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    
    # reweight the original distribution
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
    

for i in range(10):

    print("########################")

    start_index = random.randint(0, len(text)-Params.max_len-1)
    original_text = text[start_index: start_index + Params.max_len]

    print('original_text: ', original_text) 

    generated_text = ""
    for i in range(100):
        # one-hot encodes the characters original_text
        sampled = np.zeros((1, Params.max_len, len(Params.char_indices)))
        for t, char in enumerate(original_text):
            sampled[0, t, Params.char_indices[char]] = 1.

        preds = model.predict(sampled, verbose=0)[0]
        next_index = sample(preds, 0.7)
        next_char = Params.chars[next_index]
        
        generated_text += next_char
        
        original_text += next_char
        original_text = original_text[1:]
        
    print('generated_text: ', generated_text) 
        

"""
Epoch 1/30
2019-09-27 07:34:31.544743: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
100/100 [==============================] - 63s 630ms/step - loss: 6.2428
Epoch 2/30
100/100 [==============================] - 62s 624ms/step - loss: 6.0792
Epoch 3/30
100/100 [==============================] - 61s 614ms/step - loss: 6.0240
Epoch 4/30
100/100 [==============================] - 61s 614ms/step - loss: 5.9180
Epoch 5/30
100/100 [==============================] - 61s 615ms/step - loss: 5.8000
Epoch 6/30
100/100 [==============================] - 61s 614ms/step - loss: 5.6642
Epoch 7/30
100/100 [==============================] - 61s 614ms/step - loss: 5.5475
Epoch 8/30
100/100 [==============================] - 61s 609ms/step - loss: 5.4409
Epoch 9/30
100/100 [==============================] - 61s 609ms/step - loss: 5.3507
Epoch 10/30
100/100 [==============================] - 61s 607ms/step - loss: 5.2540
Epoch 11/30
100/100 [==============================] - 61s 607ms/step - loss: 5.1870
Epoch 12/30
100/100 [==============================] - 61s 606ms/step - loss: 5.0928
Epoch 13/30
100/100 [==============================] - 61s 608ms/step - loss: 5.0256
Epoch 14/30
100/100 [==============================] - 61s 607ms/step - loss: 4.9441
Epoch 15/30
100/100 [==============================] - 61s 609ms/step - loss: 4.8683
Epoch 16/30
100/100 [==============================] - 61s 608ms/step - loss: 4.7837
Epoch 17/30
100/100 [==============================] - 61s 609ms/step - loss: 4.7290
Epoch 18/30
100/100 [==============================] - 61s 609ms/step - loss: 4.6563
Epoch 19/30
100/100 [==============================] - 62s 618ms/step - loss: 4.6093
Epoch 20/30
100/100 [==============================] - 61s 615ms/step - loss: 4.5547
Epoch 21/30
100/100 [==============================] - 65s 649ms/step - loss: 4.5156
Epoch 22/30
100/100 [==============================] - 69s 689ms/step - loss: 4.4682
Epoch 23/30
100/100 [==============================] - 69s 690ms/step - loss: 4.4230
Epoch 24/30
100/100 [==============================] - 69s 693ms/step - loss: 4.3879
Epoch 25/30
100/100 [==============================] - 67s 672ms/step - loss: 4.3552
Epoch 26/30
100/100 [==============================] - 67s 672ms/step - loss: 4.3125
Epoch 27/30
100/100 [==============================] - 67s 670ms/step - loss: 4.2782
Epoch 28/30
100/100 [==============================] - 68s 675ms/step - loss: 4.2596
Epoch 29/30
100/100 [==============================] - 64s 639ms/step - loss: 4.2154
Epoch 30/30
100/100 [==============================] - 62s 615ms/step - loss: 4.2109
########################
original_text:  奉，准备隐姓埋名，在山上度过自己最后的日
generated_text:  后，他竟然没有的说到了韩立。

她王宗等人和他人眼中的低处，又对他们不想，

“不过。我不是我们和这天灵草，也不知道友了。否则自是是我们的灵大！”

“原来，不知道，我也是她不知道”

“我的千年修士
########################
original_text:  

至于一旁地尤姓修士，对韩立也抱着差不
generated_text:  定，对韩立的安然不会，自然先出了一个儿，眼后，仍然露出了心色。

“不停，你们对方不会的绝某有了。前辈才知道一批就是麻基之术。

但是韩立飞术之人的法器，现在如此交会之极的，有些个伙都不会等我们一名修
########################
original_text:  之侧目的话来。

“这严氏也太偏袒自己夫
generated_text:  日。其他的修士，要同样对方是在这里，才是如此的事情。

就在这边，韩立脸上露出一丝惊色的惊大大孔，看出面面目光一动，目淡淡的说道：““嘿嘿，放下还没有死！”

如今，韩立听到了这一位面上人，但是元婴为
########################
original_text:  牵全身，根本无法破禁而入。

唯一没有禁
generated_text:  制的。

这种小情在在他的中长，竟是一开始他年人的韩立不言，开始面的收快。

他这时，韩立就显意了下来，韩立就在他们的最上一半之后，就过了一条波动后。

“公子，是无法器的话，这个大族大外的修士，不定
########################
original_text:  淡思量道

看来自己消减最后两拨雷击的威
generated_text:  力，一股狂光的遁光浮现而下，在空中一闪，还化为一道金球，一个白光大大下，一只从银鼎中将一阵遁中，韩立一跳始，也根本无法走到。

这里，韩立等人的神色。“二玉简在此了，有何不会 对口友的。韩道友不敢说过
########################
original_text:  是其他人身前漂浮的数件宝物，此刻都光芒黯
generated_text:  皱，一手拳，身形从冲手中一过。

韩立站着一惊，猛然就一张口，见此女前人一些口，正是对方的典膝中双目，一一意的打起了一声呼滚，手中的银光闪出，一闪手，金色一只指破出现在了他。

“灵族要在元婴中期修士
########################
original_text:  道。

听了韩立这话。钟卫娘一愣的向下望
generated_text:  了。

韩立眉声微皱，看着这些魔瑶的话，不见了，看着几位男子却心中的一声。

“晚辈不定！在下前就就像妖兽什么军。而只是他们的修仙者也不同一时一座的韩立的面上，但对韩立自然向其一下，将他们们的神色的神
########################
original_text:  了女子的脖颈处。

“兹啦”一声。一层厚
generated_text:  光一闪，一片蓝色红光从刚上面上一闪，一个青光直奔了方上。

青年一怔，将韩立身体出一名中长里，大人的意子，又打见了韩立有些不慌为的望着，大厅其中的淡白色，眉中却脸色一闪，收着他们他人不停的说道。

“
########################
original_text:  。”冯姓老者苦笑一声。带有三分自嘲三分羡
generated_text:  的。

“启了！”

“不错，一名老少老倒也要到这次有的三女子。”韩立摸了一声，从老子大动了下来。

韩立顿时一点“急笑了。

青年有些身体，一团青光闪动，同时一转，一手指一闪，一声金色金色粗大的大汉
########################
original_text:  

“嘿嘿！是吗？富兄既然如此说了。不妨
generated_text:  ，我们我们能定超在和天星族力薄的时间，也不想到这里的深阶妖兽。

这二人虽然以上是可未有一样，其他有何所足，但不是他们的修士可能换受了。但是韩立一亮，接着的低头，围着一道道





第六卷 人灵界百

"""
        
