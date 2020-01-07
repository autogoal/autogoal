
from keras.layers import Input, Dense
from keras.models import Model
from autogoal.ontology._keras import Bert

from keras_bert import Tokenizer, load_vocabulary

def main():
    layer_num = 12
    vocab_path = "/root/.keras/datasets/multi_cased_L-12_H-768_A-12/vocab.txt"

    vocab = load_vocabulary(vocab_path)
    tokenizer = Tokenizer(vocab)

    # config_path = os.path.join(checkpoint_path, 'bert_config.json')
    # model_path = os.path.join(checkpoint_path, 'bert_model.ckpt')
    # model = load_trained_model_from_checkpoint(
    #     config_path,
    #     model_path,
    #     training=False,
    #     use_adapter=True,
    #     trainable=['Encoder-{}-MultiHeadSelfAttention-Adapter'.format(i + 1) for i in range(layer_num)] +
    #     ['Encoder-{}-FeedForward-Adapter'.format(i + 1) for i in range(layer_num)] +
    #     ['Encoder-{}-MultiHeadSelfAttention-Norm'.format(i + 1) for i in range(layer_num)] +
    #     ['Encoder-{}-FeedForward-Norm'.format(i + 1) for i in range(layer_num)],
    # )

    # max_seq_length = 1000
    # input_ids=[0] * max_seq_length
    # input_mask=[0] * max_seq_length
    # segment_ids=[0] * max_seq_length
    # inputs = (input_ids, input_ids, segment_ids)

    texts = [
        "This is an example of awesome text"
    ]

    max_text_length = max(len(t) for t in texts)
    tokens = [tokenizer.encode(t) for t in texts]

    x_indices, x_mask, x_segments = [
        Input((max_text_length,)), Input((max_text_length,)), Input((max_text_length,))
    ]
    x = [x_indices, x_mask, x_segments]
    bert = Bert()(x)
    dense = Dense(units=4, activation='softmax')(bert)

    model = Model(inputs=x, outputs=dense)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.summary()


if __name__ == "__main__":
    main()
