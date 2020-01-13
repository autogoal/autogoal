
import keras

from keras.layers import Input, Dense
from keras.models import Model
from keras_bert import Tokenizer, load_vocabulary


class Bert(keras.layers.Layer):
    """ Permite utilizar Bert junto a keras y representa un embeding basado en Bert
    """

    def __init__(self, n_fine_tune_layers=10, **kwargs):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        super(Bert, self).__init__(**kwargs)

    def build(self, input_shape):
        bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
        self.bert = hub.Module(
            bert_path, trainable=self.trainable, name="{}_module".format(self.name)
        )
        trainable_vars = self.bert.variables

        # Remove unused layers
        trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]

        # Select how many layers to fine tune
        trainable_vars = trainable_vars[-self.n_fine_tune_layers :]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        # Add non-trainable weights
        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(Bert, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
            "pooled_output"
        ]
        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)


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
