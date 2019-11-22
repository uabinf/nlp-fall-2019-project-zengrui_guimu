from keras.layers import Input, Convolution1D, Dot, Dense, Activation, Concatenate
from keras.models import Model

encoder_inputs = Input(shape=(None, num_encoder_tokens))
x_encoder = Convolution1D(128, kernel_size=3, activation='relu',padding='causal')(encoder_inputs)
x_encoder = Convolution1D(128, kernel_size=3, activation='relu',padding='causal', dilation_rate=2)(x_encoder)
x_encoder = Convolution1D(128, kernel_size=3, activation='relu',padding='causal', dilation_rate=4)(x_encoder)

decoder_inputs = Input(shape=(None, num_decoder_tokens))
x_decoder = Convolution1D(128, kernel_size=3, activation='relu',padding='causal')(decoder_inputs)
x_decoder = Convolution1D(128, kernel_size=3, activation='relu',padding='causal', dilation_rate=2)(x_decoder)
x_decoder = Convolution1D(128, kernel_size=3, activation='relu',padding='causal', dilation_rate=4)(x_decoder)

attention = Dot(axes=[2, 2])([x_decoder, x_encoder])
attention = Activation('softmax')(attention)
context = Dot(axes=[2, 1])([attention, x_encoder])
decoder_combined_context = Concatenate(axis=-1)([context, x_decoder])

decoder_outputs = Convolution1D(128, kernel_size=3, activation='relu',padding='causal')(decoder_combined_context)
decoder_outputs = Convolution1D(128, kernel_size=3, activation='relu',padding='causal')(decoder_outputs)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,batch_size=64,epochs=50,validation_split=0.2)
model.save('shake_cnn')
