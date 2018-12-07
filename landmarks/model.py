
from keras.models import Sequential
from keras.layers import LSTM, Input, Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Reshape, BatchNormalization
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU




def build_model_LSTM():
    inputs = Input(shape=(window_size, embedding_size))
    seq_input_drop = Dropout(initial_dropout)(inputs)

    # Sequence modeling network and final output
    if lstm_attention:
        lstm_output = LSTM(lstm_output_dim, return_sequences=True)(seq_input_drop)
        lstm_output, _ = AttentionWeightedAverage(name='attlayer', attention_type=attention_type)(lstm_output)
    else:
        lstm_output = LSTM(lstm_output_dim, return_sequences=False)(seq_input_drop)
        
    second_last = Dropout(final_dropout)(lstm_output)
    second_last_drop = Dense(second_last_dim, name="second_last", activation=activation)(second_last)
    outputs = Dense(1)(second_last_drop)
    
    return Model(inputs=inputs, outputs=outputs)




def build_model(time_step_length,N_features):
    kernel_size = 2 # 10
    model = Sequential()

    model.add(Conv1D(100, kernel_size, activation='relu', input_shape=(time_step_length,N_features)))
    model.add(BatchNormalization())

    model.add(Conv1D(100, kernel_size, activation='relu'))


    model.add(Conv1D(160, kernel_size, activation='relu'))
    model.add(Conv1D(160, kernel_size, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(32, activation='relu', name="last_dense"))
    
    
    model.add(Dense(1, activation='linear'))
    print(model.summary())
    
    return model