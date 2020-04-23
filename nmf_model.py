import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import regularizers

import os

class NMFModel(object):
    def get_gmf_model(self, num_factors=16):
        # input layer
        user_input_layer=layers.Input(shape=(1,), dtype='int32', name='user_input')
        item_input_layer=layers.Input(shape=(1,), dtype='int32', name='item_input')

        # embedding layer
        user_embedding_layer=layers.Embedding(input_dim=self.num_users, 
                               output_dim=num_factors, 
                               embeddings_regularizer=regularizers.l2(0.),
                               name='user_embedding'
                              )(user_input_layer)
        item_embedding_layer=layers.Embedding(input_dim=self.num_items, 
                               output_dim=num_factors, 
                               embeddings_regularizer=regularizers.l2(0.),
                               name='item_embedding'
                              )(item_input_layer)

        # flatten embedding vector
        user_latent=layers.Flatten()(user_embedding_layer)
        item_latent=layers.Flatten()(item_embedding_layer)

        # element wise product
        ew_product=layers.multiply([user_latent, item_latent])
        ew_product=layers.BatchNormalization()(ew_product)

        dense=layers.Dense(128, activation='tanh')(ew_product)
        prediction=layers.Dense(1, activation='tanh', name='prediction')(dense)

        self.gmf_model=Model([user_input_layer, item_input_layer], prediction)
        
        return self.gmf_model
    
    def get_mlp_model(self, layer1_dim, num_layers):
        # input layer
        user_input_layer=layers.Input(shape=(1,), dtype='int32', name='user_input')
        item_input_layer=layers.Input(shape=(1,), dtype='int32', name='item_input')

        # embedding layer
        user_embedding_layer=layers.Embedding(input_dim=self.num_users, 
                               output_dim=num_factors, 
                               embeddings_regularizer=regularizers.l2(0.),
                               name='user_embedding'
                              )(user_input_layer)
        item_embedding_layer=layers.Embedding(input_dim=self.num_items, 
                               output_dim=num_factors, 
                               embeddings_regularizer=regularizers.l2(0.),
                               name='item_embedding'
                              )(item_input_layer)
        
        # flatten embedding vector
        user_latent=layers.Flatten()(user_embedding_layer)
        item_latent=layers.Flatten()(item_embedding_layer)
        
        # concat
        x=layers.concatenate([user_latent, item_latent])
        for i in range(self.num_layers-1):
            x=Dense((int)(layer1_dim/(2**i)), activation='tanh', name='layer%s' %str(i+1))(x)
            x=BatchNormalization()(x)
        prediction=Dense(1, activation='tanh', name='prediction')(x)
        self.mlp_model=Model(input=[user_input_layer, item_input_layer], output='prediction')(x)
        
        return mlp_model
    
    def fit_model(self, model, checkpoint_path, n_batch_size=256, n_epochs=5):
        model.compile(loss='mae', optimiser='adam', metrics=['mae', 'mse'])
        
        cp_callback=tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                       save_weights_only=True,
                                                       verbose=1)
        hist=model.fit([np.array(self.user_train_input), np.array(self.item_train_input)], 
                  np.array(self.label_train_input),
                  batch_size=n_batch_size,
                  epochs=n_epochs,
                  callbacks = [cp_callback]
                 )
        return hist
    
    def predict(self, model, user_test_input, item_test_input, label_test_input):
        test_predictions = model.predict([np.array(user_test_input), np.array(item_test_input)]).flatten()
        return test_predictions
    
    def cal_result(self, pred_result, label_test_input):
        ans=0
        for i in range(len(test_predictions)):
            if label_test_input==0:
                continue
            elif label_test_input[i]==1 and test_predictions[i]>0:
                ans+=1
            elif label_test_input[i]==-1 and test_predictions[i]<0:
                ans+=1
            else:
                ans-=1
        return ans

    def __init__(self, user_train_input, item_train_input, train_labels, num_users, num_items):
        self.user_train_input=user_train_input
        self.item_train_input=item_train_input
        self.label_train_input=train_labels
        self.num_users=num_users
        self.num_items=num_items
        
        
        
        
        

