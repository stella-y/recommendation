from nmf_model import NMFModel

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import regularizers


class NMFWithMeta(NMFModel):
    def get_gmf_model(self, num_factors=16, num_movie_factors=8):

        user_input_layer=layers.Input(shape=(1,), dtype='int32', name='user_input')
        item_input_layer=layers.Input(shape=(1,), dtype='int32', name='item_input')
        genre_input_layer=layers.Input(shape=(None,), dtype='int32', name='genre_input')

        user_embedding_layer=layers.Embedding(input_dim=self.num_users, 
                               output_dim=num_factors, 
                               embeddings_regularizer=regularizers.l2(0.),
                               name='user_embedding'
                              )(user_input_layer)
        item_embedding_layer=layers.Embedding(input_dim=self.num_items, 
                               output_dim=self.num_movie_factors,
                               embeddings_regularizer=regularizers.l2(0.),
                               name='item_embedding'
                              )(item_input_layer)

        genre_embedding_layer=layers.Embedding(input_dim=self.num_genres,
                               output_dim=self.num_movie_factors,
                               mask_zero=True,
                               embeddings_regularizer=regularizers.l2(0.),
                               name='genre_embedding'
                              )(genre_input_layer)

        genre_emb_mean=tf.reduce_mean(genre_embedding_layer, 0)

        
        user_latent=layers.Flatten()(user_embedding_layer)
        item_latent=layers.Flatten()(item_embedding_layer)
        genre_latent=layers.Flatten()(genre_emb_mean)

        movie_latent=layers.concatenate([item_latent, genre_latent])
        
        ew_product=layers.multiply([user_latent, movie_latent])
        ew_product=layers.BatchNormalization()(ew_product)

        prediction=layers.Dense(1, activation='tanh', name='prediction')(ew_product)
        
        self.gmf_model=Model([user_input_layer, item_input_layer, genre_input_layer], prediction)
        
        return self.gmf_model
    
    def get_mlp_model(self, num_layers=3):
        #self.num_factors=16
        self.num_layers=num_layers
        self.layer1_dim=self.num_factors*(2**(num_layers-1))
        
        # input layer
        user_input_layer=layers.Input(shape=(1,), dtype='int32', name='user_input')
        item_input_layer=layers.Input(shape=(1,), dtype='int32', name='item_input')
        genre_input_layer=layers.Input(shape=(None,), dtype='int32', name='genre_input')

        user_embedding_layer=layers.Embedding(input_dim=self.num_users, 
                               output_dim=num_factors, 
                               embeddings_regularizer=regularizers.l2(0.),
                               name='user_embedding'
                              )(user_input_layer)
        item_embedding_layer=layers.Embedding(input_dim=self.num_items, 
                               output_dim=self.num_movie_factors,
                               embeddings_regularizer=regularizers.l2(0.),
                               name='item_embedding'
                              )(item_input_layer)

        genre_embedding_layer=layers.Embedding(input_dim=self.num_genres,
                               output_dim=self.num_movie_factors,
                               mask_zero=True,
                               embeddings_regularizer=regularizers.l2(0.),
                               name='genre_embedding'
                              )(genre_input_layer)

        genre_emb_mean=tf.reduce_mean(genre_embedding_layer, 0)

        
        user_latent=layers.Flatten()(user_embedding_layer)
        item_latent=layers.Flatten()(item_embedding_layer)
        genre_latent=layers.Flatten()(genre_emb_mean)

        movie_latent=layers.concatenate([item_latent, genre_latent])
        
        # concat
        x=layers.concatenate([user_latent, movie_latent])
        
        for i in range(self.num_layers-1):
            x=layers.Dense((int)(self.layer1_dim/(2**i)), activation='tanh', name='layer%s' %str(i+1))(x)
            x=layers.BatchNormalization()(x)
        prediction=layers.Dense(1, activation='tanh', name='prediction')(x)
        self.mlp_model=Model([user_input_layer, item_input_layer], prediction)
        
        return self.mlp_model
    
    def get_nmf_model(self):
        
        num_factors=self.num_factors
        layer1_dim=self.layer1_dim
        num_layers=self.num_layers
        
        num_users=self.num_users
        num_items=self.num_items
        num_genres=self.num_genres
        
        # input layer
        user_input_layer=layers.Input(shape=(1,), dtype='int32', name='user_input')
        item_input_layer=layers.Input(shape=(1,), dtype='int32', name='item_input')
        genre_input_layer=layers.Input(shape=(None,), dtype='int32', name='genre_input')

        # GMF embedding layer
        GMF_user_embedding=layers.Embedding(input_dim=num_users, 
                               output_dim=num_factors, 
                               embeddings_regularizer=regularizers.l2(0.),
                               name='GMF_user_embedding'
                              )(user_input_layer)
        GMF_item_embedding=layers.Embedding(input_dim=num_items, 
                               output_dim=self.num_movie_factors,
                               embeddings_regularizer=regularizers.l2(0.),
                               name='GMF_item_embedding'
                              )(item_input_layer)

        GMF_genre_embedding=layers.Embedding(input_dim=num_genres,
                               output_dim=self.num_movie_factors,
                               mask_zero=True,
                               embeddings_regularizer=regularizers.l2(0.),
                               name='GMF_genre_embedding'
                              )(genre_input_layer)
        
        GMF_genre_emb_mean=tf.reduce_mean(genre_embedding_layer, 0)
        
        # MLP embedding layer
        MLP_user_embedding=layers.Embedding(input_dim=num_users, 
                               output_dim=num_factors, 
                               embeddings_regularizer=regularizers.l2(0.),
                               name='MLP_user_embedding'
                              )(user_input_layer)
        MLP_item_embedding=layers.Embedding(input_dim=num_items, 
                               output_dim=self.num_movie_factors,
                               embeddings_regularizer=regularizers.l2(0.),
                               name='MLP_item_embedding'
                              )(item_input_layer)

        MLP_genre_embedding=layers.Embedding(input_dim=num_genres,
                               output_dim=self.num_movie_factors,
                               mask_zero=True,
                               embeddings_regularizer=regularizers.l2(0.),
                               name='MLP_genre_embedding'
                              )(genre_input_layer)
        MLP_genre_emb_mean=tf.reduce_mean(genre_embedding_layer, 0)

        # GMF
        GMF_user_latent=layers.Flatten()(GMF_user_embedding)
        GMF_item_latent=layers.Flatten()(GMF_item_embedding)
        GMF_genre_latent=layers.Flatten()(GMF_genre_emb_mean)

        GMF_movie_latent=layers.concatenate([GMF_item_latent, GMF_genre_latent])
        
        # MLP
        MLP_user_latent=layers.Flatten()(MLP_user_embedding)
        MLP_item_latent=layers.Flatten()(MLP_item_embedding)
        MLP_genre_latent=layers.Flatten()(MLP_genre_emb_mean)

        MLP_movie_latent=layers.concatenate([MLP_item_latent, MLP_genre_latent])
        
        
        # gmf - element wise product
        GMF_vector=layers.multiply([GMF_user_latent, GMF_movie_latent])
        GMF_vector=layers.BatchNormalization()(GMF_vector)
        
        # mlp
        MLP_vector=layers.concatenate([MLP_user_latent, MLP_movie_latent])
        MLP_vector=layers.BatchNormalization()(MLP_vector)
        
        for i in range(num_layers-1):
            MLP_vector=layers.Dense((int)(layer1_dim/(2**i)), activation='tanh', name='layer%s' %str(i+1))(MLP_vector)
            MLP_vector=layers.BatchNormalization()(MLP_vector)

        #NeuMF layer
        NeuMF_vector=layers.concatenate([GMF_vector, MLP_vector])
        prediction=layers.Dense(1, activation='tanh', name='prediction')(NeuMF_vector)
        
        model=Model([user_input_layer, item_input_layer], prediction)
        
        self.nmf_model=model
        self.nmf_model=self.set_nmf_weight()
        
        return self.nmf_model
    
    def fit_model(self, model_name, checkpoint_path, n_batch_size=10, n_epochs=5):
        assert model_name in ['gmf', 'mlp', 'nmf'], 'model name parameter must be gmf, mlp, nmf'

        if model_name =='gmf':
            model=self.gmf_model
        elif model_name=='mlp':
            model=self.mlp_model
        else: # model_name='nmf'
            model=self.nmf_model
        
        model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mse'])
        
        cp_callback=tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                       save_weights_only=True,
                                                       verbose=1)
        hist=model.fit([np.array(self.user_train_input), np.array(self.item_train_input), np.array(self.genre_train_input)], 
                  np.array(self.label_train_input),
                  batch_size=n_batch_size,
                  epochs=n_epochs,
                  callbacks = [cp_callback]
                 )
        return hist
    
    def predict(self, model_name, user_test_input, item_test_input):
        assert model_name in ['gmf', 'mlp', 'nmf'], 'model name parameter must be gmf, mlp, nmf'

        if model_name =='gmf':
            model=self.gmf_model
        elif model_name=='mlp':
            model=self.mlp_model
        else: # model_name='nmf'
            model=self.nmf_model
        
        pred_result = model.predict([np.array(user_test_input), np.array(item_test_input)]).flatten()
        return pred_result
    
    def __init__(self, user_train_input, item_train_input, genre_train_input, train_labels, num_users, num_items, num_genres, num_factors=16, num_movie_factors=8):
        self.user_train_input=user_train_input
        self.item_train_input=item_train_input
        self.genre_train_input=genre_train_input
        self.label_train_input=train_labels
        self.num_users=num_users
        self.num_items=num_items
        self.num_genres=num_genres
        self.num_movie_factors=num_movie_factors
        self.num_factors=num_factors
