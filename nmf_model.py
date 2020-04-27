import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import regularizers

import os

class NMFModel(object):
    def get_gmf_model(self, num_factors=16):
        self.num_factors=num_factors
        
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

        # 잘못만든듯 지워도 될 듯
        #dense=layers.Dense(128, activation='tanh')(ew_product)
        
        prediction=layers.Dense(1, activation='tanh', name='prediction')(ew_product)

        self.gmf_model=Model([user_input_layer, item_input_layer], prediction)
        
        return self.gmf_model
    
    def get_mlp_model(self, num_layers=3):
        #self.num_factors=16
        self.num_layers=num_layers
        self.layer1_dim=self.num_factors*(2**(num_layers-1))
        
        # input layer
        user_input_layer=layers.Input(shape=(1,), dtype='int32', name='user_input')
        item_input_layer=layers.Input(shape=(1,), dtype='int32', name='item_input')

        # embedding layer
        user_embedding_layer=layers.Embedding(input_dim=self.num_users, 
                               output_dim=(int)(self.layer1_dim/2), 
                               embeddings_regularizer=regularizers.l2(0.),
                               name='user_embedding'
                              )(user_input_layer)
        item_embedding_layer=layers.Embedding(input_dim=self.num_items, 
                               output_dim=(int)(self.layer1_dim/2), 
                               embeddings_regularizer=regularizers.l2(0.),
                               name='item_embedding'
                              )(item_input_layer)
        
        # flatten embedding vector
        user_latent=layers.Flatten()(user_embedding_layer)
        item_latent=layers.Flatten()(item_embedding_layer)
        
        # concat
        x=layers.concatenate([user_latent, item_latent])
        
        for i in range(self.num_layers-1):
            x=layers.Dense((int)(self.layer1_dim/(2**i)), activation='tanh', name='layer%s' %str(i+1))(x)
            x=layers.BatchNormalization()(x)
        prediction=layers.Dense(1, activation='tanh', name='prediction')(x)
        self.mlp_model=Model([user_input_layer, item_input_layer], prediction)
        
        return self.mlp_model
    
    def set_nmf_weight(self):

        gmf_model=self.gmf_model
        mlp_model=self.mlp_model
        model=self.nmf_model
        
        #GMF embedding
        gmf_user_embedding=gmf_model.get_layer('user_embedding').get_weights()
        gmf_item_embedding=gmf_model.get_layer('item_embedding').get_weights()
        model.get_layer('GMF_user_embedding').set_weights(gmf_user_embedding)
        model.get_layer('GMF_item_embedding').set_weights(gmf_item_embedding)

        #MLP embedding
        mlp_user_embedding=mlp_model.get_layer('user_embedding').get_weights()
        mlp_item_embedding=mlp_model.get_layer('item_embedding').get_weights()
        model.get_layer('MLP_user_embedding').set_weights(mlp_user_embedding)
        model.get_layer('MLP_item_embedding').set_weights(mlp_item_embedding)
        
        for i in range(self.num_layers-1):
            tmp=str(i+1)
            mlp_layer=mlp_model.get_layer('layer%s' %tmp).get_weights()
            model.get_layer('layer%s' %tmp).set_weights(mlp_layer)

        #prediction layer
        gmf_prediction=gmf_model.get_layer('prediction').get_weights()
        mlp_prediction=mlp_model.get_layer('prediction').get_weights()
        
        new_weights=np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis=0)
        new_b=gmf_prediction[1]+ mlp_prediction[1]
        model.get_layer('prediction').set_weights([0.5*new_weights, 0.5*new_b])
        
        self.nmf_model=model 
        
        return self.nmf_model
    
    def get_nmf_model(self):
        
        num_factors=self.num_factors
        layer1_dim=self.layer1_dim
        num_layers=self.num_layers
        user_input_layer=layers.Input(shape=(1,), dtype='int32', name='user_input')
        item_input_layer=layers.Input(shape=(1,), dtype='int32', name='item_input')
        
        num_users=self.num_users
        num_items=self.num_items
        
        # GMF embedding layer
        GMF_user_embedding=layers.Embedding(input_dim=num_users, output_dim=(int)(num_factors), embeddings_regularizer=regularizers.l2(0.), name='GMF_user_embedding', input_length=1)
        GMF_item_embedding=layers.Embedding(input_dim=num_items, output_dim=(int)(num_factors), embeddings_regularizer=regularizers.l2(0.), name='GMF_item_embedding', input_length=1)

        # MLP embedding layer
        MLP_user_embedding=layers.Embedding(input_dim=num_users, output_dim=(int)(layer1_dim/2), embeddings_regularizer=regularizers.l2(0.), name='MLP_user_embedding', input_length=1)
        MLP_item_embedding=layers.Embedding(input_dim=num_items, output_dim=(int)(layer1_dim/2), embeddings_regularizer=regularizers.l2(0.), name='MLP_item_embedding', input_length=1)

        #flatten GMF embedding vector
        GMF_user_latent=layers.Flatten()(GMF_user_embedding(user_input_layer))
        GMF_item_latent=layers.Flatten()(GMF_item_embedding(item_input_layer))

        # flatten MLP embeddding vector
        MLP_user_latent=layers.Flatten()(MLP_user_embedding(user_input_layer))
        MLP_item_latent=layers.Flatten()(MLP_item_embedding(item_input_layer))
        
        # gmf - element wise product
        GMF_vector=layers.multiply([GMF_user_latent, GMF_item_latent])
        GMF_vector=layers.BatchNormalization()(GMF_vector)
        
        # mlp
        MLP_vector=layers.concatenate([MLP_user_latent, MLP_item_latent])
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
    
    def load_weight(self, model_name, cp_path):
        assert model_name in ['gmf', 'mlp', 'nmf'], 'model name parameter must be gmf, mlp, nmf'
        if model_name =='gmf':
            self.gmf_model.load_weights(cp_path)
            return self.gmf_model
        elif model_name=='mlp':
            self.mlp_model.load_weights(cp_path)
            return self.mlp_model
        else: # model_name='nmf'
            self.nmf_model.load_weights(cp_path)
            return self.nmf_model
    
    def fit_model(self, model_name, checkpoint_path, n_batch_size=256, n_epochs=5):
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
        hist=model.fit([np.array(self.user_train_input), np.array(self.item_train_input)], 
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
    
    
    def print_graph(self, model_name):
        assert model_name in ['gmf', 'mlp', 'nmf'], 'model name parameter must be gmf, mlp, nmf'

        if model_name =='gmf':
            model=self.gmf_model
        elif model_name=='mlp':
            model=self.mlp_model
        else: # model_name='nmf'
            model=self.nmf_model
        model.summary()
        
    def __init__(self, user_train_input, item_train_input, train_labels, num_users, num_items, num_factors=16):
        self.user_train_input=user_train_input
        self.item_train_input=item_train_input
        self.label_train_input=train_labels
        self.num_users=num_users
        self.num_items=num_items
        self.num_factors=num_factors
        
        
        
        
        

