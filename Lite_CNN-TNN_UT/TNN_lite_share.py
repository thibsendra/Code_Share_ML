# -*- coding: latin-1 -*-
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback
from tensorflow.keras.losses import KLDivergence, MeanSquaredError
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.io import loadmat
from scipy.io import savemat
from scipy.signal import find_peaks

#%% Libérer la mémoire 
tf.keras.backend.clear_session() 

########################## Fixation de la seed ##########################

seed_value=188404475 # Choisis un entier pour la seed

# Fixer la seed pour TensorFlow
tf.random.set_seed(seed_value)

# Fixer la seed pour NumPy
np.random.seed(seed_value)

# Fixer la seed pour le générateur aléatoire de Python
random.seed(seed_value)

# Fixer la seed pour le hashage Python
os.environ['PYTHONHASHSEED'] = str(seed_value)

########################## Paramètres ##########################

gpu = 0            # Numéro du gpu utilisé
sim_size = 5000    # Nombre de données simulées (max 10000)
exp_size = 1200    # Nombre de données expérimentales (max 1200)

#Choisir le gpu
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_visible_devices(physical_devices[gpu], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[gpu], True)

########################## Import des dataset ##########################

# Construction du dataset input
Y1 = loadmat("./datasets/Dataset_real_signal_mean_train_indus.mat") 
Y1 = np.array(Y1['signal_train'])
Y2 = loadmat("./datasets/Dataset_signal_sim_indus.mat") 
Y2 = np.array(Y2['Y_tot'])
Y1 = Y1[0:exp_size,0:2000]
Y2 = Y2[0:sim_size,0:2000]
Y_input = np.concatenate((Y1, Y2))
N = len(Y_input[0])
M = len(Y_input)
Y_input = Y_input.reshape(M,N,1)

# Construction du dataset prediction
Y1_p = loadmat("./datasets/Dataset_real_deconv_mean_train_indus.mat") 
Y1_p = np.array(Y1_p['deconv_train'])
Y2_p = loadmat("./datasets/Dataset_deconv_mean_sim_indus.mat")
Y2_p = np.array(Y2_p['Deconv_mean'])
Y1_p = Y1_p[0:exp_size,0:2000]
Y2_p = Y2_p[0:sim_size,0:2000] 
Y_predicted = np.concatenate((Y1_p, Y2_p))
Y_predicted = Y_predicted.reshape(M,N,1)
Y_predicted = 1000*Y_predicted + 1

# Conversion des inputs et prédictions en dataset tensorflow
dataset = tf.data.Dataset.from_tensor_slices((Y_input, Y_predicted))

# Mélange du dataset
dataset = dataset.shuffle(buffer_size=len(Y_input))

# Batch le dataset en 100 passes par epoch
batch_size = round((exp_size + sim_size)/100)  # You can adjust this based on your memory constraints
batched_dataset = dataset.batch(batch_size)
num_incr = round(len(Y_input)/batch_size)

########################## Import des datasets test ##########################

# Input du dataset test
Y_input1 = loadmat("./datasets/Dataset_real_signal_mean_test_indus.mat")
Y_input1 = np.array(Y_input1['signal_test'])
Y_input1 = Y_input1[:,0:2000]
N1 = len(Y_input1[0])
M1 = len(Y_input1)
Y_input1 = Y_input1.reshape(M1,N1,1)

# Prédiction du dataset test
Y_predicted1 = loadmat("./datasets/Dataset_real_deconv_mean_test_indus.mat")
Y_predicted1 = np.array(Y_predicted1['deconv_test'])
Y_predicted1 = Y_predicted1[:,0:2000]
Y_predicted1 = Y_predicted1.reshape(M1,N1,1)
Y_predicted1 = Y_predicted1*1000

########################## Définition de la Custom Loss (success rate) ##########################

def detect_two_highest_peaks(signal, distance=30):
    signal = signal.reshape(2000)
    peaks, _ = find_peaks(signal[0:1980], distance=distance)
    if len(peaks) < 2:
        return None  # Retourne None si moins de 2 pics sont trouvés
    indices_T = sorted(peaks, key=lambda i: signal[i], reverse=True)
    return np.array(indices_T[:2], dtype=int)  # Retourne les positions des 2 plus hauts pics triés par position

# Paramètres du bloc du dataset test
Cl=5932.07
dt = 1/(100*10**6)
pre = 0.5

# Fonction custom loss appliqué lors de l'entrainement
def custom_peak_loss(y_true, y_pred):
    def py_custom_peak_loss(y_true, y_pred):
        total_diff = 0.0
        count=0.0
        batch_size = y_true.shape[0]
        for i in range(batch_size):
            true_signal = y_true[i].numpy()
            pred_signal = y_pred[i].numpy()
            
            true_peaks = detect_two_highest_peaks(true_signal)
            pred_peaks = detect_two_highest_peaks(pred_signal)
            
            if true_peaks is None or pred_peaks is None:
                continue  # Ignore si moins de 2 pics sont trouvés
            
            # Calculer la différence en valeur absolue des positions des pics
            diff1 = np.abs(max(true_peaks)-min(true_peaks))
            diff2 = np.abs(max(pred_peaks)-min(pred_peaks))
            d_diff = np.abs(diff1 - diff2)*Cl*dt/2*1000
            
            # Les 59 premières données sont à 2.25MHz puis le reste à 5MHz
            if i<59:
              lbd = Cl/(2.25e6)*1000
            else:
              lbd = Cl/(5e6)*1000
            
            # Condition de succès
            if d_diff<lbd*pre:
              count+=1
        
        #return Nombre_de_succès / Nombre_de_données
        return count/batch_size
    
    return tf.py_function(func=py_custom_peak_loss, inp=[y_true, y_pred], Tout=tf.float32)

# Utilisation de la fonction de loss personnalisée
@tf.function
def custom_loss(y_true, y_pred):
    return custom_peak_loss(y_true, y_pred)

########################## Fonction du suivi de l'entrainement (génère un plot toutes les 2 epoch) ##########################

def generate_and_save_prediction(epoch):

    # Générer une prédiction avec votre modèle sur un exemple de test
    key = 604
    test_example = Y_input1[key]
    prediction = model.predict(tf.expand_dims(test_example, axis=0))[0]
    if (epoch+1)%2==0:
    # Afficher et sauvegarder l'image de prédiction
      plt.plot(test_example,'b', alpha=0.5)
      #plt.plot(Y_predicted1[key]/1000,'r',alpha=0.5)
      plt.plot(prediction/max(prediction),'g')
      plt.title(f'Epoch {epoch + 1} - Prediction')
      if (epoch+1)<10:
        plt.savefig(os.path.join('./predictions/', f'prediction_epoch_0{epoch + 1}.png'))
      else:
        plt.savefig(os.path.join('./predictions/', f'prediction_epoch_{epoch + 1}.png'))
      plt.close()

########################## Définition du modèle ##########################

# Block Transformer
def transformer_block(inputs, input_shape, num_heads, mlp_units, dropout=0.3):
    mlp = int(mlp_units/2)
    inp = tf.transpose(inputs, perm=[0,2,1])
    x1 = layers.Lambda(lambda x: x[:, :mlp, :])(inp)
    x2 = layers.Lambda(lambda x: x[:, mlp:, :])(inp)
    
    # Self Multi Head Attention module    
    x1 = layers.MultiHeadAttention(num_heads=num_heads, dropout=0.2, key_dim = int(mlp/num_heads), value_dim=int(mlp/num_heads))(x1,x1) # attn_output
    
    # Parallel convolution
    x2 = tf.transpose(x2, perm=[0,2,1])
    x2 = layers.Conv1D(filters=mlp, kernel_size=61, padding='same', kernel_initializer='he_uniform')(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.ReLU()(x2)
    x2 = layers.Dropout(dropout)(x2) 
    x2 = tf.transpose(x2, perm=[0,2,1])
    
    # End of SMHA module
    attn_output = layers.Concatenate(axis=1)([x1, x2])
    temp = layers.Add()([attn_output, inp]) # attn_output
    out1 = layers.LayerNormalization(epsilon=1e-6)(temp) ## LayerNorm
    
    # FFN module
    test = tf.transpose(out1, perm=[0,2,1]) ## delete
    ff_output = layers.Dense(4*mlp_units, kernel_initializer='he_uniform')(test) ## out2 & mlp_u
    ff_output = tf.keras.layers.ReLU()(ff_output) 
    ff_output = layers.Dropout(dropout)(ff_output)
    ff_output = layers.Dense(mlp_units, kernel_initializer='he_uniform')(ff_output) ## mlp_u
    ff_output = layers.Dropout(dropout)(ff_output)
    ff_output = tf.transpose(ff_output, perm=[0,2,1]) ## delete
    temp3 = layers.Add()([out1, ff_output])
    x = layers.LayerNormalization(epsilon=1e-6)(temp3) ## LayerNorm
    x = tf.transpose(x, perm=[0,2,1])
    return x

# Fonction pour la construction du modèle
def build_model(input_shape, num_heads, num_transformer_blocks, mlp_units, dropout=0.3, mlp_dropout=0.3):

    inputs = tf.keras.Input(shape=input_shape)
    
    # Pre-processing
    x = layers.Conv1D(filters=mlp_units, kernel_size=61, padding='same', kernel_initializer='he_uniform')(inputs) 
    x = layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = layers.Dropout(mlp_dropout)(x)

    # Transformer blocks
    for _ in range(num_transformer_blocks):
        x = transformer_block(x, input_shape, num_heads, mlp_units, dropout)
 
    # Post-processing
    x = layers.Conv1D(filters=1, kernel_size=1, padding='same')(x)
    x = tf.keras.activations.tanh(x)
    x = layers.Conv1D(filters=1, kernel_size=61, padding='same', kernel_initializer='he_uniform')(x) 
    x = tf.keras.activations.relu(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

########################## Fonction de perte personnalisée R-Drop ##########################
class RDropLoss(tf.keras.losses.Loss):
    def __init__(self, lambda_rdrop=1.0, **kwargs):
        super().__init__(**kwargs)
        self.lambda_rdrop = lambda_rdrop
        self.kl_loss = KLDivergence()
        self.mse_loss = MeanSquaredError()
    
    def call(self, y_true, y_pred):
        # Calcul la perte MSE (L2) pour deux prédictions
        y_pred1, y_pred2 = y_pred
        
        mse_loss1 = self.mse_loss(y_true, y_pred1)
        mse_loss2 = self.mse_loss(y_true, y_pred2)
        
        # Calcul la divergence de perte KL entre les deux prédictions
        kl_loss = self.kl_loss(y_pred1, y_pred2)

        # Combine les pertes (MSE + KL Divergence)
        total_loss = mse_loss1/2 + mse_loss2/2 + self.lambda_rdrop * kl_loss
        return total_loss

# Etape d'entrainement custom pour R-drop
@tf.function
def train_step(model, inputs, targets, optimizer, loss_fn):
    with tf.GradientTape() as tape:
        # Passe deux fois avec un dropout différent
        preds1 = model(inputs, training=True)
        preds2 = model(inputs, training=True)
        
        # Calcul la perte (MSE + KL divergence)
        loss = loss_fn(targets, (preds1, preds2))
    
    # Réalise la descente de gradient et update le modèle
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

########################## Fonction de validation du modèle ##########################
@tf.function
def validation_step(model, inputs, targets, custom_loss_fn):
    preds = model(inputs, training=False)  # Sans dropout lors de la validation
    loss = custom_loss_fn(targets, preds)
    return loss

val_loss_list = []

########################## Construction du modèle ##########################

model = build_model(input_shape=(N, 1), num_heads=1, mlp_units=16, num_transformer_blocks=1)
model.summary()
loss_fn = RDropLoss(lambda_rdrop=0.5)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-09)
  
########################## Entrainement ##########################

# Compilation du modèle
epochs = 50  # Nombre d'epoch

# Initialisation des compteurs
count=0
best_val_loss = 0.0
val_custom_loss = []
e = 10 # Nombre d'epoch lors du warmup

# Boucle d'entrainement (model.fit)
for epoch in range(epochs):
      # Courbe du learning rate avec warmup
      if epoch+1 < e:
        new_learning_rate = 0.0001 +0.0001 * (epoch) 
      else:
        new_learning_rate = (0.0001 +0.0001 * e) / (1 + 0.1*(epoch+1-e))
      optimizer.learning_rate.assign(new_learning_rate)
      print(f'Epoch {epoch+1}/{epochs} - Learning rate = {new_learning_rate}') 
      
      # Pour chaque step de taille batch_size 
      for step, (batch_inputs, batch_targets) in enumerate(batched_dataset):
          
          # On entraine sur le batch actuel
          loss = train_step(model, batch_inputs, batch_targets, optimizer, loss_fn)
          
          # On print la loss pour ce batch
          sys.stdout.write(f'\rStep {step+1}/{num_incr}, Train Loss: {loss.numpy():.4f}')
          sys.stdout.flush()
      
      # A la fin de l'epoch, on évalue sur le dataset de validation  
      val_loss = validation_step(model, Y_input1, Y_predicted1, custom_peak_loss)
      val_custom_loss.append(val_loss)
      generate_and_save_prediction(epoch) 
      print(f'\nEpoch {epoch+1}, Validation Loss: {val_loss.numpy()}, seed value : {seed_value}')
      
      # On détermine si le success rate est meilleur que les époch précédentes et on sauvegarde la meilleur époch
      if val_loss > best_val_loss:
        print(f'Val loss improved from {best_val_loss} to {val_loss.numpy()} \n')
        best_val_loss = val_loss
        model.save("./models/TNN_BM_eddify.keras")
        count=1
      else:
        print(f'Val loss did not improve from {best_val_loss} \n')
        count+=1
      
      # Si le success rate na pas évolué pendant plus de x epoch, l'entrainement s'arrête
      if count>=15:
        break 
      val_loss_list.append(best_val_loss) #Liste pour plot ensuite la courbe du success rate
    
# Sauvegarde de la dernière époch
model.save("./models/TNN_rdrop.keras")
val_custom_loss=np.array(val_custom_loss) 
savemat('./models/TNN_indus_test1.mat', {'rate': val_custom_loss}) 

val_loss_list=np.array(val_loss_list)
print('Meilleur IA : ',max(val_loss_list))
  
########################## Libérer la mémoire ##########################
tf.keras.backend.clear_session()