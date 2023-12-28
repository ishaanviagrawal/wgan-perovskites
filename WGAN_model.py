import numpy as np
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Input
from numpy import mean
from numpy import ones
from numpy.random import randn
from keras import backend
from keras.optimizers import RMSprop
from keras.initializers import RandomNormal
from keras.constraints import Constraint
from keras import load_model
import warnings
warnings.filterwarnings("ignore")


class ClipConstraint(Constraint):
    '''
    Class to perform gradient clipping while training the model
    '''
    def __init__(self, clip_val):
        self.clip_val = clip_val

    def __call__(self, weights):
        return backend.clip(weights, -self.clip_val, self.clip_val)
 
    def get_config(self):
        return {'clip_val': self.clip_val}


class WGAN:

    def __init__(self, len, latent_dim):
        self.len = len
        self.latent_dim = latent_dim

    def gen_noise_vector(self, n_samples):
        '''
        Generating the random noise vector/latent vector that is given as input to the generator
        '''
        x = randn(self.latent_dim * n_samples)
        x = x.reshape(n_samples, self.latent_dim)
        return x

    def gen_fake_vector(self, generator, n_samples):
        '''
        Returns the fake samples created by the generator
        '''
        x = self.gen_noise_vector(n_samples)
        X = generator.predict(x)
        y = np.ones((n_samples, 1))
        return X, y

    def gen_real_vector(self, n, data):
        '''
        Randomly sampling real samples from the training data
        '''
        X = data.sample(n)
        y = -np.ones((n, 1))
        return X, y
    
    def wasserstein_loss(self, y_true, y_pred):
        return backend.mean(y_true * y_pred)
    
    def critic(self):
        weight_initial = RandomNormal(stddev=0.02)
        weight_constraint = ClipConstraint(0.01)
        
        model = Sequential()
        model.add(Input(shape=(self.len)))
        model.add(Dense(32, activation = 'relu', kernel_initializer=weight_initial, kernel_constraint=weight_constraint))
        model.add(BatchNormalization())
        model.add(Dense(64, activation = 'relu', kernel_initializer=weight_initial, kernel_constraint=weight_constraint))
        model.add(BatchNormalization())
        model.add(Dense(128, activation = 'relu', kernel_initializer=weight_initial, kernel_constraint=weight_constraint))
        model.add(BatchNormalization())
        model.add(Dense(256, activation = 'relu', kernel_initializer=weight_initial, kernel_constraint=weight_constraint))
        model.add(BatchNormalization())
        model.add(Dense(1))
        model.compile(loss = self.wasserstein_loss, optimizer = RMSprop(lr=0.00001))
        return model
    
    def generator(self):
        weight_initial = RandomNormal(stddev=0.02)
        
        model = Sequential()
        model.add(Input(shape=(self.latent_dim,)))
        model.add(Dense(256, activation = 'relu', kernel_initializer=weight_initial))
        model.add(Dense(128, activation = 'relu', kernel_initializer=weight_initial))
        model.add(Dense(64, activation = 'relu', kernel_initializer=weight_initial))
        model.add(Dense(32, activation = 'relu', kernel_initializer=weight_initial))
        model.add(Dense(self.len, kernel_initializer=weight_initial))
        return model
    
    def wgan(self, gen, critic):
        for layer in critic.layers:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = False

        model = Sequential()
        model.add(gen)
        model.add(critic)

        optimizer = RMSprop(lr=0.00001)
        model.compile(loss = self.wasserstein_loss, optimizer = optimizer)
        return model

    def training_loop(self, generator, critic, wgan, data, epochs=10, n_batch=1000, n_critic=5):
        '''
        Training loop for the WGAN, for every time the generator is updated, the critic is updated 5 times
        '''
        epoch_batch_num = int(data.shape[0] / n_batch)
        n_steps = epoch_batch_num * epochs
        half_batch = int(n_batch / 2)
        c1 = []
        c2 = []
        g = []

        for i in range(n_steps):
            c1_temp = []
            c2_temp = []

            for _ in range(n_critic):
                #Generating the real and fake samples to give to the critic
                X_real, y_real = self.gen_real_vector(half_batch, data)
                X_fake, y_fake = self.gen_fake_vector(generator, half_batch)

                #Critic is trained on the real and fake samples
                c1_loss = critic.train_on_batch(X_real, y_real)
                c1_temp.append(c1_loss)
                c2_loss = critic.train_on_batch(X_fake, y_fake)
                c2_temp.append(c2_loss)
            
            c1.append(mean(c1_temp))
            c2.append(mean(c2_temp))

            X_gan = self.gen_noise_vector(n_batch)
            y_gan = -ones((n_batch, 1))

            #training the combined generator and discriminator
            g_loss = wgan.train_on_batch(X_gan, y_gan)
            g.append(g_loss)

        return generator
    
    def train_wgan(self, data, path, len_data=15):
        '''
        Function to train and save a new model
        data: training dataset
        path: file name for the model to be saved
        len_data: length of the composition vector (14-dim composition vector + phase)
        '''
        critic = self.critic()
        generator = self.generator()
        gan_model = self.wgan(generator, critic)
        if len_data<5000:
            model = self.training_loop(generator, critic, gan_model, data, n_batch = 64)
        else:
            model = self.training_loop(generator, critic, gan_model, data, n_batch = 3000)
        model.save(path)
        return model
    
    def gen_phase(self, data_gen, n_gen):
        '''
        Calculates the final phase for all the compounds
        data_gen: data generated by the model
        n_gen: total number of generated compounds

        '''
        for i in range(n_gen):
            data_gen.loc[i, 'Phase'] = math.floor(data_gen.loc[i, 'Phase'])
            if data_gen.loc[i, 'Phase'] < 1:
                data_gen.loc[i, 'Phase'] = 1

        data_gen['Phase'] = data_gen['Phase'].astype(int)
        return data_gen

    def clean_comp(self, data_gen, n_gen):
        '''
        n_gen: number of generated samples
        Ensures that the sum of fractions of all A, B, and X site species present is 1
        '''
        for i in range(n_gen):
            for a in ['K', 'Rb', 'Cs', 'MA', 'FA', 'Ca', 'Sr', 'Ba', 'Ge', 'Sn', 'Pb','Cl', 'Br', 'I']:
                if data_gen.loc[i, a] < 0:
                    data_gen.loc[i, a] = abs(data_gen.loc[i, a])

            sum_A = data_gen.loc[i, "K"] + data_gen.loc[i, "Rb"] + data_gen.loc[i, "Cs"] + data_gen.loc[i, "MA"] + data_gen.loc[i, "FA"]
            sum_B = data_gen.loc[i, "Ca"] + data_gen.loc[i, "Sr"] + data_gen.loc[i, "Ba"] + data_gen.loc[i, "Ge"] + data_gen.loc[i, "Sn"] \
                + data_gen.loc[i, "Pb"]
            sum_X = data_gen.loc[i, "Cl"] + data_gen.loc[i, "Br"] + data_gen.loc[i, "I"]

            for k in ['K', 'Rb', 'Cs', 'MA', 'FA']:
                data_gen.loc[i, k] = data_gen.loc[i, k]/sum_A

            for k in ['Ca', 'Sr', 'Ba', 'Ge', 'Sn', 'Pb']:
                data_gen.loc[i, k] = data_gen.loc[i, k]/sum_B
            
            for k in ['Cl', 'Br', 'I']:
                data_gen.loc[i, k] = data_gen.loc[i, k]/sum_X

            sum_A = data_gen.loc[i, "K"] + data_gen.loc[i, "Rb"] + data_gen.loc[i, "Cs"] + data_gen.loc[i, "MA"] + data_gen.loc[i, "FA"]
            sum_B = data_gen.loc[i, "Ca"] + data_gen.loc[i, "Sr"] + data_gen.loc[i, "Ba"] + data_gen.loc[i, "Ge"] + data_gen.loc[i, "Sn"] \
                + data_gen.loc[i, "Pb"]
            sum_X = data_gen.loc[i, "Cl"] + data_gen.loc[i, "Br"] + data_gen.loc[i, "I"]

            if sum_A != 1:
                data_gen.loc[i, 'K'] = 1.00000000 - (data_gen.loc[i, "Rb"] + data_gen.loc[i, "Cs"] + data_gen.loc[i, "MA"] + data_gen.loc[i, "FA"])

            if sum_B != 1:
                data_gen.loc[i, 'Pb'] = 1.00000000 - (data_gen.loc[i, "Ca"] + data_gen.loc[i, "Sr"] + data_gen.loc[i, "Ba"] + data_gen.loc[i, "Ge"] \
                                                        + data_gen.loc[i, "Sn"])

            if sum_X != 1:
                data_gen.loc[i, 'Cl'] = 1.00000000 - (data_gen.loc[i, "Br"] + data_gen.loc[i, "I"])

        return data_gen
    
    def calculate_properties(self, data_gen, n_gen):
        '''
        Calulates the properties of A/B/X site species
        '''
        A_list = ['A_ion_rad', 'A_BP', 'A_MP', 'A_dens', 'A_at_wt', 'A_EA', 'A_IE', 'A_hof', 'A_hov', 'A_En', 'A_at_num', 'A_period']
        B_list = ['B_ion_rad', 'B_BP', 'B_MP', 'B_dens', 'B_at_wt', 'B_EA', 'B_IE', 'B_hof', 'B_hov', 'B_En', 'B_at_num', 'B_period']
        X_list = ['X_ion_rad', 'X_BP', 'X_MP', 'X_dens', 'X_at_wt', 'X_EA', 'X_IE', 'X_hof', 'X_hov', 'X_En', 'X_at_num', 'X_period']

        sp_properties = pd.read_csv('C:/BTP/WGAN_CompositionVector/species_properties.csv')
        sp_properties = sp_properties.set_index("element")
        sp_properties.set_axis([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], axis='columns', inplace=True)

        for i in range(n_gen):

            for k in range(len(A_list)):
                data_gen.loc[i, A_list[k]] = data_gen.loc[i, "K"]*sp_properties.loc["K", k] + data_gen.loc[i, "Rb"]*sp_properties.loc["Rb", k] \
                    + data_gen.loc[i, "Cs"]*sp_properties.loc["Cs", k] + data_gen.loc[i, "FA"]*sp_properties.loc["FA", k] \
                        + data_gen.loc[i, "MA"]*sp_properties.loc["MA", k]

            for k in range(len(B_list)):
                data_gen.loc[i, B_list[k]] = data_gen.loc[i, "Ca"]*sp_properties.loc["Ca", k] + data_gen.loc[i, "Sr"]*sp_properties.loc["Sr", k]\
                      + data_gen.loc[i, "Ge"]*sp_properties.loc["Ge", k] + data_gen.loc[i, "Ba"]*sp_properties.loc["Ba", k] + \
                        data_gen.loc[i, "Sn"]*sp_properties.loc["Sn", k] + data_gen.loc[i, "Pb"]*sp_properties.loc["Pb", k]

            for k in range(len(X_list)):
                data_gen.loc[i, X_list[k]] = data_gen.loc[i, "Cl"]*sp_properties.loc["Cl", k] + data_gen.loc[i, "Br"]*sp_properties.loc["Br", k] \
                    + data_gen.loc[i, "I"]*sp_properties.loc["I", k]

        return data_gen
    
    def gen_novel_comps(self, train_data, gen=load_model("model.h5"), n_gen=1000):
        '''
        Function to generate novel compositions with calculated properties given a trained model
        train_data: training dataset
        n_gen: total number of compositions to be generated
        '''
        noise_vector = self.gen_noise_vector(n_gen)
        X = gen.predict(noise_vector)
        comp_vector = pd.DataFrame(data=X,  columns=train_data.columns)
        comp_vector = self.gen_phase(comp_vector, n_gen)
        comp_vector = self.clean_comp(comp_vector, n_gen)
        comp_vector = self.calculate_properties(comp_vector, n_gen)
        return comp_vector