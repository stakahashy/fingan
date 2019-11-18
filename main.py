import tensorflow as tf
from tensorflow import set_random_seed
from fin_data import data_generator
from keras.models import Sequential
from keras.layers.merge import Add
from keras.optimizers import Adam,Nadam
from models import discriminator_model, generator_model_cnn,generator_model_mlp,generator_model_mlp_cnn,generator_model_mlp_cnn_plus
import matplotlib.pyplot as plt
from stats import acf
import numpy as np
from numpy.random import seed
import visualize
import stylized_facts as sf
from datetime import datetime as dt
import os
from shutil import copyfile
import argparse 

parser = argparse.ArgumentParser(description='FIN-GAN implementation')
parser.add_argument('--batch_size',type=int,default=24)
parser.add_argument('--generator_model',type=str,default='mlp-cnn')
parser.add_argument('--epochs',type=int,default=10)
parser.add_argument('--batches',type=int,default=1024)
parser.add_argument('--folder_name',type=str,default='')
parser.add_argument('--generator_lr',type=float,default='2e-4')
parser.add_argument('--discriminator_lr',type=float,default='1e-5')
parser.add_argument('--log_interval',type=int,default=50)
parser.add_argument('--seed',type=int,default=1)

args = parser.parse_args()
seed(args.seed)
set_random_seed(args.seed)

dg = data_generator()
batch_size = 24
dg.batch_size = batch_size
batches = 1024
epochs = 10                 
timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
timestamp += '_'
timestamp += args.folder_name
os.mkdir('./imgs/%s'%(timestamp))
os.mkdir('./npy/%s'%(timestamp))
os.mkdir('./weights/%s'%(timestamp))
os.mkdir('./imgs/%s/acf'%(timestamp))
os.mkdir('./imgs/%s/dist'%(timestamp))
os.mkdir('./imgs/%s/time_series'%(timestamp))
os.mkdir('./imgs/%s/leverage'%(timestamp))
copyfile('./main.py','./imgs/%s/main.py'%(timestamp))
copyfile('./models.py','./imgs/%s/models.py'%(timestamp))
with open('./imgs/%s/hyper_parameters.txt'%(timestamp),'w') as w:
    w.write(str(args))
def train():
    #preparing generator
    if args.generator_model == 'mlp-cnn':
        generator,generator_statistics = generator_model_mlp_cnn()
    elif args.generator_model == 'mlp':
        generator,generator_statistics = generator_model_mlp()
    elif args.generator_model == 'cnn':
        generator = generator_model_cnn()
    elif args.generator_model == 'plus':
        generator = generator_model_mlp_cnn_plus()
    else:
        import sys
        sys.exit()
    #preparing discriminator
                                                      
    statistics_opt = Adam(lr=0.0001)
    generator_statistics.compile(loss='mean_squared_error',optimizer=statistics_opt)

    discriminator = discriminator_model()
    d_opt = Adam(lr=args.discriminator_lr, beta_1=0.1)
    discriminator.compile(loss='binary_crossentropy', optimizer=d_opt)
    discriminator.trainable = False
    for e in discriminator.layers:
        e.trainable = False
    gan = Sequential([generator,discriminator])
    g_opt = Adam(lr=args.generator_lr, beta_1=0.5)
    gan.compile(loss='binary_crossentropy', optimizer=g_opt)
    
    g_loss_recorder = []
    d_loss_recorder = []
    g_losses_recorder = []
    d_losses_recorder = []
    #start training
    for epoch in range(args.epochs):
        for index in range(args.batches):
            noise = np.array([np.random.normal(0,1.0,size=100) for _ in range(batch_size)])
            real_series = dg.real_data()
            real_series = np.nan_to_num(real_series)
            generated_series = generator.predict(noise, verbose=0)
            if index % args.log_interval == 0:
                sf.acf(generated_series,'./imgs/%s/acf/acf_abs_%i_%i'%(timestamp,epoch,index),for_abs=True)
                sf.acf(generated_series,'./imgs/%s/acf/acf_raw_%i_%i'%(timestamp,epoch,index),for_abs=False)
                sf.acf(generated_series,'./imgs/%s/acf/acf_abs_linear_%i_%i'%(timestamp,epoch,index),for_abs=True,scale='linear')
                sf.acf(generated_series,'./imgs/%s/acf/acf_raw_linear_%i_%i'%(timestamp,epoch,index),for_abs=False,scale='linear')
                sf.leverage_effect(generated_series,'./imgs/%s/leverage/leverage_%i_%i'%(timestamp,epoch,index))
                sf.distribution(generated_series, './imgs/%s/dist/distribution_%i_%i'%(timestamp,epoch,index),'linear')
                sf.distribution(generated_series, './imgs/%s/dist/distribution_%i_%i'%(timestamp,epoch,index),'log')
                visualize.time_series(generated_series[0],'./imgs/%s/time_series/generated_time_series_%i_%i'%(timestamp,epoch,index))
                np.save('./npy/%s/generated_time_series_%i_%i.npy'%(timestamp,epoch,index),generated_series)
            # update discriminator
            X = np.concatenate((real_series, generated_series))
            y = np.concatenate([np.random.uniform(0.9,1.1,batch_size),np.random.uniform(0.1,0.3,batch_size)])
            d_loss = discriminator.train_on_batch(X, y)
            d_loss_recorder.append(d_loss)
            # update generator
            y = np.array([1.]*batch_size,dtype=np.float32)
            g_loss = gan.train_on_batch(noise, y)
            g_loss_recorder.append(g_loss)
            print("epoch: %d, batch: %d, g_loss: %f, d_loss: %f" % (epoch, index, g_loss, d_loss))
            generator.save_weights('./weights/%s/generator_%i_%i.h5'%(timestamp,epoch,index))
train()
