from __future__ import print_function, division

import os
from re import I

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

from keras.datasets import mnist
import tensorflow as tf
from util_mnist import *
import emnist
from keras.layers import Dense, Reshape
from keras.layers import BatchNormalization, Activation
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential

import matplotlib.pyplot as plt

import numpy as np

class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = setting.height_data
        self.img_cols = setting.width_data
        self.channels = setting.depth
        self.latent_dim = setting.latent

        self.gen_input = tf.placeholder(tf.float32, [setting.batch_size, self.latent_dim], name='gen-input')
        self.disc_input = tf.placeholder(tf.float32,[setting.batch_size,setting.height_data,setting.width_data,setting.depth],name='inp')
        # Build the generator
        self.generator = self.build_generator()
        self.generator = self.generator(self.gen_input)

        # Build the discriminator
        self.r_logits, self.r_rep, self.r_prob = self.build_discriminator(self.disc_input)
        # Build the discrimintaor with same params but with the generator as input
        self.f_logits, self.g_rep, self.f_prob = self.build_discriminator(self.generator,reuse=True)
        self.t_dloss = []
        self.t_gloss = []
        self.t_acc = []
        self.t_epoch = []
        self.v_dloss = []
        self.v_gloss = []
        self.v_epoch = []
        self.v_acc = []




    def build_generator(self):
        with tf.variable_scope('gen'):
            model = Sequential()
            model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
            model.add(Reshape((7, 7, 128)))
            model.add(UpSampling2D())
            model.add(Conv2D(128, kernel_size=3, padding="same"))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation("relu"))
            model.add(UpSampling2D())
            model.add(Conv2D(64, kernel_size=3, padding="same"))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation("relu"))
            model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
            model.add(Activation("tanh"))
            
            #print(model.weights)

            return model

    def build_discriminator(self, input, reuse=False):
        with tf.variable_scope('disc', reuse=reuse):
            inp1=input
            out1_res=conv_ops(inp1,kernel=3,rules=64,outp=32,stride=1,padding='SAME',name='layer1')
            out2_res=conv_ops(out1_res,kernel=3,rules=64,outp=32,stride=1,padding='SAME',name='layer2')
            out3_res=pooling_ops(out2_res,kernel=2,rules=128,outp=32,name='layer3')
            out4_res=conv_ops(out3_res,kernel=3, rules=128,stride=1,outp=32, padding='SAME',name='layer4')
            out5_res=conv_ops(out4_res,kernel=3, rules=128,stride=1,outp=32, padding='SAME',name='layer5')
            out6_res=pooling_ops(out5_res,kernel=2, rules=128,outp=32,name='layer6')
            out_f=out6_res
            out_s=out_f.get_shape().as_list()
            out7_flat=tf.reshape(out_f,(out_s[0],-1))
            out8=dense_layer(out7_flat,shape_w=[out_s[1]*out_s[2]*out_s[3],512],shape_b=[512],name='dense')
            out8_1=tf.nn.relu(out8)
            prob = tf.placeholder(tf.float32, name='keep_prob')
            d1=tf.nn.dropout(out8_1, prob)
            d2=d1
            out_final=dense_layer(d2,shape_w=[512,setting.n_class],shape_b=[setting.n_class],name='last')
            pred=tf.nn.softmax(out_final)
            return pred, out_final, prob
    
    def train(self, epochs, dataset, save_interval=50):
        print('\n\nTraining ' + dataset +':')
        # Load the dataset
        if dataset == 'mnist':
            (X_train, _), (X_valid, _) = mnist.load_data()
        else:
            X_train, _ = emnist.extract_training_samples('balanced')
            X_valid, _ = emnist.extract_test_samples('balanced')
 
        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)
        X_valid = X_valid / 127.5 - 1.
        X_valid = np.expand_dims(X_valid, axis=3)
        np.random.shuffle(X_train)

        # One hot adversarial ground truths
        valid = np.ones((setting.batch_size, 1), dtype=np.int32)
        t = np.zeros((setting.batch_size, setting.n_class))
        t[np.arange(valid.size), valid] = 1.0
        valid = t
        fake = np.zeros((setting.batch_size, 1), dtype=np.int32)
        t = np.zeros((setting.batch_size, setting.n_class))
        t[np.arange(fake.size), fake] = 1.0
        fake = t
        
        # Generator loss caluclation
        gen_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.f_logits,labels=valid))
        gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="gen")
        gen_step = tf.train.AdamOptimizer(setting.gen_lr, setting.beta).minimize(gen_loss,var_list = gen_vars)

        # Discriminator loss, accuaracy, etc
        target=tf.placeholder(tf.int32,[setting.batch_size,setting.n_class],name='target')
        err1=tf.nn.softmax_cross_entropy_with_logits_v2(labels=target,logits=self.r_rep)
        err=tf.reduce_mean(err1)
        correct_pred=tf.equal(tf.argmax(self.r_logits,1),tf.argmax(target,1))
        accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
        global_step=tf.train.get_or_create_global_step()
        train_op=tf.train.AdamOptimizer(learning_rate=setting.disc_lr).minimize(err,global_step=global_step)

        gpu_options=tf.GPUOptions(visible_device_list="0")
        init=tf.global_variables_initializer()

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(init)
            for epoch in range(epochs+1):

                # Select a random half of images
                idx = np.random.randint(0, X_train.shape[0], setting.batch_size)
                imgs = X_train[idx]

                # Sample noise and generate a batch of new images
                noise = np.random.normal(0, 1, (setting.batch_size, self.latent_dim))
                
                # Generate images
                gen_imgs = sess.run([self.generator], feed_dict={self.gen_input: noise})[0]
                
                # ---------------------
                #  Train the discriminator in two steps, once on real images and once on generated images
                # ---------------------

                args={self.disc_input:imgs,target:valid,self.r_prob:setting.dropout}
                _,dr_loss,dr_acc,step1,pred_f,err11,logit11=sess.run([train_op,err,accuracy,global_step,correct_pred,err1,self.r_rep],feed_dict=args)
                args={self.disc_input:gen_imgs,target:fake,self.r_prob:setting.dropout}
                _,df_loss,df_acc,step1,pred_f,err11,logit11=sess.run([train_op,err,accuracy,global_step,correct_pred,err1,self.r_rep],feed_dict=args)
                d_loss = (dr_loss + df_loss) * 0.5
                d_acc = (dr_acc + df_acc) * 0.5

                # Train the generator 
                _, gloss = sess.run([gen_step, gen_loss], feed_dict={self.gen_input: noise, self.f_prob:setting.dropout})

                print ("%d\t[D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss, 100*d_acc, gloss))
                self.t_gloss.append(gloss)
                self.t_dloss.append(d_loss)
                self.t_epoch.append(epoch)
                self.t_acc.append(d_acc)
                # If at save interval => save generated image samples
                if epoch % save_interval == 0:
                    self.save_imgs(epoch, sess, dataset)

            # Test accuracy
            d_tloss = 0
            d_tacc = 0
            g_loss = 0
            i = 0
            print('\n\nTesting ' + dataset +':')
            # Test the network on validation set
            for x in range(int(len(X_valid)/ setting.batch_size)):
                imgs = X_valid[x:x+setting.batch_size]
                noise = np.random.normal(0, 1, (setting.batch_size, self.latent_dim))
                gen_imgs = sess.run([self.generator], feed_dict={self.gen_input: noise})[0]
                data, lables = createBatch(imgs, valid, gen_imgs, fake)

                args = {self.disc_input:data,target:lables,self.r_prob:1.0}
                loss,acc, _ = sess.run([err,accuracy,correct_pred],feed_dict=args)
                gloss = sess.run([gen_loss], feed_dict={self.gen_input: noise, self.f_prob:1.0})[0]
                d_tloss += loss
                d_tacc += acc
                g_loss += gloss
                i += 1

                self.v_gloss.append(gloss)
                self.v_dloss.append(loss)
                self.v_epoch.append(x)
                self.v_acc.append(acc)
                if x % save_interval == 0:
                    self.save_imgs(x, sess, dataset, True)
            d_tloss = d_tloss / i
            d_tacc = d_tacc / i
            g_loss = g_loss / i
            print('Network test result: [D loss: %f, acc.: %.2f%%] [G loss: %f]' % (d_tloss, 100*d_tacc, g_loss))
                


    def save_imgs(self, epoch, sess, dataset, trained=False):
        r, c = 4, 4
        noise = np.random.normal(0, 1, (setting.batch_size, self.latent_dim))
        g = sess.run([self.generator], feed_dict={self.gen_input: noise})[0]

        # Rescale images 0 - 1
        gen_imgs = 0.5 * g + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        if trained:
            fig.savefig("images_trained_" + dataset + "/" + dataset + "_%d.png" % epoch)
        else:
            fig.savefig("images_" + dataset + "/" + dataset + "_%d.png" % epoch)
        plt.close()
    
    def plotGraphLossTraning(self):
        plt.plot(self.t_epoch, self.t_dloss, label='Dloss')
        plt.plot(self.t_epoch, self.t_gloss, label='Gloss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Traning Loss')
        plt.legend()
        plt.show()
    
    def plotGraphLossValidation(self):
        plt.plot(self.v_epoch, self.v_dloss, label='Dloss')
        plt.plot(self.v_epoch, self.v_gloss, label='Gloss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.legend()
        plt.show()

    def plotGraphLoss(self):
        plt.plot(self.t_epoch, self.t_dloss, label='Dloss Traning')
        plt.plot(self.t_epoch, self.t_gloss, label='Gloss Traning')
        plt.plot(self.v_epoch, self.v_dloss, label='Dloss validation')
        plt.plot(self.v_epoch, self.v_gloss, label='Gloss validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Traning and Validation loss')
        plt.legend()
        plt.show()

    def plotGraphAccTraning(self):
        plt.plot(self.t_epoch, self.t_acc, label='accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Traning accuracy')
        plt.show()
    
    def plotGraphAccValidation(self):
        plt.plot(self.v_epoch, self.v_acc, label='accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation accuracy')
        plt.show()

    def plotGraphAcc(self):
        plt.plot(self.t_epoch, self.t_acc, label='Traning accuracy')
        plt.plot(self.v_epoch, self.v_acc, label='Validation accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Traning and Validation accuracy')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
    dcgan = DCGAN()
    #dcgan.train(epochs=setting.num_epoch, dataset='mnist', save_interval=200)
    dcgan.train(epochs=setting.num_epoch, dataset='emnist', save_interval=200)

    dcgan.plotGraphLossTraning()
    dcgan.plotGraphLossValidation()
    dcgan.plotGraphLoss()
    dcgan.plotGraphAccTraning()
    dcgan.plotGraphAccValidation()
    dcgan.plotGraphAcc()
