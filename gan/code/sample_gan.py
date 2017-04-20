import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import tensorflow as tf


sess = tf.InteractiveSession()
mu = 0.8
sigma = 0.1
num_samples = 100000
num_bins = 100


class GenerativeNetwork:
    dim_z = 1
    dim_g = 1

    def __init__(self):
        rand_uni = tf.random_uniform_initializer(-1e1, 1e1)

        self.z_input = tf.placeholder(tf.float32, shape=[None, self.dim_z], name="z-input")
        self.w0 = tf.Variable(rand_uni([self.dim_z, self.dim_g]))
        self.b0 = tf.Variable(rand_uni([self.dim_g]))

        self.g = tf.nn.sigmoid(tf.matmul(self.z_input, self.w0) + self.b0)

    def generate(self, z_i):
        g_i = sess.run([self.g], feed_dict = {self.z_input: z_i})
        return g_i[0]


class Discriminator:
    dim_x = 1
    dim_d = 1
    num_hidden_neurons = 10
    learning_rate = 1e-1

    def __init__(self):
        self.x_input = tf.placeholder(tf.float32, shape=[None, self.dim_x], name="x-input")
        self.d_target = tf.placeholder(tf.float32, shape=[None, self.dim_d], name="d-target")

        rand_uni = tf.random_uniform_initializer(-1e-2, 1e-2)

        self.w0 = tf.Variable(rand_uni([self.dim_x, self.num_hidden_neurons]))
        self.b0 = tf.Variable(rand_uni([self.num_hidden_neurons]))
        self.w1 = tf.Variable(rand_uni([self.num_hidden_neurons, self.dim_d]))
        self.b1 = tf.Variable(rand_uni([self.dim_d]))

        self.d = self.getNetwork(self.x_input)

        self.loss = tf.losses.mean_squared_error(self.d, self.d_target)
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def discriminate(self, x_i):
        d_i = sess.run([self.d], feed_dict={self.x_input: x_i})
        return d_i[0]

    def train(self, x_i, d_i):
        error, _ = sess.run([self.loss, self.opt], feed_dict={self.x_input: x_i, self.d_target:d_i})
        return error

    def getNetwork(self, input):
        temp = tf.nn.tanh(tf.matmul(input, self.w0) + self.b0)
        return tf.nn.sigmoid(tf.matmul(temp, self.w1) + self.b1)


def draw(x, z, g, D):
    # draw histogram
    bins = np.linspace(0, 1, num_bins)
    px, edge_x = np.histogram(x, bins=bins, density=True)
    pz, edge_z = np.histogram(z, bins=bins, density=True)
    pg, edge_g = np.histogram(g, bins=bins, density=True)
    v = np.linspace(0, 1, len(px))

    v_i = np.reshape(v, (len(v), D.dim_x))
    db = D.discriminate(v_i)
    db = np.reshape(db, len(v))

    plt.plot(v, px, 'b--', linewidth=1)
    plt.plot(v, pz, 'r--', linewidth=1)
    plt.plot(v, pg, 'y--', linewidth=1)
    plt.plot(v, db, 'k--', linewidth=1)

    plt.title('1D GAN Test')
    plt.xlabel('Data Values')
    plt.ylabel('Probability density')
    plt.grid()
    #plt.show()
    #plt.close()


def main():
    x = np.random.normal(mu, sigma, num_samples)
    z = np.random.uniform(0, 1, num_samples)
    g = np.ndarray(num_samples)

    # define networks
    G = GenerativeNetwork()
    D = Discriminator()

    # generate data
    tf.global_variables_initializer().run()

    # below code makes slow processing
    # for i in range(0, num_samples, 1):
    #     z_i = np.reshape(z[i], (1, G.dim_z))
    #     g[i] = G.generate(z_i)

    # fore better speed
    x_i = np.reshape(x, (num_samples, D.dim_x))
    z_i = np.reshape(z, (num_samples, G.dim_z))
    g_i = G.generate(z_i)
    g = np.reshape(g_i, num_samples)

    d_x_i = np.ndarray(shape=(num_samples, D.dim_x))
    d_x_i.fill(1.0)

    d_g_i = np.ndarray(shape=(num_samples, D.dim_x))
    d_g_i.fill(0.0)

    # pre-training
    for iter in range(0, 200, 1):
        D.train(x_i, d_x_i)
        D.train(g_i, d_g_i)
        if iter % 100 == 0:
            print(D.train(x_i, d_x_i))
            print(D.train(g_i, d_g_i))

    # draw(x, z, g, D)

    # GAN algorithm
    # Training Generator
    D_from_g = D.getNetwork(G.g)

    loss_g = tf.reduce_mean(-tf.log(D_from_g))
    opt_g = tf.train.GradientDescentOptimizer(1e-3).minimize(loss_g)

    # Training Discriminator
    # Note D.d is d from x_i (discriminated real input, should be 1)
    loss_d = tf.reduce_mean(-tf.log(D.d) - tf.log(1.0 - D_from_g))
    opt_d = tf.train.GradientDescentOptimizer(1e-3).minimize(loss_d)

    # Train both
    frame_num = 0
    for tr in range(0, 10000, 1):
        # train Discriminator from real / generated samples
        D.train(x_i, d_x_i)
        D.train(g_i, d_g_i)

        # GAN update
        sess.run([loss_g, opt_g], feed_dict={G.z_input: z_i})
        sess.run([loss_d, opt_d], feed_dict={D.x_input: x_i, G.z_input: z_i})

        if tr % 1000 == 0:
            error_g, _ = sess.run([loss_g, opt_g], feed_dict={G.z_input: z_i})
            error_d, _ = sess.run([loss_d, opt_d], feed_dict={D.x_input: x_i, G.z_input: z_i})
            print(error_g, error_d)

            # generate g from z_again to respond the training of Generator
            g_i = G.generate(z_i)
            g = np.reshape(g_i, (num_samples))

            draw(x, z, g, D)
            print("Frame num ", frame_num)

            filename = "./capture" + str(frame_num).zfill(5) + ".png"
            plt.savefig(filename)
            frame_num += 1
            plt.close()

if __name__ == '__main__':
    main()
