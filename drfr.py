from __future__ import division

import tensorflow as tf
from util.net_builder import *
from tensorflow.python.ops import data_flow_ops
from util.file_reader import *
from util.progress import *
from datetime import datetime


class DualReferenceFR(object):
    def __init__(self):

        # data directories
        self.data_dir = '/home/bingzhang/Documents/Dataset/CACD/CACD2000'
        self.data_info = '/home/bingzhang/Documents/Dataset/CACD/celenew.mat'
        # image size
        self.image_height = 250
        self.image_width = 250
        self.image_channel = 3
        # net parameters
        self.step = 0
        self.learning_rate = 0.01
        self.batch_size = 21  # must be a multiple of 3
        self.feature_dim = 1024
        self.embedding_size = 128
        self.max_epoch = 10000
        self.delta = 0.2
        self.nof_sampled_id = 40
        self.nof_images_per_id = 10
        self.sampled_examples = self.nof_images_per_id * self.nof_sampled_id
        # placeholder
        self.path_placeholder = tf.placeholder(tf.string, [None, 1], name='paths')
        self.label_placeholder = tf.placeholder(tf.int64, [None, 1], name='labels')
        self.batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        self.affinity_watch = tf.placeholder(tf.float32, [None, self.sampled_examples, self.sampled_examples, 1])
        self.affinity_watch_binarized = tf.placeholder(tf.float32,
                                                       [None, self.sampled_examples, self.sampled_examples, 1])
        # ops
        self.input_queue = data_flow_ops.FIFOQueue(capacity=1000000, dtypes=[tf.string, tf.int64], shapes=[(1,), (1,)])
        self.enqueue_op = self.input_queue.enqueue_many([self.path_placeholder, self.label_placeholder])

        nof_process_threads = 4
        images_and_labels = []
        for _ in range(nof_process_threads):
            file_paths, labels = self.input_queue.dequeue()
            images = []
            for file_path in tf.unstack(file_paths):
                file_content = tf.read_file(file_path)
                image = tf.image.decode_jpeg(file_content)
                image.set_shape((self.image_height, self.image_width, self.image_channel))
                images.append(tf.image.per_image_standardization(image))
            images_and_labels.append([images, labels])
        self.image_batch, self.label_batch = tf.train.batch_join(images_and_labels,
                                                                 batch_size=self.batch_size_placeholder,
                                                                 enqueue_many=True,
                                                                 capacity=nof_process_threads * self.batch_size,
                                                                 shapes=[
                                                                     (self.image_height, self.image_width,
                                                                      self.image_channel), ()],
                                                                 allow_smaller_final_batch=True)

        self.embedding = self.net_forward(self.image_batch)
        self.id_embedding = self.get_id_embeddings(self.embedding)
        self.id_loss = self.get_triplet_loss(self.id_embedding)
        self.id_opt = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9).minimize(self.id_loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=self.coord, sess=self.sess)
        self.summary_op = self.build_summary()

    def net_forward(self, image_batch):
        net, _ = inference(image_batch, keep_probability=1.0, bottleneck_layer_size=128, phase_train=True,
                           weight_decay=0.0, reuse=None)
        feature = slim.fully_connected(net, self.feature_dim, activation_fn=None,
                                       weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                       weights_regularizer=slim.l2_regularizer(0.0))
        return feature

    def get_id_embeddings(self, feature):
        with tf.variable_scope('id_embedding'):
            id_embeddings = slim.fully_connected(feature, self.embedding_size, activation_fn=None,
                                                 weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                 weights_regularizer=slim.l2_regularizer(0.01), scope='id_embedding')
            weights = slim.get_model_variables('id_embedding')[0]
            bias = slim.get_model_variables('id_embedding')[1]
            variable_summaries(weights, 'weight')
            variable_summaries(bias, 'bias')
            id_embeddings = tf.nn.l2_normalize(id_embeddings, dim=1, epsilon=1e-12, name='id_embeddings')
        return id_embeddings

    def get_triplet_loss(self, embeddings):
        anchor = embeddings[0:self.batch_size:3][:]
        positive = embeddings[1:self.batch_size:3][:]
        negative = embeddings[2:self.batch_size:3][:]

        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), self.delta)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
        return loss

    def build_summary(self):
        with tf.name_scope('Affinity'):
            tf.summary.image('original', self.affinity_watch)
            tf.summary.image('binarized', self.affinity_watch_binarized)
        with tf.name_scope('Loss'):
            tf.summary.scalar('id_loss', self.id_loss)
        return tf.summary.merge_all()

    # def test_module(self):
    #     CACD = FileReader(self.data_dir,self.data_info,reproducible=True,contain_val=False)
    #     path_array, label_array = CACD.select_identity_path(self.nof_sampled_id, self.nof_images_per_id)
    #     path_array = np.reshape(path_array,(-1,1))
    #
    #     label_array = np.reshape(np.arange(self.nof_sampled_id*self.nof_images_per_id),(-1,1))
    #     # label_array = np.reshape(label_array,(-1,1))
    #     for step in range(4):
    #         print('before %d enqueue, size is %d' % (step, self.sess.run(self.input_queue.size())))
    #         self.sess.run(self.enqueue_op,
    #                       feed_dict={self.path_place_holder: path_array, self.label_place_holder: label_array})
    #         sum,label= self.sess.run([self.summary_op,self.label_batch])
    #         # sio.savemat('test.mat',{'im':image})
    #         # sum = self.sess.run(self.summary_op)
    #         print(label)
    #         sum,label= self.sess.run([self.summary_op,self.label_batch])
    #         print(label)
    #         print('after dequeue, size is %d' %self.sess.run(self.input_queue.size()))
    #         self.summary_writer.add_summary(sum, step)


    def train(self):
        CACD = FileReader(self.data_dir, self.data_info, reproducible=True, contain_val=False)
        summaryWriter = tf.summary.FileWriter(os.path.join('./log/', datetime.now().isoformat()), self.sess.graph)

        for triplet_selection in range(self.max_epoch):
            # select some examples to forward propagation
            path_array, label_array = CACD.select_identity_path(self.nof_sampled_id, self.nof_images_per_id)
            nof_examples = self.nof_sampled_id * self.nof_images_per_id
            path_array = np.reshape(path_array, (-1, 1))
            label_array = np.reshape(np.arange(nof_examples), (-1, 1))
            embeddings_array = np.zeros(shape=(nof_examples, self.embedding_size))

            # FIFO enqueue
            self.sess.run(self.enqueue_op,
                          feed_dict={self.path_placeholder: path_array, self.label_placeholder: label_array})

            # forward propagation to get current embeddings
            print('Forward propagation to get current embeddings\n')
            nof_batches = int(np.ceil(nof_examples / self.batch_size))
            for i in range(nof_batches):
                batch_size = min(nof_examples - i * self.batch_size, self.batch_size)
                emb, label = self.sess.run([self.id_embedding, self.label_batch],
                                           feed_dict={self.batch_size_placeholder: batch_size})
                embeddings_array[label, :] = emb

            # compute affinity matrix
            aff = []
            for index in range(nof_examples):
                aff.append(np.sum(np.square(embeddings_array[index][:] - embeddings_array), 1))
            aff_binarized = binarize_affinity(aff, self.nof_images_per_id)

            # select triplets to train
            triplet = select_triplets(embeddings_array, self.nof_sampled_id, self.nof_images_per_id, self.delta)
            triplet_path_array = path_array[triplet][:]
            triplet_label_array = label_array[triplet][:]
            triplet_path_array = np.reshape(triplet_path_array, (-1, 1))
            triplet_label_array = np.reshape(triplet_label_array, (-1, 1))
            nof_triplets = len(triplet_path_array)
            print('%d triplets selected' % int(nof_triplets / 3))

            # FIFO enqueue
            self.sess.run(self.enqueue_op, feed_dict={self.path_placeholder: triplet_path_array,
                                                      self.label_placeholder: triplet_label_array})
            # train on selected triplets
            nof_batches = int(np.ceil(nof_triplets / self.batch_size))
            for i in range(nof_batches):
                batch_size = min(nof_triplets - i * self.batch_size, self.batch_size)
                sum, loss, _ = self.sess.run([self.summary_op,self.id_loss, self.id_opt],
                                             feed_dict={self.batch_size_placeholder: batch_size,
                                                        self.affinity_watch: np.reshape(aff, [1, self.sampled_examples,
                                                                                              self.sampled_examples,
                                                                                              1]),
                                                        self.affinity_watch_binarized: np.reshape(aff_binarized, [1,
                                                                                                                  self.sampled_examples,
                                                                                                                  self.sampled_examples,
                                                                                                                  1])})
                progress(i + 1, nof_batches, str(triplet_selection) + 'th epoch',
                         'batches loss:' + str(loss))  # a command progress bar to watch training progress
                self.step += 1
                summaryWriter.add_summary(sum, self.step)


def binarize_affinity(aff, k):
    temp = np.argsort(aff)
    ranks = np.arange(len(aff))[np.argsort(temp)]
    ranks[np.where(ranks > k)] = 255
    return ranks


def select_triplets(embeddings, nof_attr, nof_images_per_attr, delta):
    aff = []
    triplet = []
    for anchor_id in range(nof_attr * nof_images_per_attr):
        dist = np.sum(np.square(embeddings - embeddings[anchor_id]), 1)
        aff.append(dist)
        for pos_id in range(anchor_id + 1, (anchor_id // nof_images_per_attr + 1) * nof_images_per_attr):
            neg_dist = np.copy(dist)
            neg_dist[anchor_id:(anchor_id // nof_images_per_attr + 1) * nof_images_per_attr] = np.NAN
            neg_ids = np.where(neg_dist - dist[pos_id] < delta)[0]
            nof_neg_ids = len(neg_ids)
            if nof_neg_ids > 10:
                rand_id = np.random.randint(nof_neg_ids)
                neg_id = neg_ids[rand_id]
                triplet.append([anchor_id, pos_id, neg_id])
    np.random.shuffle(triplet)
    return triplet


if __name__ == '__main__':
    instance = DualReferenceFR()
    instance.train()
