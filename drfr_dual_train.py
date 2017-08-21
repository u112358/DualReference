from __future__ import division

from util.inception_resnet_v1 import *
from tensorflow.python.ops import data_flow_ops
from util.file_reader import *
from util.progress import *
from datetime import datetime


class DualReferenceFR(object):
    def __init__(self):
        # data directories
        # self.data_dir = '/home/bingzhang/Documents/Dataset/CACD/CACD2000'
        # self.data_info = '/home/bingzhang/Documents/Dataset/CACD/celenew.mat'
        # self.val_dir = '/home/bingzhang/Documents/Dataset/ZID/LFW/lfw'

        self.data_dir = '/scratch/BingZhang/dataset/CACD2000_Cropped'
        self.data_info = '/scratch/BingZhang/dataset/CACD2000/celenew2.mat'
        self.val_dir = '/scratch/BingZhang/lfw/'
        self.val_list = './data/val_list.txt'
        # model directory
        self.log_dir = '/scratch/BingZhang/logs_all_in_one/drfr_dual_train'
        self.model_dir = '/scratch/BingZhang/models_all_in_one/DRFRQDualTrain0.006lrbatch30-Model'
        # image size
        self.image_height = 250
        self.image_width = 250
        self.image_channel = 3
        # net parameters
        self.step = 0
        self.learning_rate = 0.006
        self.batch_size = 30  # must be a multiple of 3
        self.feature_dim = 1024
        self.embedding_size = 128
        self.max_epoch = 10000
        self.delta = 0.2
        self.nof_sampled_id = 45
        self.nof_images_per_id = 20
        self.id_sampled_examples = self.nof_images_per_id * self.nof_sampled_id
        self.nof_sampled_age = 20
        self.nof_images_per_age = 45
        self.age_sampled_examples = self.nof_images_per_age * self.nof_sampled_age
        self.val_size = 144
        # placeholder
        self.path_placeholder = tf.placeholder(tf.string, [None, 3], name='paths')
        self.label_placeholder = tf.placeholder(tf.int64, [None, 3], name='labels')
        self.batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')

        # for id
        self.id_affinity_watch = tf.placeholder(tf.float32,
                                                [None, self.id_sampled_examples, self.id_sampled_examples, 1])
        self.id_affinity_watch_binarized = tf.placeholder(tf.float32,
                                                          [None, self.id_sampled_examples, self.id_sampled_examples, 1])
        self.affinity_on_val = tf.placeholder(tf.float32, [None, self.val_size, self.val_size, 1])
        self.nof_id_triplets_placeholder = tf.placeholder(tf.int16, name='nof_id_triplets')

        # for age
        self.age_affinity_watch = tf.placeholder(tf.float32,
                                                 [None, self.age_sampled_examples, self.age_sampled_examples, 1])
        self.age_affinity_watch_binarized = tf.placeholder(tf.float32,
                                                           [None, self.age_sampled_examples, self.age_sampled_examples,
                                                            1])
        self.nof_age_triplets_placeholder = tf.placeholder(tf.int16, name='nof_age_triplets')
        # ops
        self.input_queue = data_flow_ops.FIFOQueue(capacity=1000000, dtypes=[tf.string, tf.int64], shapes=[(3,), (3,)])
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
        self.id_loss = self.get_triplet_loss_v2(self.id_embedding)
        self.age_embedding = self.get_age_embeddings(self.embedding)
        self.age_loss = self.get_triplet_loss_v2(self.age_embedding)
        self.age_summary_op, self.id_summary_op ,self.average_op = self.build_summary()
        with tf.control_dependencies([self.average_op]):
            self.id_opt = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9,
                                                     use_nesterov=True).minimize(self.id_loss)
            self.age_opt = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9,
                                                      use_nesterov=True).minimize(self.age_loss)

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
                                                 weights_regularizer=slim.l2_regularizer(0.0), scope='id_embedding')
            weights = slim.get_model_variables('id_embedding')[0]
            bias = slim.get_model_variables('id_embedding')[1]
            variable_summaries(weights, 'weight')
            variable_summaries(bias, 'bias')
            id_embeddings = tf.nn.l2_normalize(id_embeddings, dim=1, epsilon=1e-12, name='id_embeddings')
        return id_embeddings

    def get_age_embeddings(self, feature):
        with tf.variable_scope('age_embedding'):
            age_embeddings = slim.fully_connected(feature, self.embedding_size, activation_fn=None,
                                                  weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                  weights_regularizer=slim.l2_regularizer(0.0), scope='age_embedding')
            weights = slim.get_model_variables('age_embedding')[0]
            bias = slim.get_model_variables('age_embedding')[1]
            variable_summaries(weights, 'weight')
            variable_summaries(bias, 'bias')
            age_embeddings = tf.nn.l2_normalize(age_embeddings, dim=1, epsilon=1e-12, name='id_embeddings')
        return age_embeddings

    def get_triplet_loss(self, embeddings):
        anchor = embeddings[0:self.batch_size:3][:]
        positive = embeddings[1:self.batch_size:3][:]
        negative = embeddings[2:self.batch_size:3][:]

        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

        with tf.name_scope('distances'):
            tf.summary.histogram('positive', pos_dist)
            tf.summary.histogram('negative', neg_dist)

        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), self.delta)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
        return loss

    def get_triplet_loss_v2(self, embeddings):
        with tf.name_scope('loss_012'):
            anchor = embeddings[0:self.batch_size:3][:]
            positive = embeddings[1:self.batch_size:3][:]
            negative = embeddings[2:self.batch_size:3][:]
            pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
            neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

            with tf.name_scope('distances'):
                tf.summary.histogram('positive_012', pos_dist)
                tf.summary.histogram('negative_012', neg_dist)

            basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), self.delta)
            loss_012 = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

        with tf.name_scope('loss_102'):
            anchor = embeddings[1:self.batch_size:3][:]
            positive = embeddings[0:self.batch_size:3][:]
            negative = embeddings[2:self.batch_size:3][:]
            pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
            neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

            with tf.name_scope('distances'):
                tf.summary.histogram('positive_102', pos_dist)
                tf.summary.histogram('negative_102', neg_dist)

            basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), self.delta)
            loss_102 = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
        return loss_012 + loss_102

    def build_summary(self):
        with tf.name_scope('affinity'):
            tf.summary.image('id_original', self.id_affinity_watch)
            tf.summary.image('id_binarized', self.id_affinity_watch_binarized)
            tf.summary.image('age_original', self.age_affinity_watch)
            tf.summary.image('age_binarized', self.age_affinity_watch_binarized)
            tf.summary.image('val', self.affinity_on_val)
        with tf.name_scope('loss'):
            tf.summary.scalar('id_loss', self.id_loss)
            tf.summary.scalar('age_loss', self.age_loss)

            average = tf.train.ExponentialMovingAverage(0.9)
            average_op = average.apply([self.id_loss, self.age_loss])
            tf.summary.scalar('id_loss_aver', average.average(self.id_loss))
            tf.summary.scalar('age_loss_aver', average.average(self.age_loss))
        with tf.name_scope('embeddings'):
            tf.summary.histogram('id_embeddings', self.id_embedding)
            tf.summary.histogram('age_embeddings', self.age_embedding)
        with tf.name_scope('nof_triplet'):
            tf.summary.scalar('nof_id_triplet', self.nof_id_triplets_placeholder)
            tf.summary.scalar('nof_age_triplet', self.nof_age_triplets_placeholder)

        var = tf.get_collection(tf.GraphKeys.SUMMARIES)
        age_sum = [v for v in var if str(v).__contains__('age')]
        id_sum = [v for v in var if str(v).__contains__('id')]
        return tf.summary.merge(age_sum),tf.summary.merge(id_sum), average_op

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
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        CACD = FileReader(self.data_dir, self.data_info, val_data_dir=self.val_dir, val_list=self.val_list,
                          reproducible=True, contain_val=True)
        summary_writer = tf.summary.FileWriter(
            os.path.join(self.log_dir, datetime.now().isoformat()), sess.graph)
        var = tf.trainable_variables()
        var = [v for v in var if str(v).__contains__('Inception')]
        saver = tf.train.Saver(var)
        saver.restore(sess, '/scratch/BingZhang/facenet4drfr/model/20170512-110547/model-20170512-110547.ckpt-250000')
        # saver.restore(sess, '/home/bingzhang/Workspace/PycharmProjects/20170512-110547/model-20170512-110547.ckpt-250000')
        aff_val = np.zeros((self.val_size, self.val_size))
        for triplet_selection in range(self.max_epoch):

            # ID Step
            # select some examples to forward propagation
            paths, labels = CACD.select_identity_path(self.nof_sampled_id, self.nof_images_per_id)
            nof_examples = self.nof_sampled_id * self.nof_images_per_id
            path_array = np.reshape(paths, (-1, 3))
            label_array = np.reshape(np.arange(nof_examples), (-1, 3))
            embeddings_array = np.zeros(shape=(nof_examples, self.embedding_size))

            # FIFO enqueue
            sess.run(self.enqueue_op,
                     feed_dict={self.path_placeholder: path_array, self.label_placeholder: label_array})

            # forward propagation to get current embeddings
            print('ID STEP\nForward propagation to get current embeddings')
            nof_batches = int(np.ceil(nof_examples / self.batch_size))
            for i in range(nof_batches):
                batch_size = min(nof_examples - i * self.batch_size, self.batch_size)
                emb, label = sess.run([self.id_embedding, self.label_batch],
                                      feed_dict={self.batch_size_placeholder: batch_size})
                embeddings_array[label, :] = emb

            # compute affinity matrix on batch
            aff = []
            for index in range(nof_examples):
                aff.append(np.sum(np.square(embeddings_array[index][:] - embeddings_array), 1))
            aff_binarized = binarize_affinity(aff, self.nof_images_per_id)
            # affinity matrix on

            # select triplets to train
            triplet = select_triplets(embeddings_array, self.nof_sampled_id, self.nof_images_per_id, self.delta)
            triplet_path_array = paths[triplet][:]
            triplet_label_array = labels[triplet][:]
            nof_triplets = len(triplet_path_array)
            print('%d triplets selected' % nof_triplets)

            # FIFO enqueue
            sess.run(self.enqueue_op, feed_dict={self.path_placeholder: triplet_path_array,
                                                 self.label_placeholder: triplet_label_array})
            # train on selected triplets
            nof_batches = int(np.ceil(nof_triplets * 3 / self.batch_size))
            for i in range(nof_batches):
                batch_size = min(nof_triplets * 3 - i * self.batch_size, self.batch_size)
                if self.step % 200 == 0:
                    _sum, loss, _ = sess.run(
                        [self.id_summary_op, self.id_loss, self.id_opt],
                        feed_dict={self.batch_size_placeholder: batch_size,
                                   self.id_affinity_watch: np.reshape(aff, [1, self.id_sampled_examples,
                                                                            self.id_sampled_examples,
                                                                            1]),
                                   self.id_affinity_watch_binarized: np.reshape(aff_binarized, [1,
                                                                                             self.id_sampled_examples,
                                                                                             self.id_sampled_examples,
                                                                                             1]),
                                   self.nof_id_triplets_placeholder: nof_triplets,
                                   self.affinity_on_val: np.reshape(aff_val, [1, self.val_size, self.val_size, 1])})

                    # write in summary
                    summary_writer.add_summary(_sum, self.step)
                else:
                    loss, _ = sess.run([self.id_loss, self.id_opt], feed_dict={self.batch_size_placeholder: batch_size})
                progress(i + 1, nof_batches, str(triplet_selection) + 'th Epoch',
                         'Batches loss:' + str(loss))  # a command progress bar to watch training progress
                self.step += 1
                # save model
                if self.step % 200000 == 0:
                    saver.save(sess, self.model_dir, global_step=self.step)

            for _ in range(4):
                # AGE Step
                # select some examples to forward propagation
                paths, labels = CACD.select_age_path(self.nof_sampled_age, self.nof_images_per_age)
                nof_examples = self.nof_sampled_age * self.nof_images_per_age
                path_array = np.reshape(paths, (-1, 3))
                label_array = np.reshape(np.arange(nof_examples), (-1, 3))
                embeddings_array = np.zeros(shape=(nof_examples, self.embedding_size))

                # FIFO enqueue
                sess.run(self.enqueue_op,
                         feed_dict={self.path_placeholder: path_array, self.label_placeholder: label_array})

                # forward propagation to get current embeddings
                print('AGE STEP\nForward propagation to get current embeddings')
                nof_batches = int(np.ceil(nof_examples / self.batch_size))
                for i in range(nof_batches):
                    batch_size = min(nof_examples - i * self.batch_size, self.batch_size)
                    emb, label = sess.run([self.age_embedding, self.label_batch],
                                          feed_dict={self.batch_size_placeholder: batch_size})
                    embeddings_array[label, :] = emb

                # compute affinity matrix on batch
                aff = []
                for index in range(nof_examples):
                    aff.append(np.sum(np.square(embeddings_array[index][:] - embeddings_array), 1))
                aff_binarized = binarize_affinity(aff, self.nof_images_per_age)
                # affinity matrix on

                # select triplets to train
                triplet = select_triplets(embeddings_array, self.nof_sampled_age, self.nof_images_per_age, self.delta)
                triplet_path_array = paths[triplet][:]
                triplet_label_array = labels[triplet][:]
                nof_triplets = len(triplet_path_array)
                print('%d triplets selected' % nof_triplets)

                # FIFO enqueue
                sess.run(self.enqueue_op, feed_dict={self.path_placeholder: triplet_path_array,
                                                     self.label_placeholder: triplet_label_array})
                # train on selected triplets
                nof_batches = int(np.ceil(nof_triplets * 3 / self.batch_size))
                for i in range(nof_batches):
                    batch_size = min(nof_triplets * 3 - i * self.batch_size, self.batch_size)
                    if self.step % 200 == 0:
                        _sum, loss, _ = sess.run(
                            [self.age_summary_op, self.age_loss, self.age_opt],
                            feed_dict={self.batch_size_placeholder: batch_size,
                                       self.age_affinity_watch: np.reshape(aff, [1, self.age_sampled_examples,
                                                                                self.age_sampled_examples,
                                                                                1]),
                                       self.age_affinity_watch_binarized: np.reshape(aff_binarized, [1,
                                                                                                    self.age_sampled_examples,
                                                                                                    self.age_sampled_examples,
                                                                                                    1]),
                                       self.nof_age_triplets_placeholder: nof_triplets,
                                       self.affinity_on_val: np.reshape(aff_val,
                                                                        [1, self.val_size, self.val_size, 1])})

                        # write in summary
                        summary_writer.add_summary(_sum, self.step)
                    else:
                        loss, _ = sess.run([self.age_loss, self.age_opt],
                                           feed_dict={self.batch_size_placeholder: batch_size})
                    progress(i + 1, nof_batches, str(triplet_selection) + 'th Epoch',
                             'Batches loss:' + str(loss))  # a command progress bar to watch training progress
                    self.step += 1
                    # save model
                    if self.step % 200000 == 0:
                        saver.save(sess, self.model_dir, global_step=self.step)

            # perform a validation on lfw
            val_paths = CACD.get_val(self.val_size)
            val_path_array = np.reshape(val_paths, (-1, 3))
            val_label_array = np.reshape(np.arange(self.val_size), (-1, 3))
            val_embeddings_array = np.zeros(shape=(self.val_size, self.embedding_size))

            # FIFO enqueue
            sess.run(self.enqueue_op,
                     feed_dict={self.path_placeholder: val_path_array,
                                self.label_placeholder: val_label_array})

            # forward propagation to get val embeddings
            print('Forward propagation on validation set')
            nof_batches = int(np.ceil(self.val_size / self.batch_size))
            for i in range(nof_batches):
                batch_size = min(self.val_size - i * self.batch_size, self.batch_size)
                emb, label = sess.run([self.id_embedding, self.label_batch],
                                      feed_dict={self.batch_size_placeholder: batch_size})
                val_embeddings_array[label, :] = emb
            aff_val = []
            for index in range(self.val_size):
                aff_val.append(np.sum(np.square(val_embeddings_array[index][:] - val_embeddings_array), 1))


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
