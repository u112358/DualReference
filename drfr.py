import tensorflow as tf
from util.net_builder import *
from tensorflow.python.ops import data_flow_ops
from util.file_reader import *


class DualReferenceFR(object):
    def __init__(self):

        self.data_dir ='D:\Dataset\CACD2000'
        self.data_info = 'D:\Dataset\celenew.mat'

        self.image_height = 250
        self.image_width = 250
        self.image_channel = 3

        self.learning_rate = 0.01
        self.batch_size = 100
        self.feature_dim = 1024
        self.embedding_size = 128
        self.max_epoch = 20
        self.delta = 0.2
        self.nof_sampled_id = 20
        self.nof_images_per_id = 10

        self.path_place_holder = tf.placeholder(tf.string, [None, 1], name='path_place_holder')
        self.label_place_holder = tf.placeholder(tf.int64, [None, 1], name='label_place_holder')

        self.input_queue = data_flow_ops.FIFOQueue(capacity=1000000, dtypes=[tf.string, tf.int64], shapes=[(1,), (1,)])
        self.enqueue_op = self.input_queue.enqueue_many([self.path_place_holder, self.label_place_holder])

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
        self.image_batch, self.label_batch = tf.train.batch_join(images_and_labels, batch_size=self.batch_size, enqueue_many=True,
                                                       capacity=nof_process_threads*self.batch_size, shapes=[
                (self.image_height, self.image_width, self.image_channel), ()], allow_smaller_final_batch=True)
        tf.summary.image('image_batch', self.image_batch,100)

        self.embedding = self.net_forward(self.image_batch)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=self.coord, sess=self.sess)
        self.summary_writer = tf.summary.FileWriter('./log/train', self.sess.graph)
        self.summary_op = tf.summary.merge_all()

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

    def test_module(self):
        CACD = FileReader(self.data_dir,self.data_info,reproducible=True,contain_val=False)
        path_array, label_array = CACD.select_identity_path(self.nof_sampled_id, self.nof_images_per_id)
        path_array = np.reshape(path_array,(-1,1))

        label_array = np.reshape(np.arange(self.nof_sampled_id*self.nof_images_per_id),(-1,1))
        # label_array = np.reshape(label_array,(-1,1))
        for step in range(4):
            print('before %d enqueue, size is %d' % (step, self.sess.run(self.input_queue.size())))
            self.sess.run(self.enqueue_op,
                          feed_dict={self.path_place_holder: path_array, self.label_place_holder: label_array})
            sum,label= self.sess.run([self.summary_op,self.label_batch])
            # sio.savemat('test.mat',{'im':image})
            # sum = self.sess.run(self.summary_op)
            print(label)
            sum,label= self.sess.run([self.summary_op,self.label_batch])
            print(label)
            print('after dequeue, size is %d' %self.sess.run(self.input_queue.size()))
            self.summary_writer.add_summary(sum, step)

    def train(self):
        CACD = FileReader(self.data_dir,self.data_info,reproducible=True,contain_val=False)
        path_array, label_array = CACD.select_identity_path(self.nof_sampled_id, self.nof_images_per_id)
        path_array = np.reshape(path_array, (-1, 1))
        label_array = np.reshape(label_array, (-1, 1))

if __name__ == '__main__':
    instance = DualReferenceFR()
    instance.test_module()
