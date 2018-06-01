import numpy as np
import tensorflow as tf


def calculate_distances(y, Y):
    """
    Calculate the euclidean distance between the points y and the cluster positions Y

    :param y: the embeddings. shape: (batch_size, emb_size)
    :param Y: a vector of cluster positions. shape: (n_clusters, emb_size)
    :return: distance. shape: (batch_size, n_clusters)
    """
    diff = tf.add(tf.expand_dims(y, 1), -Y)  # shape (batch_size, n_cluster, emb_size)
    return tf.sqrt(tf.reduce_sum(tf.square(diff), axis=2))


def assign_clusters(distances):
    """

    :param distances: between clusters. shape: (batch_size, n_clusters)
    :return: index of the closest cluster for each data point. shape: (batch_size, emb_size)
    """
    return tf.argmin(distances, axis=1)


def calculate_new_cluster_positions(y, clusters, n_clusters):
    """

    :param y:
    :param clusters: a vector of integers with the cluster to which the element of y is assigned. shape: (batch_size, )
    :return: Y, cluster position vectors. shape: (n_clusters, emb_size)
    """
    Y = np.zeros((n_clusters, y.shape[-1]))
    for c in range(n_clusters):
        idx = tf.equal(clusters, c)
        Y[c] = tf.reduce_mean(tf.boolean_mask(y, idx), axis=0)
    return tf.constant(Y)


def train_clustering(y, n_steps, n_clusters):
    Y = np.random.random((n_clusters, y.shape[-1]))
    clusters = np.random.randint(n_clusters, size=y.shape[0])
    n = 0
    while n < n_steps:
        distances = calculate_distances(y, Y)
        clusters = assign_clusters(distances)
        Y = calculate_new_cluster_positions(y, clusters, n_clusters)
        # accuracy = tf.equal(labels, clusters)
        # print(accuracy)
        n += 1
    return Y, clusters


if __name__ == '__main__':
    bs, cs, es, N = 100, 10, 3, 200
    cc = tf.constant(np.arange(100) // 10, dtype=tf.float64)
    a = np.random.randn(bs, es)
    y = tf.constant(a) + tf.expand_dims(cc, 1)

    Y_new, clusters = train_clustering(y, N, cs)
    # accuracy = tf.equal(clusters, labels)
    with tf.Session() as sess:
        test = sess.run([y, Y_new])
        print(test)
