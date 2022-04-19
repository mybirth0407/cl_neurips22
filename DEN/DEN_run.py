import os
import tensorflow as tf
import numpy as np
import DEN
from tensorflow.examples.tutorials.mnist import input_data
from loader import get_data_loader, transform
import wandb


np.random.seed(1004)
flags = tf.app.flags
flags.DEFINE_integer("max_iter", 4300, "Epoch to train")
flags.DEFINE_float("lr", 0.001, "Learing rate(init) for train")
flags.DEFINE_integer("batch_size", 64, "The size of batch for 1 iteration")
flags.DEFINE_string(
    "checkpoint_dir", "checkpoints", "Directory path to save the checkpoints"
)
flags.DEFINE_list(
    "dims", [784 * 3, 312, 128, 10], "Dimensions about layers including output"
)
flags.DEFINE_integer("n_classes", 10, "The number of classes at each task")
flags.DEFINE_float("l1_lambda", 0.00001, "Sparsity for L1")
flags.DEFINE_float("l2_lambda", 0.0001, "L2 lambda")
flags.DEFINE_float("gl_lambda", 0.001, "Group Lasso lambda")
flags.DEFINE_float("regular_lambda", 0.5, "regularization lambda")
flags.DEFINE_integer(
    "ex_k", 10, "The number of units increased in the expansion processing"
)
flags.DEFINE_float("loss_thr", 0.01, "Threshold of dynamic expansion")
flags.DEFINE_float("spl_thr", 0.05, "Threshold of split and duplication")

flags.DEFINE_string("task_1", "BG", "fg or bg")
flags.DEFINE_string("task_2", "BG", "fg or bg")
flags.DEFINE_string("test_db", "BG", "fg or bg")
flags.DEFINE_float("biased_r1", 0.0, "biased ratio")

FLAGS = flags.FLAGS

# mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
# trainX = mnist.train.images
# valX = mnist.validation.images
# testX = mnist.test.images

# task_permutation = []
# for task in range(10):
#     task_permutation.append(np.random.permutation(784))

# print(task_permutation)

# trainXs, valXs, testXs = [], [], []
# for task in range(10):
#     trainXs.append(trainX[:, task_permutation[task]])
#     valXs.append(valX[:, task_permutation[task]])
#     testXs.append(testX[:, task_permutation[task]])
indices = np.random.permutation(np.arange(60000))
num_step = 10

dic_tr_indices = {}
dic_val_indices = {}
part_tr = int(50000 / num_step)
part_val = int(10000 / num_step)

for i in range(1, 11):
    dic_tr_indices[i] = indices[
        (i - 1) * part_tr
        + (i - 1) * part_val : i * part_tr
        + (i - 1) * part_val
    ]
    dic_val_indices[i] = indices[
        i * part_tr + (i - 1) * part_val : i * part_tr + i * part_val
    ]

dic_tr_dl = {}
dic_val_dl = {}
colormap_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
trainXs, trainYs = [], []
valXs, valYs = [], []

for i in range(1, 11):
    trainX, trainY = [], []
    valX, valY = [], []
    # different bias on each task
    tmp = colormap_idxs[0]
    colormap_idxs = colormap_idxs[1:]
    colormap_idxs.append(tmp)

    dic_tr_dl[i] = get_data_loader(
        batch_size=FLAGS.batch_size,
        mode=FLAGS.task_1,
        train=True,
        transform=transform,
        data_label_correlation=FLAGS.biased_r1,
        data_indices=dic_tr_indices[i],
        colormap_idxs=colormap_idxs,
    )

    for images, labels in dic_tr_dl[i]:
        images = images.numpy()
        labels = np.eye(10)[labels.numpy()]
        trainX.extend(images)
        trainY.extend(labels)
    trainX = np.asarray(trainX)
    trainY = np.asarray(trainY)
    trainXs.append(trainX)
    trainYs.append(trainY)

    dic_val_dl[i] = get_data_loader(
        batch_size=FLAGS.batch_size,
        mode=FLAGS.task_1,
        train=True,
        transform=transform,
        data_label_correlation=FLAGS.biased_r1,
        data_indices=dic_val_indices[i],
        colormap_idxs=colormap_idxs,
    )

    for images, labels in dic_val_dl[i]:
        images = images.numpy()
        labels = np.eye(10)[labels.numpy()]
        valX.extend(images)
        valY.extend(labels)
    valX = np.asarray(valX)
    valY = np.asarray(valY)
    valXs.append(valX)
    valYs.append(valY)

# trainXs, trainYs = np.asarray(trainXs), np.asarray(trainYs)
# valXs, valYs = np.asarray(valXs), np.asarray(valYs)

unbiased_te_dl = get_data_loader(
    batch_size=FLAGS.batch_size,
    mode=FLAGS.test_db,
    train=False,
    transform=transform,
    data_label_correlation=0.1,
)

testX, testY = [], []
for images, labels in unbiased_te_dl:
    images = images.numpy()
    labels = np.eye(10)[labels.numpy()]
    testX.extend(images)
    testY.extend(labels)
testX = np.asarray(testX)
testY = np.asarray(testY)

model = DEN.DEN(FLAGS)

params = dict()
avg_perf = []

writer = wandb.init(name="den/biased_mnist", project="CL_forget")

for t in range(FLAGS.n_classes):
    logs = dict()
    data = (
        trainXs[t],
        trainYs[t],
        valXs[t],
        valYs[t],
        testX,
        testY,
    )

    model.sess = tf.Session()
    print("\n\n\tTASK %d TRAINING\n" % (t+1))

    model.T = model.T+1
    model.task_indices.append(t+1)
    model.load_params(params, time=1)
    perf, sparsity, expansion = model.add_task(t+1, data)

    params = model.get_params()
    model.destroy_graph()
    model.sess.close()

    model.sess = tf.Session()
    print("\n OVERALL EVALUATION")
    model.load_params(params)
    # temp_perfs = []
    # for j in range(t+1):
    #     print(j, t)
    #     temp_perf = model.predict_perform(j+1, testX, testY)
    #     temp_perfs.append(temp_perf)
    accuracy_unbiased = model.predict_perform(t+1, testX, testY)

    # unbiased 에 대한 acc
    # avg_perf.append(sum(temp_perfs) / float(t+1))
    # print("   [*] avg_perf: %.4f" % avg_perf[t])
    # logs["test/acc"] = avg_perf[t]
    logs["test/acc"] = accuracy_unbiased
    model.destroy_graph()
    model.sess.close()

    if t+1 > 1:
        model.sess = tf.Session()
        model.load_params(params)
        accuracy_prev_task = model.predict_perform(t+1, valXs[t-1], valYs[t-1])
        print(f"accuracy of prev task {accuracy_prev_task}")
        logs["valid/task_pre_acc_after"] = accuracy_prev_task
        model.destroy_graph()
        model.sess.close()

    model.sess = tf.Session()
    model.load_params(params)
    accuracy_current_task = model.predict_perform(t+1, valXs[t], valYs[t])
    print(f"accuracy of current task {accuracy_current_task}")
    logs["valid/task_acc"] = accuracy_current_task
    model.destroy_graph()
    model.sess.close()

    writer.log(logs)
