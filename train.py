import ast
import pathlib
import random
import matplotlib.pyplot as plt
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from keras.layers import (Input, Reshape, Dense, Lambda, Concatenate, TimeDistributed, Conv2D,
                          LayerNormalization, Add, Softmax, LSTM, Bidirectional)

seed = 12
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
pd.set_option('future.no_silent_downcasting', True)

BATCH_SIZE = 1
EPOCHS = 500
steps_per_epoch = 1000
TRAIN_SEQ_SIZE = 2000
LABEL_SMOOTHING = 10
EMBED_SIZE = 64
HIDDEN_SIZE = 128
KERNEL_SIZE = 64
eps = 1e-8
data_dir = pathlib.Path('./MABe-mouse-behavior-detection')

# csv
train_metadata = pd.read_csv(data_dir / 'train.csv')
# get all the individual behaviors, independent of actor and target
behaviors = (train_metadata['behaviors_labeled']
             .apply(lambda x: ast.literal_eval(x) if x is not np.nan else x)
             .explode('behaviors_labeled'))
behaviors = behaviors.dropna().str.split(',').str[2].unique()
# Get which videos contain the labeled behavior
labeled_videos = []
train_metadata['x_cm_scale'] = train_metadata['video_width_pix']/train_metadata['pix_per_cm_approx']
train_metadata['x_cm_scale'] = train_metadata['x_cm_scale']/(train_metadata['x_cm_scale'].max())
train_metadata['y_cm_scale'] = train_metadata['video_height_pix']/train_metadata['pix_per_cm_approx']
train_metadata['y_cm_scale'] = train_metadata['y_cm_scale']/(train_metadata['y_cm_scale'].max())
for _, row in train_metadata.loc[train_metadata['behaviors_labeled'].notna()].iterrows():
    # This video sucks
    if row['lab_id'] == "AdaptableSnail" and row['video_id'] == 1212811043:
        continue
    labeled_videos.append({
        'lab': row['lab_id'],
        'video': row['video_id'],
        'seconds_per_frame': 1 / row['frames_per_second'],
        'video_width_pix': row['video_width_pix'],
        'video_height_pix': row['video_height_pix'],
        'x_cm_scale': row['x_cm_scale'],
        'y_cm_scale': row['y_cm_scale'],
    })
# Find and get rid of behaviors labeled to be in videos but not actually present
# and get rid of videos whose files are not found
remove_videos = []
for i, video_dict in enumerate(labeled_videos):
    try:
        annot_df = pd.read_parquet(data_dir / f"train_annotation/"
                                   f"{video_dict['lab']}/{video_dict['video']}.parquet")
        if annot_df.empty:
            remove_videos.append(i)
    except FileNotFoundError:
        remove_videos.append(i)

_labeled_videos = []
for i, video_dict in enumerate(labeled_videos):
    if i not in remove_videos:
        _labeled_videos.append(video_dict)
labeled_videos = _labeled_videos
num_videos = len(labeled_videos)

# get rid of the incorrectly formatted behavior names, sort, and add a nothing class
behaviors = list(set(map(lambda x: x.replace("'", ""), behaviors)))
behaviors = sorted(behaviors)
behaviors = ["nothing"] + behaviors
NUM_BEHAVIORS = len(behaviors)
# get all the individual tracked body parts
body_parts = (train_metadata['body_parts_tracked']
              .apply(lambda x: ast.literal_eval(x) if x is not np.nan else x).explode().unique())
body_parts = sorted(body_parts)
BODY_PARTS = len(body_parts)
FEATURES = BODY_PARTS + 1
# map them to integers
body_parts_map = {x: i for i, x in enumerate(body_parts)}
behaviors_map = {x: i for i, x in enumerate(behaviors)}


def get_sequence_from_df(track_df, annot_df, video_width, video_height):
    min_frame, max_frame = track_df['video_frame'].min(), track_df['video_frame'].max()
    labeled_frames = annot_df['start_frame'].tolist()
    if len(labeled_frames) > 0:
        start_frame = random.choice(labeled_frames) - random.randint(0, TRAIN_SEQ_SIZE)
        start_frame = max(min_frame, min(start_frame, max_frame - TRAIN_SEQ_SIZE))
    else:
        start_frame = random.randint(min_frame, max_frame - TRAIN_SEQ_SIZE)
    end_frame = start_frame + TRAIN_SEQ_SIZE
    track_frames_filter = (track_df['video_frame'] >= start_frame) & (track_df['video_frame'] < end_frame)
    track_seq_df = track_df[track_frames_filter]
    pd.options.mode.chained_assignment = None
    track_seq_df['bodypart'] = track_seq_df['bodypart'].replace(body_parts_map)
    track_seq_df['x'] = track_seq_df['x'] / video_width
    track_seq_df['y'] = track_seq_df['y'] / video_height
    track_seq_df['video_frame'] = track_seq_df['video_frame'] - start_frame
    # Mice are not always incremented by 1 in order or 0 indexed in a given segment
    mice_ids = track_df['mouse_id'].unique()
    zero_ind_mice_ids = {x: i for i, x in enumerate(mice_ids)}
    track_seq_df['mouse_id'] = track_seq_df['mouse_id'].replace(zero_ind_mice_ids)
    num_mice = len(mice_ids)
    # Literally no mice data
    if num_mice == 0:
        return None
    pd.options.mode.chained_assignment = 'warn'
    track_array = np.zeros((num_mice, TRAIN_SEQ_SIZE, FEATURES, 3))
    for row in track_seq_df.itertuples():
        track_array[row.mouse_id, row.video_frame, row.bodypart] = [row.x, row.y, 1.0]
    coords = track_array[:, :, :-1, :2]
    mask = track_array[:, :, :-1, 2] == 1.0
    sum_coords = np.sum(coords * mask[..., None], axis=2)
    count_coords = np.sum(mask, axis=2, keepdims=True)
    count_coords[count_coords == 0] = 1
    mean_coords = sum_coords / count_coords
    track_array[:, :, :-1, :2] -= mean_coords[:, :, None, :] * mask[..., None]
    track_array[:, :, -1, :2] = mean_coords
    track_array[:, :, -1, 2] = 1.0

    # Now that we have the positions ready, we can get the annotations
    annot_frames_filter = ~((annot_df['stop_frame'] <= start_frame) | (annot_df['start_frame'] >= end_frame))
    # print([annot_df['start_frame'] < end_frame, annot_df['stop_frame'] >= start_frame])
    annot_seq_df = annot_df[annot_frames_filter]
    pd.options.mode.chained_assignment = None
    # Formatting
    annot_seq_df['action'] = annot_seq_df['action'].replace(behaviors_map)
    annot_seq_df['start_frame'] = (annot_seq_df['start_frame'] - start_frame)
    annot_seq_df['stop_frame'] = (annot_seq_df['stop_frame'] - start_frame)
    annot_seq_df['agent_id'] = annot_seq_df['agent_id'].replace(zero_ind_mice_ids)
    annot_seq_df['target_id'] = annot_seq_df['target_id'].replace(zero_ind_mice_ids)
    pd.options.mode.chained_assignment = 'warn'
    annotations = np.zeros((num_mice, num_mice, TRAIN_SEQ_SIZE, NUM_BEHAVIORS))
    annotations[:, :, :, 0] = 1
    annotations_mask = np.ones((num_mice, num_mice, TRAIN_SEQ_SIZE))
    for agent in range(num_mice):
        for target in range(num_mice):
            relation_filter = (annot_seq_df['agent_id'] == agent) & (annot_seq_df['target_id'] == target)
            relation_seq_df = annot_seq_df[relation_filter]
            if relation_seq_df.empty:
                if random.random() > 0.05:
                    annotations_mask[agent, target, :] = 0

    for row in annot_seq_df.itertuples():
        a = row.agent_id
        t = row.target_id
        s = max(row.start_frame, 0)
        e = min(row.stop_frame, TRAIN_SEQ_SIZE - 2) + 1
        cls = row.action
        annotations[a, t, s:e, :] = 0
        annotations[a, t, s:e, cls] = 1

    annotations = annotations.reshape((num_mice * num_mice, TRAIN_SEQ_SIZE, NUM_BEHAVIORS))
    annotations_mask = annotations_mask.reshape((num_mice * num_mice, TRAIN_SEQ_SIZE))
    if np.sum(annotations_mask) < 1.0:
        return None
    return track_array, annotations, annotations_mask


def ds_generator():
    def internal(video):
        track_df = pd.read_parquet(f"./MABe-mouse-behavior-detection/train_tracking/"
                                   f"{video['lab']}/{video['video']}.parquet")
        track_df = track_df.fillna(0)
        annot_df = pd.read_parquet(f"./MABe-mouse-behavior-detection/train_annotation/"
                                   f"{video['lab']}/{video['video']}.parquet")
        seq_out = None
        while seq_out is None:
            seq_out = get_sequence_from_df(track_df, annot_df, video['video_width_pix'],
                                           video['video_height_pix'])
        return seq_out, video['x_cm_scale'], video['y_cm_scale'], video['seconds_per_frame']

    random.shuffle(labeled_videos)
    for video in labeled_videos:
        yield internal(video)


tf_ds = tf.data.Dataset.from_generator(ds_generator, output_signature=(
    (tf.TensorSpec((None, TRAIN_SEQ_SIZE, FEATURES, 3)),  # seq
     tf.TensorSpec((None, TRAIN_SEQ_SIZE, NUM_BEHAVIORS)),  # annotations
     tf.TensorSpec((None, TRAIN_SEQ_SIZE))),  # mask
    tf.TensorSpec(()),  # x_cm
    tf.TensorSpec(()),  # y_cm
    tf.TensorSpec(()),  # time
))
tf_ds = tf_ds.padded_batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
(v, w, x), _x, y, z = next(iter(tf_ds))


# This is code to pair the actions into relations by reshaping.
@keras.saving.register_keras_serializable()
def make_pairs(x):
    # x: (B, M, T, E)
    M = tf.shape(x)[1]
    # Expand dims to prepare for broadcasting
    x1 = tf.expand_dims(x, axis=2)  # (B, M, 1, T, E)
    x2 = tf.expand_dims(x, axis=1)  # (B, 1, M, T, E)
    # Tile to get all pairs
    x1_tiled = tf.tile(x1, [1, 1, M, 1, 1])  # (B, M, M, T, E)
    x2_tiled = tf.tile(x2, [1, M, 1, 1, 1])  # (B, M, M, T, E)
    # Stack pair dimension
    pairs = tf.stack([x1_tiled, x2_tiled], axis=-2)  # (B, M, M, T, 2, E)
    return pairs


@keras.saving.register_keras_serializable()
def scale_broadcast(x):
    scale, embedding = x
    scale = tf.reshape(scale, (-1, 1, 1, 1))  # (batch, 1, 1, 1)
    return scale * tf.ones_like(embedding[..., :1])


def create_model():
    sequence_input = Input(shape=(None, TRAIN_SEQ_SIZE, FEATURES, 3))
    x_scale_input = Input(shape=(1,))
    y_scale_input = Input(shape=(1,))
    time_scale_input = Input(shape=(1,))
    x = Reshape((-1, TRAIN_SEQ_SIZE, FEATURES * 3))(sequence_input)
    # Embed the pos data
    x = Dense(EMBED_SIZE, activation='leaky_relu')(x)
    x = Dense(EMBED_SIZE, activation='leaky_relu')(x)

    # Concatenate real-life scale information and embed again
    x_scale_broadcast = Lambda(
        scale_broadcast,
        output_shape=(None, TRAIN_SEQ_SIZE, 1)
    )([x_scale_input, x])

    y_scale_broadcast = Lambda(
        scale_broadcast,
        output_shape=(None, TRAIN_SEQ_SIZE, 1)
    )([y_scale_input, x])

    time_scale_broadcast = Lambda(
        scale_broadcast,
        output_shape=(None, TRAIN_SEQ_SIZE, 1)
    )([time_scale_input, x])

    x = Concatenate(axis=-1)([x, x_scale_broadcast, y_scale_broadcast, time_scale_broadcast])
    x = Dense(EMBED_SIZE, activation='leaky_relu')(x)
    x = Dense(EMBED_SIZE, activation='leaky_relu')(x)
    # pairs
    x = Lambda(make_pairs)(x)
    x = Reshape((-1, TRAIN_SEQ_SIZE, 2 * EMBED_SIZE))(x)
    # Now that we have pairs, we can go conv
    # X kernel size is 1 so that we go through each mice pair individually
    # TimeDistributed maintains independence across relations
    r = Conv2D(filters=HIDDEN_SIZE,
               kernel_size=(1, 2),
               padding='same',
               activation='leaky_relu')(x)
    r = LayerNormalization()(r)
    x = Concatenate(axis=-1)([x, r])
    x = Conv2D(filters=HIDDEN_SIZE,
               kernel_size=(1, 8),
               padding='same',
               activation='leaky_relu')(x)
    x = LayerNormalization()(x)

    # End with logits
    x = Conv2D(filters=2*HIDDEN_SIZE,
               kernel_size=(1, KERNEL_SIZE),
               dilation_rate=4,
               padding='same',
               activation='leaky_relu')(x)
    x = Dense(NUM_BEHAVIORS)(x)
    return keras.Model(inputs=[sequence_input, x_scale_input, y_scale_input, time_scale_input], outputs=[x])


model = create_model()
model.summary()
out = model([v, _x, y, z])
print(f"Output Shape: {out.shape}")

class_weights = np.concatenate([[0.04], np.ones((NUM_BEHAVIORS - 1,))]).astype(np.float32)
class_weights = tf.convert_to_tensor(class_weights)


def weighted_loss(y_true, y_pred, mask):
    loss = tf.keras.losses.categorical_crossentropy(
        y_true, y_pred, from_logits=True
    )
    # This just gets the weight per time step, thanks chatgpt
    weights = tf.reduce_sum(y_true * class_weights, axis=-1)
    loss = loss * weights
    loss = loss * mask
    loss = tf.reduce_sum(loss)
    loss = loss/(tf.reduce_sum(mask) + eps)
    return loss


def masked_accuracy(y_true, y_pred, mask):
    pred_class = tf.argmax(y_pred, axis=-1)
    true_class = tf.argmax(y_true, axis=-1)
    correct = tf.cast(tf.equal(pred_class, true_class), tf.float32)
    correct = correct * mask  # only count masked positions
    return tf.reduce_sum(correct) / (tf.reduce_sum(mask) + 1e-8)


def masked_labeled_accuracy(y_true, y_pred):
    # ignores timesteps labeled nothing
    label_mask = tf.cast(1 - y_true[..., 0], tf.float32)
    pred_class = tf.argmax(y_pred, axis=-1)
    true_class = tf.argmax(y_true, axis=-1)
    correct = tf.cast(tf.equal(pred_class, true_class), tf.float32)
    correct = correct * label_mask  # only count masked positions
    return tf.reduce_sum(correct) / (tf.reduce_sum(label_mask) + 1e-8)


optimizer = keras.optimizers.Adam(3e-4)


@tf.function
def train_step(seq, label, mask, x_cm_scale, y_cm_scale, time_scale):
    with tf.GradientTape() as tape:
        pred = model([seq, x_cm_scale, y_cm_scale, time_scale])
        loss_val = weighted_loss(label, pred, mask)
    grads = tape.gradient(loss_val, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    acc = masked_accuracy(label, pred, mask)
    labeled_acc = masked_labeled_accuracy(label, pred)
    return loss_val, acc, labeled_acc


avg_epoch_losses = []
max_losses = []
min_losses = []
epoch_accuracies = []
epoch_labeled_accuracies = []
total_freq = tf.zeros((NUM_BEHAVIORS,))
try:
    for _epoch in range(EPOCHS):
        epoch = _epoch + 1
        avg_epoch_loss = 0
        min_epoch_loss = 10000.0
        max_epoch_loss = 0.0
        avg_epoch_acc = 0.0
        avg_epoch_labeled_acc = 0.0
        pbar = tqdm.tqdm(enumerate(tf_ds), total=num_videos)
        pbar.set_description(f'Epoch {epoch}')
        for step, ((seq, label, mask), x_cm_scale, y_cm_scale, time_scale) in pbar:
            seq = tf.convert_to_tensor(seq)
            label = tf.convert_to_tensor(label)
            mask = tf.convert_to_tensor(mask)
            x_cm_scale = tf.convert_to_tensor(x_cm_scale)
            y_cm_scale = tf.convert_to_tensor(y_cm_scale)
            time_scale = tf.convert_to_tensor(time_scale)
            loss_val, acc_val, labeled_acc_val = train_step(seq, label, mask, x_cm_scale, y_cm_scale, time_scale)
            avg_epoch_loss = (avg_epoch_loss * step + loss_val.numpy()) / (step + 1)
            min_epoch_loss = min(loss_val.numpy(), min_epoch_loss)
            max_epoch_loss = max(loss_val.numpy(), max_epoch_loss)
            avg_epoch_acc = (avg_epoch_acc * step + acc_val.numpy()) / (step + 1)
            avg_epoch_labeled_acc = (avg_epoch_labeled_acc * step + labeled_acc_val.numpy()) / (step + 1)
            pbar.set_postfix_str(f"avg loss: {avg_epoch_loss:.5f} labeled_acc: {avg_epoch_labeled_acc:.5f} "
                                 f"max loss: {max_epoch_loss:.5f} acc: {avg_epoch_acc:.4f}")
        avg_epoch_losses.append(avg_epoch_loss)
        min_losses.append(min_epoch_loss)
        max_losses.append(max_epoch_loss)
        epoch_accuracies.append(avg_epoch_acc)
        epoch_labeled_accuracies.append(avg_epoch_labeled_acc)
        if epoch % 1 == 0:
            model.save(f"./saved/model_{epoch}.keras")
except KeyboardInterrupt:
    pass

plt.figure(figsize=(10, 5))
plt.plot(avg_epoch_losses, label='avg loss')
plt.yscale('log')
plt.legend()
plt.title("Loss per Epoch")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(epoch_accuracies, label='masked accuracy', color='green')
plt.plot(epoch_labeled_accuracies, label='masked labeled accuracy', color='red')
plt.ylim(0, 1)
plt.title("Masked Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
