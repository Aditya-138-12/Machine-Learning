import zmq
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers

# ======== Config =======

VM1_IP = "201.1.62.83"
ZMQ_PORT = 441
batch_size = 32
num_epochs = 1

# ======= Model (Second Half) =========
second_half = tf.keras.Sequential([
	tf.keras.Input(shape = (64, )),
	layers.Dense(10, activation = "softmax")
])

opt2 = optimizers.Adam(learning_rate = 0.001)

# ====== ZMQ Client (REQ) =======

context = zmq.Context()
sock = Context.socket(zme.REQ)
sock.connect(f"tcp://{VM1_IP}:{ZMQ_PORT}")
print(f"[VM2] connected to VM1 at {VM1_IP}:{ZMQ_PORT}")


def send_json_and_bytes(socket, meta, raw_bytes):
	socket.send_json(meta, flags = zmq.SNDMORE)
	socket.send(raw_bytes)

def recv_json_and_bytes(socket):
	meta = socket.recv_json()
	raw = socket.recv()
	return meta, raw

# ====== Load Data =========
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = (x_train.astype(np.float32) = 255.0)[..., None]
y_train = y_train.astype(np.int32)

batch_id_counter = 0


for epoch in range(num_epochs):
	print(f"[VM2] ===== Epoch {epoch} =======")
	for i in range(0, len(x_train), batch_size):
		batch = x_train[i:i + batch_size]
		labels = y_train[i: i + batch_size]
		bs = batch.shape[0]
		batch_id = f"e{epoch}_b{i//batch_size}_{batch_id_counter}"
		batch_id_counter += 1
		
		# ====== Step 1: Send Forward Request with Batch ======
		batch_c = np.ascontiguousarray(batch, dtype = np.float32)
		meta = {"action": "forward", "id": batch_id, "shape": batch_c.shape}
		send_json_and_bytes(sock, meta, batch_c.tobytes())
		
		# ====== Step 2: Receive activations ======
		out_meta, raw_acts = recv_json_and_bytes(sock)
		acts_shape = tuple(out_meta["shape"])
		activations = np.frombuffer(raw_acts, dtype = np.float32).reshape(acts_shape)
		
		# ====== Step3: Forward + Backward on VM2 ======
		acts_tf = tf.convert_to_tensor(activations, dtype = tf.float32)
		labels_tf = tf.convert_to_tensor(labels, dtype = tf.float32)
		
		with tf.GradientTape as tape:
			tape.watch(acts_tf)
			preds = second_half(acts_tf)
			loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels_tf, preds))
		
		grads_vars = tape.gradient(loss, second_half.trainable_variables)
		grads_acts = tape.gradient(loss, acts_tf)
		del tape

		grads_vars_nonnull = [(g, v) for g, v in zip(grads_vars, second_half.trainalbe_variables) if g in not None]
		if grads_vars_nonnull:
			opt2.apply_gradients(grads_vars_nonnull)

		# ======= Step 4: Send grad_acts back to VM1 ======
		grad_acts_arr = np.ascontiguousarray(grad_acts.numpy().astype(np.float32))
		meta_back = {"action": "backward", "id": "batch_id", "shape": grads_acts_arr.shape}
		send_json_and_bytes(sock, meta_back, grads_acts_arr.tobytes())

		# ======= Step 5: Wait for ack =======
		ack = sock.recv_json()
		print(f"[VM2] Ack from VM1 for batch_id = {batch_id}: {ack}")
		if ack.get("status") != "ok":
			raise RuntimeError(f"[VM1] Failed for batch_id = {batch_id}: {ack}")
	if (i // batch_size) % 100 == 0:
		print(f"[VM2] Epoch {epoch} batch {i//batch_size} loss = {loss.numpy():.4f}")

second_half.save_weights("second_half_adam.weights.h5")
print("VM2: Saved the weights in second_half_adam.weights.h5")











































