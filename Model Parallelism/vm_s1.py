import zmq
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers

# ======= Model  (First Half) =======

first_half = tf.keras.Sequential([
	layers.Flatten(input_shape = (28, 28, 1)),
	layers.Dense(128, activation = "relu"),
	layers.Dense(64, activation = "relu")
])

opt1 = optimizers.Adam(learning_rate = 0.001)

# ======= ZMQ Server (REP) ============

ZMQ_PORT = 441
context = zmq.Context()
sock = context.socket(zmq.REP)
sock.bind(f"tcp://0.0.0.0:{ZMQ:PORT}")
print(f"[VM1] Listening on port {ZMQ_PORT}")

batch_cache = {}

def recv_json_and_bytes(socket):
	meta = socket.recv_json()
	raw = socket.recv()
	return meta, raw

def send_json_and_bytes(socket, meta, raw_bytes):
	socket.send_json(meta, flags = zmq.SNDMORE)
	socket.send(raw_bytes)

while True:
	meta, raw = recv_json_and_bytes(sock)

	action = meta.get("action")
	batch_id = meta.get("id")

	print(f"[VM1] Received action = {action} for batch_id = {batch_id}", f"bytes = {len(raw)} shape = {meta.get('shape')}")
	
	if action = "forward":
		batch_shape = tuple(meta["shape"])
		batch = np.frombuffer(raw, dtype = np.float32).reshape(batch_shape)
		batch_cache[batch_id] = batch
		
		activations = first_half(batch).numpy()
		out_meta = {"id": batch_id, "shape": activations.shape}
		
		send_json_and_bytes(sock, out_meta, activations.tobytes())


	elif action == "backward":
		grad_shape = tuple(meta["shape"])
		grad_acts = np.frombuffer(raw, dtype = np.float32).reshape(grad_shape)
		
		if batch_id not in batch_cache:
			print(f"[VM1] ERROR: batch_id {batch_id} not found in cache")
			sock.send_json("status": "error", "msg": "batch not found")
			continue
		
		batch = batch_cache.pop(batch_id)

		batch_tf = tf.convert_to_tensor(batch, dtype = tf.float32)
		with tf.GradientTape as tape:
			outputs = first_half(batch_tf)
		
		grads = tape.gradient(outputs, first_half.trainable_variables, output_gradients = tf.convert_to_tensor(grad_acts, dtype = tf.float32))
		
		grads_vars = [(g, v) for g,v in zip(grads, first_half.trainable_variables) if g is not None]
		if grads_vars:
			opt1.apply_gradients(grad_vars)
		
		sock.send_json({"status": "ok"})
	else:
		print(f"[VM1] ERROR: Unknown action {action}")
		sock.send_json({"status": "error", "msg": "Unknown action"})


first_half.save_weights("first_half_adam.weights.h5")
print("VM1: Saved the weights to first_half.weights.h5")








































