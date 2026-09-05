import numpy as np
import onnxruntime as ort
from PIL import Image

# --- ArcFace ---
img = Image.open("/workspace/calib112/000000.jpg").convert("RGB").resize((112,112))
arr = np.array(img).astype(np.float32)
arr = (arr - 127.5) / 127.5
arr = arr.transpose(2,0,1)[None, ...]  # NCHW

sess = ort.InferenceSession("/workspace/arcface_mbf_static.onnx", providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
out_name = sess.get_outputs()[0].name
ort_out = sess.run([out_name], {in_name: arr})[0].reshape(-1)

def cosine(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b)))

bf16_npz = np.load("/workspace/out_bf16.npz")
print("BF16 npz keys/shapes:", [(k, bf16_npz[k].shape) for k in bf16_npz.keys()])
bf16_out = bf16_npz["516_BatchNormalization_f32"]

print("ArcFace onnxruntime output shape:", ort_out.shape)
print("ArcFace BF16 cvimodel output shape:", bf16_out.shape)
print("ArcFace cosine(onnxruntime, BF16) =", cosine(ort_out, bf16_out))

# INT8 cvimodel was not produced (tolerance check failed before codegen) -> skip
print("ArcFace INT8: cvimodel not generated (model_deploy aborted at tolerance check, cosine=0.972985/euclid=0.762330 < required 0.95/0.8) -- no comparison possible")

# --- Liveness (BGR, no normalization) ---
img2 = Image.open("/workspace/calib112/000000.jpg").convert("RGB").resize((80,80))
arr2 = np.array(img2).astype(np.float32)
arr2 = arr2[:, :, ::-1]  # RGB->BGR
arr2 = arr2.transpose(2,0,1)[None, ...]

sess2 = ort.InferenceSession("/workspace/liveness_minifasnet.onnx", providers=["CPUExecutionProvider"])
in_name2 = sess2.get_inputs()[0].name
out_name2 = sess2.get_outputs()[0].name
ort_out2 = sess2.run([out_name2], {in_name2: arr2})[0].reshape(-1)

live_bf16 = np.load("/workspace/out_live_bf16.npz")
print("Liveness BF16 npz keys/shapes:", [(k, live_bf16[k].shape) for k in live_bf16.keys()])
live_bf16_out = live_bf16["logits_Flatten_f32"] if "logits_Flatten_f32" in live_bf16 else live_bf16["logits_Flatten"]

print("Liveness onnxruntime output shape:", ort_out2.shape)
print("Liveness BF16 cvimodel output shape:", live_bf16_out.shape)
print("Liveness cosine(onnxruntime, BF16) =", cosine(ort_out2, live_bf16_out))
