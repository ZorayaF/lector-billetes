import os
import tensorflow as tf
import tf2onnx
import onnx

print("1. Cargando modelo Keras...")
model = tf.keras.models.load_model("cop_model.keras")

print("2. Convirtiendo a formato universal ONNX...")
input_signature = [tf.TensorSpec([1, 224, 224, 3], tf.float32, name='input_image')]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)

output_path = "cop-app/public/cop_model.onnx"
os.makedirs("cop-app/public", exist_ok=True)
onnx.save(onnx_model, output_path)

print(f"-> ¡ÉXITO TOTAL! Modelo generado: {output_path} ({os.path.getsize(output_path) / (1024*1024):.2f} MB)")
