import tensorflow as tf
import numpy as np

try:
    interpreter = tf.lite.Interpreter(model_path="/Users/safamoqbel/Downloads/gsr/model.tflite")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)

# Correctly create the input array with the correct shape.
user_input = float(input("Enter GSR resistance value (e.g., 0.75): "))
test_value = np.array([user_input] * 30, dtype=np.float32)
input_shape = input_details[0]['shape']
test_value = test_value.reshape(input_shape)

try:
    interpreter.set_tensor(input_details[0]['index'], test_value)
except Exception as e:
    print(f"Error setting input tensor: {e}")
    exit()

interpreter.invoke()

try:
    output = interpreter.get_tensor(output_details[0]['index'])
except Exception as e:
    print(f"Error getting output tensor: {e}")
    exit()

print("Model output:", output)

threshold_low = 0.4
threshold_high = 0.6
output_value = output[0][0]

if output_value < threshold_low:
    print("Relaxed")
elif output_value > threshold_high:
    print("Under Stress")
else:
    print("Uncertain / Intermediate")
