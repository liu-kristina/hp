from pathlib import Path

import numpy as np
import torch # for whatever reason necessary
import tensorflow as tf
import onnxruntime as rt
import openvino as ov
import numpy
import matplotlib.pyplot as plt
import transformers
from optimum.intel import OVModelForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM


from transformers import AutoTokenizer, pipeline


from cnn_test import get_data

# Huggingface test

# Saving the model
model_checkpoint = "mistralai/Mistral-Small-24B-Base-2501"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# RAM limited in WSL


# # Open Vino runtime
save_directory = "tmp/ov/"
ov_model = OVModelForCausalLM.from_pretrained(model_checkpoint, export=True)
ov_model.save_pretrained(save_directory)

# Onnx runtime
# save_directory = "tmp/onnx/"
# ort_model = ORTModelForCausalLM.from_pretrained(model_checkpoint, export=True)
# ort_model.save_pretrained(save_directory)

# laod model
model_directory = "tmp/ov/"
ov_model = OVModelForCausalLM.from_pretrained(model_directory)
pipe = pipeline("text-generation", model=ov_model, tokenizer=tokenizer, device="cpu")
result = pipe("He never went out without a book under his arm")
print(result)
# 
# pipe = pipeline("text-generation", model=ort_model, tokenizer=tokenizer)
# results = pipe("He's a dreadful magician and")

# tokenizer.save_pretrained(save_directory)

# To silent linters about unused package
# torch.cuda.is_available()

# class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#                'dog', 'frog', 'horse', 'ship', 'truck']

# MODEL_PATH = Path("models")
# sess = rt.InferenceSession(
#     Path(MODEL_PATH, "test.onnx"), providers=rt.get_available_providers())

# input_name = sess.get_inputs()[0].name

# print(sess.get_providers())
# train_images, test_images, train_labels, test_labels = get_data()

# test_image = test_images.astype(numpy.float32)[500, :, :, :]
# test_image = test_image[np.newaxis, :, :, :]

# input_name = sess.get_inputs()[0].name
# pred_onx = sess.run(None, input_feed={input_name: test_image})
# print(pred_onx)

# score = tf.nn.softmax(pred_onx[0])
# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )
# plt.xticks([])
# plt.yticks([])
# plt.grid(False)
# plt.imshow(np.squeeze(test_image))
# plt.xlabel(f"{class_names[np.argmax(score)]} ({100 * np.max(score):.2f}% confidence)")


# # OpenVino
# core = ov.Core()
# ov_model = ov.convert_model(Path(MODEL_PATH, "test.onnx"))
# compiled_model = core.compile_model(model=ov_model, device_name="CPU")
# ov.runtime.save_model(model=ov_model, output_model=Path(MODEL_PATH, "ov", "test.xml"))

# output_layer = compiled_model.output(0)
# result_infer = compiled_model([test_image])[output_layer]
# result_index = np.argmax(result_infer)

# score = tf.nn.softmax(pred_onx[0])
# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )
# plt.xticks([])
# plt.yticks([])
# plt.grid(False)
# plt.imshow(np.squeeze(test_image))
# plt.xlabel(f"{class_names[np.argmax(score)]} ({100 * np.max(score):.2f}% confidence)")

