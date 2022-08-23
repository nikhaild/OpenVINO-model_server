#
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import queue
import ovmsclient
from ovmsclient.tfs_compat.grpc.tensors import NP_TO_TENSOR_MAP
import numpy as np
from logger import get_logger
import multiprocessing
import cv2
#from pprint import pprint

class InferenceExecutor(multiprocessing.Process):
	
	def __init__(self, id, ovms_info, binary_input, input_queue, result_queue):
		multiprocessing.Process.__init__(self)

		self.abort_event = multiprocessing.Event()
		self.exit_event = multiprocessing.Event()
		self.exit_ready = multiprocessing.Event()

		self.logger = get_logger(f"Inference-Executor-{id}")
		self.id = id
		self.ovms_url = ovms_info["ovms_url"]
		self.model_name = ovms_info["model_name"]
		self.model_version = ovms_info["model_version"]
		self.binary_input = binary_input
		self.input_queue = input_queue
		self.result_queue = result_queue

	def shutdown(self):
		self.logger.debug(f"Shutting down Inference-Executor-{self.id}")
		self.exit_event.set()
		self.exit_ready.wait()
		self.logger.debug("Inference Executor thread stopped")
		self._flush_queues()
		self.logger.debug("Flushed Inference Executor IO buffers")

	def _flush_queues(self):
		while not self.input_queue.empty():
			self.input_queue.get_nowait()
		while not self.result_queue.empty():
			self.result_queue.get_nowait()

	def _make_ovms_call(self, func, *args):
		try:
			result = func(*args)
		except (ConnectionError, TimeoutError) as error:
			# TO DO: Consider retrying, just like with the stream reader
			self.logger.error(f"Could not connect to OVMS - {str(error)}")
			return None
		except ovmsclient.ModelServerError as error:
			self.logger.error(f"Call to OVMS resulted in error - {str(error)}")
			return None
		return result

	def _predict(self, ovms_client, inputs):
		for (input_name, input_data) in inputs.items():
			if input_name != 'image':
				continue
			frame = input_data 
			if self.binary_input:
				_, jpeg_encoded_frame = cv2.imencode('.jpeg', frame)
				input_data = jpeg_encoded_frame.tobytes()
			else:
				frame = np.expand_dims(frame, axis=0)
				input_data = ovmsclient.make_tensor_proto(frame, dtype=NP_TO_TENSOR_MAP[np.float32].TensorDtype)
			inputs[input_name] = input_data

		result = self._make_ovms_call(ovms_client.predict, inputs, self.model_name, self.model_version)
		return result

	def _get_model_metadata(self, ovms_client):
		model_metadata = self._make_ovms_call(ovms_client.get_model_metadata, self.model_name)

		if model_metadata is None:
			self.logger.info("Issuing abort signal...")
			self.abort_event.set()
			self.exit_event.wait()
			self.exit_ready.set()
			return None

		# Filter input param which will be synthesized.
		discard_inputs = ['image_info:0']
		model_metadata["inputs"] = {k: v for (k, v) in model_metadata["inputs"].items() if k not in discard_inputs}
		
		if len(model_metadata["outputs"]) > 1:
			self.logger.error("Unexpected number of model inputs or outputs. Expecting single input and single output.")
			self.logger.info("Issuing abort signal...")
			self.abort_event.set()
			self.exit_event.wait()
			self.exit_ready.set()
			return None
		return model_metadata

	def run(self):
		ovms_client = ovmsclient.make_grpc_client(self.ovms_url)
		model_metadata = self._get_model_metadata(ovms_client)
		if model_metadata is None:
			return
		

		while not self.exit_event.is_set():
			try:
				input = self.input_queue.get(timeout=1)
			except queue.Empty:
				continue
			inputs = {'image': input}
			if self.binary_input:
				inputs['image_info:0'] = [[255.0, 255.0, 255.0]]
			else:
				#pprint(input[0][0])
				input_data = np.expand_dims([input[0][0]], axis=0)
				#pprint(input_data[0])
				inputs['image_info:0'] = ovmsclient.make_tensor_proto(input_data[0], dtype=NP_TO_TENSOR_MAP[np.float32].TensorDtype)
			#pprint(inputs)
			result = self._predict(ovms_client, inputs)

			if result is None:
				self.logger.info("Issuing abort signal...")
				self.abort_event.set()
				self.exit_event.wait()
				continue

			self.result_queue.put((input, result))
		self.exit_ready.set()
		return
