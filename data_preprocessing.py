import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
import json
import os

import config


def extract_save_log_mel_FBE(data_path,json_path,sample_rate,n_mels,n_fft, hop_length):
	
	data = {"mapping": [],"labels": [],"logmel": []}
	
	for i, (root, dirnames, filenames) in enumerate(os.walk(data_path)):

		if root is not data_path:
			semantic_label = root.split("/")[-1]
			data["mapping"].append(semantic_label)
			print("\nProcessing: {}".format(semantic_label))

			for f in filenames:
				file_path = os.path.join(root, f)
				print(file_path)
				signal, fs = librosa.load(file_path, sr=sample_rate)
				print(signal.shape)
				if signal.shape[0]<16000:
					pad_length=16000-signal.shape[0]
					signal=np.pad(signal, (0, pad_length), 'constant', constant_values=(0.0,0.0))
	
				mel_spec = librosa.feature.melspectrogram(signal, fs, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
				log_mel_spec= librosa.power_to_db(mel_spec)
				log_mel_spec = log_mel_spec.T
				

				data["logmel"].append(log_mel_spec.tolist())
				data["labels"].append(i-1)

	# save Log_mel_filter_bank_energies to json file
	os.makedirs(config.path_to_json,exist_ok=True)
	json_filename='data_logmel.json'
	with open(os.path.join(config.path_to_json,json_filename), "w") as fp:
		json.dump(data, fp, indent=4)			
	

	print(np.array(data['logmel']).shape)
	print(np.array(data['labels']).shape)
	print(data['labels'])




    			

extract_save_log_mel_FBE(config.path_to_data,config.path_to_json,config.sample_rate,config.n_mels,config.n_fft, config.hop_length)
