import matplotlib.pyplot as plt
import pandas as pd
import re
import pickle
import os


def run(scores_fn, x_range, analysis_save_path, close_up_range=0):
	folder_name = 'ModelMetrics/'
	if not os.path.isdir(analysis_save_path + folder_name):
		os.mkdir(analysis_save_path + folder_name)

	with open(scores_fn, 'r') as f:
		import re
		dic = {}
		dic['loss'] = []
		dic['acc'] = []
		dic['val_loss'] = []
		dic['val_acc'] = []
		for line in f:  #Line is a string
			#split the string on whitespace, return a list of numbers 
			# (as strings)
			numbers_str = line.split()
			#convert numbers to floats
			numbers_float = [float(x) for x in numbers_str if re.match("^\d+?\.\d+?$", x) is not None]
			if len(numbers_float) == 0:
				continue
			elif len(numbers_float) != 4:
				print(numbers_float)
				raise 'Incorrect Score file format. SHould have 4 floats.'
			dic['acc'].append(numbers_float[0])
			dic['loss'].append(numbers_float[1])
			dic['val_acc'].append(numbers_float[2])
			dic['val_loss'].append(numbers_float[3])

	# print(len(dic['acc']))
	# print(len(dic['val_loss']))
	# assert False
	data = pd.DataFrame({'Epochs': x_range[close_up_range:], 'Train_acc':dic['acc'][close_up_range:], 'Test_acc':dic['val_acc'][close_up_range:]})
	plt.plot( 'Epochs', 'Train_acc', data=data, marker='o', markerfacecolor='red', markersize=0.5, color='orange', linewidth=0.25)
	plt.plot( 'Epochs', 'Test_acc', data=data, marker='o', markerfacecolor='green', markersize=0.5, color='green', linewidth=0.25)
	axes = plt.gca()
	plt.legend()
	save_name = f'{folder_name}accuracy_closeup{close_up_range}.png' if close_up_range != 0 else f'{folder_name}accuracy.png'
	plt.savefig(analysis_save_path + save_name, dpi=2000)
	plt.close()		

	data = pd.DataFrame({'Epochs': x_range[close_up_range:], 'Train_loss':dic['loss'][close_up_range:], 'Test_loss':dic['val_loss'][close_up_range:]})
	plt.plot( 'Epochs', 'Train_loss', data=data, marker='o', markerfacecolor='red', markersize=0.5, color='orange', linewidth=0.25)
	plt.plot( 'Epochs', 'Test_loss', data=data, marker='o', markerfacecolor='green', markersize=0.5, color='green', linewidth=0.25)
	axes = plt.gca()
	plt.legend()
	save_name = f'{folder_name}loss_closeup{close_up_range}.png' if close_up_range != 0 else f'{folder_name}loss.png'
	plt.savefig(analysis_save_path + save_name, dpi=2000)
	plt.close()		
