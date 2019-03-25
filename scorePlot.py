import matplotlib.pyplot as plt
import pandas as pd
import re
import pickle


def run(scores_fn, x_range, analysis_save_path, close_up_range=0):



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
				raise 'Incorrect Score file format. SHould have 4 floats.'
			dic['acc'].append(numbers_float[0])
			dic['loss'].append(numbers_float[1])
			dic['val_acc'].append(numbers_float[2])
			dic['val_loss'].append(numbers_float[3])


	data = pd.DataFrame({'Epochs': x_range[close_up_range:], 'Train_acc':dic['acc'][close_up_range:], 'Test_acc':dic['val_acc'][close_up_range:]})
	plt.plot( 'Epochs', 'Train_acc', data=data, marker='o', markerfacecolor='red', markersize=4, color='orange', linewidth=2)
	plt.plot( 'Epochs', 'Test_acc', data=data, marker='o', markerfacecolor='green', markersize=4, color='orange', linewidth=2)
	axes = plt.gca()
	plt.legend()
	save_name = f'accuracy_closeup{close_up_range}.png' if close_up_range != 0 else 'accuracy.png'
	plt.savefig(analysis_save_path + save_name, dpi=400)
	plt.close()		

	data = pd.DataFrame({'Epochs': x_range[close_up_range:], 'Train_loss':dic['loss'][close_up_range:], 'Test_loss':dic['val_loss'][close_up_range:]})
	plt.plot( 'Epochs', 'Train_loss', data=data, marker='o', markerfacecolor='red', markersize=4, color='orange', linewidth=2)
	plt.plot( 'Epochs', 'Test_loss', data=data, marker='o', markerfacecolor='green', markersize=4, color='orange', linewidth=2)
	axes = plt.gca()
	plt.legend()
	save_name = f'loss_closeup{close_up_range}.png' if close_up_range != 0 else 'loss.png'
	plt.savefig(analysis_save_path + save_name, dpi=400)
	plt.close()		
