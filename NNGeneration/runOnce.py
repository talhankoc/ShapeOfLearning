
import SimpleNN as model
import sys

### model, node_count_list, epochs, divider
default_parameters = [[512],[256],[128], [96], [64], [48], [32], [24], [16], [12], [8], [4]]

model.feedFashionDataset()
root_directory = 'Saved Models/Fashion/'
divider = 1
try:
	divider = int(sys.argv[4])
except Exception as e:
	divider = 1
node_count_list = []
node_count_list.append(int(sys.argv[2]))

model.makeAndRunModel(node_count_list, None, e=int(sys.argv[3]), divider=divider, model_type=sys.argv[1])
