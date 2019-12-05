import torch
import torch.nn as nn
import pickle
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(0)

PATH = '../cifar_net.pth'
BATCH_SIZE = 4
MOMENTUM = 0.9
LEARNING_RATE = 0.001
EPOCH = 7

class SimpleNet(nn.Module):
	def __init__(self):
		super(SimpleNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

# class SimpleNet(nn.Module):

# 	def __init__(self, num_classes=10):
# 		super(SimpleNet, self).__init__()
# 		self.features = nn.Sequential(
# 			nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
# 			nn.ReLU(inplace=True),
# 			nn.MaxPool2d(kernel_size=3, stride=2),
# 			nn.Conv2d(64, 192, kernel_size=5, padding=2),
# 			nn.ReLU(inplace=True),
# 			nn.MaxPool2d(kernel_size=3, stride=2),
# 			nn.Conv2d(192, 384, kernel_size=3, padding=1),
# 			nn.ReLU(inplace=True),
# 			nn.Conv2d(384, 256, kernel_size=3, padding=1),
# 			nn.ReLU(inplace=True),
# 			nn.Conv2d(256, 256, kernel_size=3, padding=1),
# 			nn.ReLU(inplace=True),
# 			nn.MaxPool2d(kernel_size=3, stride=2),
# 		)
# 		self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
# 		self.classifier = nn.Sequential(
# 			nn.Dropout(0.5),
# 			nn.Linear(256 * 6 * 6, 4096),
# 			nn.ReLU(inplace=True),
# 			nn.Dropout(0.5),
# 			nn.Linear(4096, 4096),
# 			nn.ReLU(inplace=True),
# 			nn.Linear(4096, num_classes)
# 		)

# 	def forward(self, x):
# 		x = self.features(x)
# 		x = self.avgpool(x)
# 		x = x.view(x.size(0), 256 * 6 * 6)
# 		logits = self.classifier(x)
# 		probas = F.softmax(logits, dim=1)
# 		return logits, probas

# class Unit(nn.Module):
# 	def __init__(self,in_channels,out_channels):
# 		super(Unit,self).__init__()
# 		self.conv = nn.Conv2d(in_channels=in_channels,kernel_size=3,out_channels=out_channels,stride=1,padding=1)
# 		self.bn = nn.BatchNorm2d(num_features=out_channels)
# 		self.relu = nn.ReLU()

# 	def forward(self,input):
# 		output = self.conv(input)
# 		output = self.bn(output)
# 		output = self.relu(output)
# 		return output

# class SimpleNet(nn.Module):
# 	def __init__(self,num_classes=10):
# 		super(SimpleNet,self).__init__()

# 		#Create 14 layers of the unit with max pooling in between
# 		self.unit1 = Unit(in_channels=3,out_channels=32)
# 		self.unit2 = Unit(in_channels=32, out_channels=32)

# 		self.pool1 = nn.MaxPool2d(kernel_size=2)

# 		self.unit4 = Unit(in_channels=32, out_channels=64)
# 		self.unit5 = Unit(in_channels=64, out_channels=64)

# 		self.pool2 = nn.MaxPool2d(kernel_size=2)

# 		self.unit8 = Unit(in_channels=64, out_channels=128)
# 		self.unit9 = Unit(in_channels=128, out_channels=128)

# 		self.pool3 = nn.MaxPool2d(kernel_size=2)

# 		self.unit12 = Unit(in_channels=128, out_channels=128)
# 		self.unit13 = Unit(in_channels=128, out_channels=128)

# 		self.avgpool = nn.AvgPool2d(kernel_size=4)

# 		#Add all the units into the Sequential layer in exact order
# 		self.net = nn.Sequential(self.unit1, self.unit2, self.pool1, self.unit4, self.unit5, self.pool2, self.unit8, self.unit9, self.pool3,
# 		               self.unit12, self.unit13, self.avgpool)

# 		self.fc = nn.Linear(in_features=128,out_features=num_classes)

# 	def forward(self, input):
# 		output = self.net(input)
# 		output = output.view(-1,128)
# 		output = self.fc(output)
# 		return output


class CifarModel():
	def __init__(self):
		print("""
    		**********************
    		CIFAR DATASET
    		**********************
    	""")


	def load_cfar10_batch(self, name):
		with open('./cfar10/{}'.format(name), 'rb') as file:
			batch = pickle.load(file, encoding='bytes')
			features = batch[b'data'] #.reshape((len(batch[b'data']), 3, 32, 32)).transpose(0, 2, 3, 1)
			labels = batch[b'labels']
		return features, labels


	def load_cfar10_data(self):
		x1, y1 = self.load_cfar10_batch('data_batch_1')
		x2, y2 = self.load_cfar10_batch('data_batch_2')
		x3, y3 = self.load_cfar10_batch('data_batch_3')
		x4, y4 = self.load_cfar10_batch('data_batch_4')

		self.X_train = np.concatenate((x1,x2,x3,x4))
		self.y_train = y1+y2+y3+y4
		self.X_test, self.y_test = self.load_cfar10_batch('test_batch')

	def normalize_numpy(self, x):
		min_val = np.min(x)
		max_val = np.max(x)
		x = (x-min_val) / (max_val-min_val)
		return x

	# def train_and_test_dt(self):
	# 	dt_classifier = DTClassifier('./cfar10/data_batch')
	# 	model = Model(model_type=dt_classifier)
	# 	model.perform_experiment_for_cifar(self.X_train, self.X_test, self.y_train, self.y_test)

	def normalize_data(self):
		self.X_train_scaled = self.normalize_numpy(self.X_train)
		self.X_test_scaled = self.normalize_numpy(self.X_test)

	def reshape_and_convert_to_tensor(self):
		self.X_train_scaled_tensor = self.X_train_scaled.reshape(self.X_train_scaled.shape[0], 3, 32, 32)
		self.X_test_scaled_tensor = self.X_test_scaled.reshape(self.X_test_scaled.shape[0], 3, 32, 32)
		self.X_train_scaled_tensor = torch.from_numpy(self.X_train_scaled_tensor).float()
		self.X_test_scaled_tensor = torch.from_numpy(self.X_test_scaled_tensor).float()
		
		self.y_train_tensor = torch.tensor(np.array(self.y_train).astype(np.int64))
		self.y_test_tensor = torch.tensor(np.array(self.y_test).astype(np.int64))

	def train_and_save_cnn(self):
		self.load_cfar10_data()
		self.normalize_data()
		self.reshape_and_convert_to_tensor()

		net = SimpleNet()
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
		batch_size = BATCH_SIZE

		print("strating cnn training ===>")

		for epoch in range(EPOCH):  # loop over the dataset multiple times
			for i in range(0, len(self.X_train_scaled_tensor), batch_size):
				inputs = self.X_train_scaled_tensor[i:i+batch_size]   
				labels = self.y_train_tensor[i:i+batch_size]   

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward + backward + optimize
				outputs = net(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()

			print("Epoch {} final minibatch had loss {}".format(epoch, loss.item()))
		
		print('Finished cnn Training, Saving model')
		torch.save(net.state_dict(), PATH)

	def test_cnn(self):
		print("strating cnn testing ====>")
		net = SimpleNet()
		net.load_state_dict(torch.load(PATH))

		classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
		class_correct = [0. for i in range(10)]
		class_total = [0. for i in range(10)]
		correct = 0
		total = 0
		with torch.no_grad():
			for i in range(self.X_test_scaled_tensor.shape[0]):
				image =  self.X_test_scaled_tensor[i].reshape(1, 3,32,32)
				label = self.y_test_tensor[i]
				outputs = net(image)
				_, predicted = torch.max(outputs.data, 1)
				total += 1
				class_total[label] += 1
				if predicted == label:
					correct += 1
					class_correct[label] += 1

		print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
		for i in range(10):
			print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
