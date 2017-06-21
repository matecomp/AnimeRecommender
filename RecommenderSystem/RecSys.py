import numpy as np

class RecSys:
	#Constructor
	def __init__(self, name2id, id2name, rating_generator, k, reg, eta, epochs):
		len_user = 73517
		len_w = 34528
		self.U = self.get_randvec(len_user,k)
		self.Bu = np.zeros((len_user,1))
		self.W = self.get_randvec(len_w,k)
		self.Bw = np.zeros((len_w,1))
		self.k = k
		self.name2id = name2id
		self.id2name = np.array(id2name)
		self.know_train = dict()
		self.know_test = dict()
		self.uD = dict()
		self.wD = dict()
		self.rating_generator = rating_generator
		self.process_train()
		self.process_train()
		self.process_test()
		self.calcule_mean()
		self.reg = reg
		self.eta = eta
		self.epochs = epochs

	#Build init vector for user or item
	def get_randvec(self, N, D):
		return np.random.randn(N,D) / D

	#Get valid item rates
	def process_train(self):
		data = next(self.rating_generator)
		know = self.know_train
		uD = self.uD
		wD = self.wD
		for rate in data:
			uid = int(rate[0])
			wid = int(rate[1])
			value = float(rate[2])
			if value != -1:
				know[uid,wid] = value
				if uD.get(uid) is None:
					uD[uid] = []
				uD[uid].append(wid)
				if wD.get(wid) is None:
					wD[wid] = []
				wD[wid].append(uid)

	#Get test data
	def process_test(self):
		data = next(self.rating_generator)
		know = self.know_test

		for rate in data:
			uid = int(rate[0])
			wid = int(rate[1])
			value = float(rate[2])
			if value != -1:
				know[uid,wid] = value

	def calcule_mean(self):
		self.user_mean = dict()
		self.global_mean = 0.0
		uD = self.uD
		know = self.know_train
		count = 0.0
		#For each item, calculate mean and subtract on know vector
		for uid in self.uD:
			sumv = 0
			for wid in uD[uid]:
				sumv += know[uid,wid]
				count += 1
			self.global_mean += sumv
			sumv /= len(uD[uid])
			self.user_mean[uid] = sumv

		self.global_mean /= count

	def cost(self, uid, wid, uorw):
		# get references
		U = self.U[uid]
		Bu = self.Bu[uid]
		W = self.W[wid]
		Bw = self.Bw[wid]
		user_mean = self.user_mean[uid]
		global_mean = self.global_mean
		know = self.know_train[uid,wid]
		reg = self.reg

		error = np.dot(U,W.T) + global_mean + Bu + Bw - know
		loss = 0.5 * error**2
		if uorw == 0:
			regularization = 0.5 * reg * np.linalg.norm(U)**2
		else:
			regularization = 0.5 * reg * np.linalg.norm(W)**2
		return loss + regularization, error

	def train(self):

		uD = self.uD
		Bu = self.Bu
		wD = self.wD
		Bw = self.Bw
		k = self.k
		U = self.U
		W = self.W
		reg = self.reg
		eta = self.eta
		tloss = 0.0

		#For each user
		for uid in uD:
			#Animes that rate for uid:
			Ruw = uD[uid]
			grad = np.zeros(k).astype(np.float)
			gradB = 0.0
			loss = 0.0
			for wid in Ruw:
				temp, error = self.cost(uid,wid,0)
				grad += error*W[wid] + reg*U[uid]
				gradB += error
				loss += temp
			grad = grad / len(Ruw)
			gradB = gradB / len(Ruw)
			loss /= len(Ruw)
			tloss += loss
			U[uid] -= eta*grad
			Bu[uid] -= eta*gradB

		#For each item
		for wid in wD:
			#User that rate the anime wid:
			Ruw = wD[wid]
			grad = np.zeros(k).astype(np.float)
			gradB = 0.0
			loss = 0.0
			for uid in Ruw:
				temp, error = self.cost(uid,wid,1)
				grad += error*U[uid] + reg*W[wid]
				gradB += error
				loss += temp
			grad = grad / len(Ruw)
			gradB = gradB / len(Ruw)
			loss /= len(Ruw)
			tloss += loss
			W[wid] -= eta*grad
			Bw[wid] -= eta*gradB


		return tloss / (len(uD) + len(wD))

	#Unsuperved learning... Don't need to y
	def fit(self):
		epochs = self.epochs
		for epoch in range(epochs):
			loss = self.train()
			if(epoch%1 == 0):
				print("Epoch {0} : loss {1}".format(epoch,loss))

	def item_neighbors(self, item_name, samples=5):
		item_id = self.name2id[item_name]
		item_vector = self.W[item_id]
		normalize_dots =  np.dot(item_vector,self.W.T) 
		normalize_dots /= np.linalg.norm(self.W,axis=1)
		neighbors = np.argsort(normalize_dots)[::-1]
		return self.id2name[neighbors[1:samples]]


	def predict(self, user, item):
		a = self.U[user]
		b = self.W[item]
		return np.dot(a,b) + self.global_mean + self.Bu[user] + self.Bw[item]

	def evaluate(self, test=False, log=False):
		if test is True:
			know = self.know_test
		else:
			know = self.know_train

		loss = 0.0
		count = 0
		for user, item in know:
			if self.uD.get(user) is None:
				continue
			if self.wD.get(item) is None:
				continue
			count += 1
			y = know[user,item]
			y_= self.predict(user,item)
			loss += (y-y_)**2
			if log is True:
				print("User({0}) and Item({1}):".format(user, self.id2name[item]))
				print("Rate = {0}, Predict = {1}".format(y,y_))
				print("Diff = {0}\n".format(abs(y-y_)))
		return (loss/count)**(0.5)

	def save(self, directory):
		np.save(directory+"/user", self.U)
		np.save(directory+"/item", self.W)

	def load(self, directory):
		self.U = np.load(directory+"/user.npy")
		self.W = np.load(directory+"/item.npy")