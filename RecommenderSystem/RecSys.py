import numpy as np

class RecSys:
	#Constructor
	def __init__(self, name2id, id2name, rating_generator, k, reg, eta, epochs):
		len_user = 73517
		len_w = 34528
		self.U = self.get_randvec(len_user,k)
		self.W = self.get_randvec(len_w,k)
		self.k = k
		self.name2id = name2id
		self.id2name = np.array(id2name)
		self.know = dict()
		self.uD = dict()
		self.wD = dict()
		self.rating_generator = rating_generator
		self.process_ratings()
		self.calcule_mean()
		self.reg = reg
		self.eta = eta
		self.epochs = epochs

	#Build init vector for user or item
	def get_randvec(self, N, D):
		return np.random.randn(N,D) / D

	#Get valid item rates
	def process_ratings(self):
		data = next(self.rating_generator)
		know = self.know
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

	def calcule_mean(self):
		self.user_mean = dict()
		#For each item, calculate mean and subtract on know vector
		for uid in self.uD:
			sumv = 0
			for wid in self.uD[uid]:
				sumv += self.know[uid,wid]
			sumv /= len(self.uD[uid])
			self.user_mean[uid] = sumv

	def cost(self, uid, wid, uorw):
		error = np.dot(self.U[uid],self.W[wid].T) + self.user_mean[uid] - self.know[uid,wid]
		loss = 0.5 * error**2
		if uorw == 0:
			regularization = 0.5 * self.reg * np.linalg.norm(self.U[uid])**2
		else:
			regularization = 0.5 * self.reg * np.linalg.norm(self.W[wid])**2
		return loss + regularization, error

	def train(self):
		tloss = 0.0

		#For each user
		for uid in self.uD:
			#Animes that rate for uid:
			Ruw = self.uD[uid]
			grad = np.zeros(self.k).astype(np.float)
			loss = 0.0
			for wid in Ruw:
				temp, error = self.cost(uid,wid,0)
				grad += error*self.W[wid] + self.reg*self.U[uid]
				loss += temp
			grad = grad / len(Ruw)
			loss /= len(Ruw)
			tloss += loss
			self.U[uid] -= self.eta*grad

		#For each item
		for wid in self.wD:
			#User that rate the anime wid:
			Ruw = self.wD[wid]
			grad = np.zeros(self.k).astype(np.float)
			loss = 0.0
			for uid in Ruw:
				temp, error = self.cost(uid,wid,1)
				grad += error*self.U[uid] + self.reg*self.W[wid]
				loss += temp
			grad = grad / len(Ruw)
			loss /= len(Ruw)
			tloss += loss
			self.W[wid] -= self.eta*grad


		return tloss / len(self.uD)

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
		return np.dot(a,b) + self.user_mean[user]

	def evaluate(self, log=False):
		loss = 0.0
		for user, item in self.know:
			y = self.know[user,item]
			y_= self.predict(user,item)
			loss += (y-y_)**2
			if log is True:
				print("User({0}) and Item({1}):".format(user, self.id2name[item]))
				print("Rate = {0}, Predict = {1}".format(y,y_))
				print("Diff = {0}\n".format(abs(y-y_)))
		return (loss/len(self.know))**(0.5)

	def save(self, directory):
		np.save(directory+"/user", self.U)
		np.save(directory+"/item", self.W)

	def load(self, directory):
		self.U = np.load(directory+"/user.npy")
		self.W = np.load(directory+"/item.npy")