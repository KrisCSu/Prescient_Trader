import numpy as np
import csv
import tkinter as tk
from tkinter import ttk
import tkinter.messagebox
from tkinter import *
import tkinter.font
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import collections
from sklearn import preprocessing
from neupy import algorithms, environment
from pprint import pprint
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from multiprocessing import Process,Queue
from sklearn.metrics import explained_variance_score
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from mpl_finance import candlestick2_ohlc
import datetime

#Input: Tker symbol
#Output: Returns true if the ticker is a valid symbol in our ticker list
def tickers( tker):
	with open('tickers.csv') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			if row['Ticker'] == tker:
				return True
			elif row['Name'] == tker:
				return True

#Input: Takes in the ticker symbol of the stuck, algorithm chosen by the user, time interval, whether the user wants to do a tDF, what 
# technical indicators to use, whether to include Price, what type of Class, and a list to return the outputs
#Output: Sends the necessary variables along with the data this method retrieved to the PredictAndTest method
def getData(tker,alg,time,tDF,tI,incP,typeClass,return_list):

	print("Getting Data")

	dataDict= []
	symb = tker
	#Do not use 60min since that does not work.
	#Cannot figure out how, but the keys for date and time are different.
	tInterval = time
	ourKey = '2ONP54KRQ2G73D3K'

	#Using the wrapper class to get the different data for the inputted ticker
	#In a try-catch because sometime the API fails
	try:
		#Making the retry number high so the API call should not fail
		ts = TimeSeries(key = ourKey, output_format = 'json',retries=10000)
		#tInterval has first letter capitalized when getting input for neatness, it is then lowered here for the API call
		#Here we are getting the price data for the requested time interval
		if tInterval == 'Daily':
			priceData, meta_priceData = ts.get_daily(symbol= symb, outputsize='full')
			tInterval = 'daily'
		elif tInterval == 'Weekly':
			priceData, meta_priceData = ts.get_weekly(symbol = symb)
			tInterval = 'weekly'
		elif tInterval =='Monthly':
			priceData, meta_priceData = ts.get_monthly(symbol = symb)
			tInterval = 'monthly'
		else:
			priceData, meta_priceData = ts.get_intraday(symbol = symb, interval = tInterval, outputsize='full')
			#Intraday date and time key includes seconds, while it does not for technicial indicators
			for day in priceData.copy():
				priceData[day[:-3]] = priceData.pop(day)


		ti = TechIndicators(key = ourKey, output_format ='json')
		#Here we get the technical indicators the user chose from the menu, unless they chose to do a tDF.
		if tDF == "No":
			if 0 in tI:
				macD, meta_macD = ti.get_macd(symbol = symb, interval = tInterval,  series_type = 'close')
				dataDict.append(macD)
			if 1 in tI:
				stoch, meta_stoch = ti.get_stoch(symbol = symb, interval = tInterval, fastkperiod = '5')
				dataDict.append(stoch)
			if 2 in tI:
				ema, meta_ema = ti.get_ema(symbol = symb, interval = tInterval, time_period = '8', series_type = 'close')
				dataDict.append(ema)
			if 3 in tI:
				rsi, meta_rsi = ti.get_rsi(symbol = symb, interval = tInterval, time_period = '14', series_type='close')
				dataDict.append(rsi)
			if 4 in tI:
				plusDI, meta_plusDI = ti.get_plus_di(symbol = symb, interval = tInterval, time_period = '14')
				minusDI, meta_minusDI = ti.get_minus_di(symbol = symb, interval = tInterval, time_period = '14')
				dataDict.append(plusDI)
				dataDict.append(minusDI)
			if 5 in tI:
				cci, meta_cci = ti.get_cci(symbol = symb, interval = tInterval, time_period = '14')
				dataDict.append(cci)
			if 6 in tI:
				willR, meta_willR = ti.get_willr(symbol = symb, interval = tInterval, time_period = '14')
				dataDict.append(willR)
			if 7 in tI:
				oBV, meta_OBV = ti.get_obv(symbol = symb, interval = tInterval)
				dataDict.append(oBV)
			if 8 in tI:
				aD, meta_AD = ti.get_ad(symbol = symb, interval = tInterval)
				dataDict.append(aD)
			if 9 in tI:
				bBands, meta_bBands = ti.get_bbands(symbol = symb, interval = tInterval, time_period = '14', series_type = 'close')
				dataDict.append(bBands)
			if 10 in tI:
				aroon, meta_aroon = ti.get_aroon(symbol = symb, interval = tInterval, time_period = '14')
				dataDict.append(aroon)
			if 11 in tI:
				adx, meta_adx = ti.get_adx(symbol = symb, interval = tInterval, time_period = '14')
				dataDict.append(adx)
			if(incP == "Yes"):
				dataDict.append(priceData)
		#Since a tDF requires a manual transformation, we choose the technical indicators for it, which are retrieved here
		else:
			macD, meta_macD = ti.get_macd(symbol = symb, interval = tInterval,  series_type = 'close')
			stoch, meta_stoch = ti.get_stoch(symbol = symb, interval = tInterval, fastkperiod = '5')
			ema, meta_ema = ti.get_ema(symbol = symb, interval = tInterval, time_period = '8', series_type = 'close')
			rsi, meta_rsi = ti.get_rsi(symbol = symb, interval = tInterval, time_period = '14', series_type='close')
			plusDI, meta_plusDI = ti.get_plus_di(symbol = symb, interval = tInterval, time_period = '14')
			minusDI, meta_minusDI = ti.get_minus_di(symbol = symb, interval = tInterval, time_period = '14')
			cci, meta_cci = ti.get_cci(symbol = symb, interval = tInterval, time_period = '14')
			willR, meta_willR = ti.get_willr(symbol = symb, interval = tInterval, time_period = '14')
			dataDict.append(macD)
			dataDict.append(stoch)
			dataDict.append(ema)
			dataDict.append(rsi)
			dataDict.append(plusDI)
			dataDict.append(minusDI)
			dataDict.append(cci)
			dataDict.append(willR)
	#Returning with nothing tells our main process that there was an error
	except ValueError:
		return

	#Sometimes the API may return a empty list which will cause an error for the training
	for list in dataDict:
		if not list:
			return

	#Need atleast one of the dictionaries to be ordered by date, so we can organize all our data.
	priceData = collections.OrderedDict(sorted(priceData.items()))

	#dates used for drawing graph afterwards
	dates = []
	for day in priceData:
		dates.append(day)

	#Parsing the data
	#All the data for each day is put into an array, which is put into trainingData, sorted by dates
	trainingData = []
	#Putting the ordered price data into priceList, instead of using a Dictionary
	#So if the user doesn't want to include price in training, we still have a list to train
	#for the actual outputs/
	priceList = []
	#If the user is using this in the middle of the day/week/month, the data will have a partial time interval up to the current time
	#Here we remove the last day if it is the same date as today, so we can get a guess for the timer interval asked.
	todaysDate = str(datetime.datetime.now())[:10]
	if todaysDate in priceData.keys():
		del priceData[todaysDate]

	for day in priceData:
		#Making sure the day is in the priceData and each of the technical indicator lists.
		#The API doesn't necessarily give the data for every data for the different requests
		if all(day in indic for indic in dataDict):
			vector = []
			priceData[day] = collections.OrderedDict(sorted(priceData[day].items()))
			for stat in priceData[day]:
				vector.append(float(priceData[day][stat]))
			priceList.append(vector)

			vector = []
			for x in range(0,len(dataDict)):
			#Order the different stats within a data, so the order is consistent across vectors
				dataDict[x][day] = collections.OrderedDict(sorted(dataDict[x][day].items()))
			#Combine all the different data into a vector
			for x in range(0,len(dataDict)):
				for stat in dataDict[x][day]:
					vector.append(float(dataDict[x][day][stat]))
			trainingData.append(vector)

	#If the user wanted to do a tDF, the training data is transformed, otherwise the trainingData goes straight to PredictAndTest
	trendDeter = False if tDF == "No" else True
	#delete first element of priceList because trendDeter, remove first element form training data
	if trendDeter:
		del priceList[0]
		predictAndTest(trendDeterTransform(trainingData), priceList, trendDeter, alg, tInterval, dates, typeClass, return_list)
	else:
		predictAndTest(trainingData, priceList, trendDeter, alg, tInterval, dates, typeClass, return_list)

#Input: Training data for a algorithm. Assumes the data is in a certain order.
#Transforms the different values to 1 or -1 depending on if the technical indicator implies the stock is going up or down.
#Output: An array of arrays with 1's and -1's for training.
def trendDeterTransform(trainingData):
	trendTrainingData = []
	#Starting at 1, because to determine trend for most indicators need to compare current value with a previous value
	for x in range(1,len(trainingData)):
		vector = []
		#0 is macd
		vector.append(-1 if (trainingData[x][0]-trainingData[x-1][0]<0) else 1)
		#skipping 1 because its macd_signal, which is not clear on determinig trend
		#2 is mac_d hist, which is difference between signal line and macd
		vector.append(-1 if (trainingData[x][2]-trainingData[x-1][2]<0) else 1)
		#3-4 is stoch slowk and slowd
		vector.append(-1 if (trainingData[x][3]-trainingData[x-1][3]<0) else 1)
		vector.append(-1 if (trainingData[x][4]-trainingData[x-1][4]<0) else 1)
		#5 is ema
		vector.append(-1 if (trainingData[x][5]-trainingData[x-1][5]<0) else 1)
		#6 is rsi
		#overbought zone
		if trainingData[x][6]>70:
			vector.append(-1)
		#oversold zone
		elif trainingData[x][6]<30:
			vector.append(1)
		else:
			vector.append(-1 if (trainingData[x][6]-trainingData[x-1][6]<0) else 1)
		#7 and 8 are plus and minus DI respectively
		vector.append(-1 if (trainingData[x][7]-trainingData[x][8]<0) else 1)
		#9 is cci
		#overbought zone
		if trainingData[x][9]>200:
			vector.append(-1)
		#oversold zone
		elif trainingData[x][9]<-200:
			vector.append(1)
		else:
			vector.append(-1 if (trainingData[x][9]-trainingData[x-1][9]<0) else 1)
		#10 is william %r
		vector.append(-1 if (trainingData[x][10]-trainingData[x-1][10]<0) else 1)

		trendTrainingData.append(vector)
	return trendTrainingData

#Input: Takes in an array of arrays, which should have the data to train for each day. A price list, which are meant to show the outcomes
# of each day. a Boolean for tDF, whether a user chose to do it or not. Which algorithm the user chose. The dates for the graph in the 
# results page. What type of class the user chose. The return_list, which will have the outcomes in it.
def predictAndTest(trainingData, priceList, trendDeter, alg, tInterval, dates, typeClass, return_list):

	print("Training")

	#True if we want a prediction for if the stock will close higher tomorrow than it will open tomorrow.
	#False if we want a prediction for if the stock will close higher tomorrow than it closed today.
	openToClose = True if typeClass == "Open to Close" else False

	#Keeping track prediction accuracy
	total = 0
	correct = 0

	#Performing about 50 tests
	for num in range(len(trainingData)-50, len(trainingData)-1):

		#Training with around 750 datapoints of stock info (about 3 years if using daily)
		#As long as the stock has that many data points (depending on when it went public)
		#Using all datapoints increases runtime. This variable has the array index for the first day.
		firstDay = 0 if len(trainingData)<750 else len(trainingData)-750

		#Making a new list with a subset of all data, to train for a prediction for  the following day (after the subsets data)
		#The list is from firstDay, up to length of trainingData - num (the counter loop variable)
		scaledTrainingData = []
		for day in range(firstDay,num):
			scaledTrainingData.append(trainingData[day])
		#If using trend determinisitic transformation, data is already transformed, no need to scale
		if not trendDeter:
			scaledTrainingData = preprocessing.scale(scaledTrainingData)

		#Categorical output, a list of whether the stock went up or down
		#0 if stock went down, 1 if it went up
		catOutput = []
		#Starting at the day after firstDay because firstDay results in that day
		for day in range(firstDay+1,num+1):
			if openToClose:
				catOutput.append(0 if (priceList[day][3]-priceList[day][0]<0) else 1 )
			else:
				catOutput.append(0 if (priceList[day][3]-priceList[day-1][3]<0) else 1)

		#The algorithms take in a list to make predictions.
		#We give them one input, which is the following day after all the training examples fed in.
		predTD = []
		if not trendDeter:
			predTD.append(preprocessing.scale(trainingData[num]))
		else:
			predTD.append(trainingData[num])

		#The actual value of if the stock went up or down, to compare to the prediction.
		if openToClose:
			actual = 0 if (priceList[num+1][3]-priceList[num+1][0]<0) else 1
		else:
			actual = 0 if (priceList[num+1][3]-priceList[num][3]<0) else 1

		#Adding to the total number of predictions made
		total = total+1

		#PNN Training
		if(alg == "Probabilistic Neural Network (Classification)"):
			pnn = algorithms.PNN(std=25)
			pnn.train(scaledTrainingData, catOutput)
			pred = pnn.predict(predTD)
			#Keeping track of how many PNN got correct
			if pred[0] == actual:
				correct = correct +1

		#MLP Training
		if(alg == "Multi-layer Perceptron (Classification)"):
			mlp = MLPClassifier()
			mlp.fit(scaledTrainingData, catOutput)
			pred = mlp.predict(predTD)
			#Keeping track of how many MLP got correct
			if pred[0] == actual:
				correct = correct +1

		#SVM Training
		if(alg == "Support Vector Machine (Classification)"):
			svmAlg = svm.SVC()
			svmAlg.fit(scaledTrainingData, catOutput)
			pred = svmAlg.predict(predTD)
			#Keeping track of how many SVM got correct
			if pred[0] == actual:
				correct=correct+1

	#Getting the final prediction for the user
	#Basically redoing what is done in the loop for one final prediction
	tdLen = len(scaledTrainingData)
	#The training data
	scaledTrainingData = []
	for day in range(firstDay,tdLen-1):
		scaledTrainingData.append(trainingData[day])
	if not trendDeter:
		scaledTrainingData = preprocessing.scale(scaledTrainingData)
	predTD = []
	predTD.append(trainingData[tdLen-1])

	#The categorical output to train
	catOutput = []
	for day in range(firstDay+1,tdLen):
		if openToClose:
			catOutput.append(0 if (priceList[day][3]-priceList[day][0]<0) else 1 )
		else:
			catOutput.append(0 if (priceList[day][3]-priceList[day-1][3]<0) else 1)

	predRate = correct/total
	fPred = []
	if(alg == "Probabilistic Neural Network (Classification)"):
		print("PNN:" + str(predRate))
		pnn = algorithms.PNN(std=25)
		pnn.train(scaledTrainingData, catOutput)
		fPred = pnn.predict(predTD)
	elif(alg == "Multi-layer Perceptron (Classification)"):
		print("MLP:" + str(predRate))
		mlp = MLPClassifier()
		mlp.fit(scaledTrainingData, catOutput)
		fPred = mlp.predict(predTD)
	elif(alg == "Support Vector Machine (Classification)"):
		print("SVM:" + str(predRate))
		svmAlg = svm.SVC()
		svmAlg.fit(scaledTrainingData, catOutput)
		fPred = svmAlg.predict(predTD)

	#Inverting the prediction rate and prediction if it is under 50%
	#Meaning since the algorithm is reliably guessing wrong, the opposite is more likely to be correct
	newPredRate = predRate if predRate>.5 else 1-predRate
	newfPred = fPred if predRate == newPredRate else 1-fPred

	openValue = []
	highValue = []
	lowValue = []
	closeValue = []

	# program freezes when use large number of daily
	# tuning carefully
	#This is the necessary information needed for the graph
	if tInterval == 'daily':
		dates = dates[-50:]
		for day in range(len(priceList)-50, len(priceList)):
			openValue.append(priceList[day][0])
			highValue.append(priceList[day][1])
			lowValue.append(priceList[day][2])
			closeValue.append(priceList[day][3])
	elif tInterval == 'weekly':
		dates = dates[-52:]
		for day in range(len(priceList)-52, len(priceList)):
			openValue.append(priceList[day][0])
			highValue.append(priceList[day][1])
			lowValue.append(priceList[day][2])
			closeValue.append(priceList[day][3])
	elif tInterval == 'monthly':
		dates = dates[-12:]
		for day in range(len(priceList)-12, len(priceList)):
			openValue.append(priceList[day][0])
			highValue.append(priceList[day][1])
			lowValue.append(priceList[day][2])
			closeValue.append(priceList[day][3])
	else:
		dates = dates[-60:]
		for day in range(len(priceList)-60, len(priceList)):
			openValue.append(priceList[day][0])
			highValue.append(priceList[day][1])
			lowValue.append(priceList[day][2])
			closeValue.append(priceList[day][3])

	return_list.put(newPredRate)
	return_list.put(fPred[0])
	return_list.put(dates)
	return_list.put(openValue)
	return_list.put(highValue)
	return_list.put(lowValue)
	return_list.put(closeValue)
	return

#Main menu window the user first sees
class MainMenu(tk.Frame):
	def __init__(self, parent, controller):
		self.controller = controller

		tk.Frame.__init__(self,parent, bg="gray10")

		#Title widget
		self.title = tk.Label(self,text='Prescient Trader',fg='green',bg='gray10',font='nimbusmonol  40')

		#Widget where the user enters the stock they want a prediction for
		self.searchBox = tk.Entry(self,width=30,font='nimbus 14', justify='center')
		self.searchBox.insert(0, "Specify A Stock Ticker")
		self.searchBox.bind("<Button-1>", lambda event: self.clear_entry(event))

		#The button that starts the algorithm
		self.predButton = tk.Button(self,text='Predict',fg='green',bg='gray8',font ='nimbussansl 8',highlightbackground='black', command = lambda: self.checkInp())
		self.predButton.configure(state = "normal", relief="raised", activeforeground= "white", activebackground="gray8")

		#The widget where the user chooses an algorithm
		self.alg = tk.StringVar(self)
		self.alg.set("Choose Algorithm")
		self.chooseAlg = tk.OptionMenu(self,self.alg,"Probabilistic Neural Network (Classification)","Multi-layer Perceptron (Classification)", "Support Vector Machine (Classification)")
		self.chooseAlg.configure(font = 'nimbussans1 8',bg='gray8',fg='green',width="40")
		self.chooseAlg["menu"].configure(bg='gray8',fg='green')

		#The widget where the user chooses a time interval
		self.time = tk.StringVar(self)
		self.time.set("Time Interval")
		self.chooseTime = tk.OptionMenu(self,self.time,"Monthly","Weekly","Daily","30min","15min","5min","1min")
		self.chooseTime.configure(font = 'nimbussans1 8',bg='gray8',fg='green',width="40")
		self.chooseTime["menu"].configure(bg='gray8',fg='green')

		#The widget where the user decides to do a tDF or not
		self.tDF = tk.StringVar(self)
		self.tDF.set("Trend Deterministic Transformation (Classification)")
		self.tDF.trace("w",self.handleTDF)
		self.chooseTDF = tk.OptionMenu(self,self.tDF,"Yes","No")
		self.chooseTDF.configure(font = 'nimbussans1 8',bg='gray8',fg='green',width="40")
		self.chooseTDF["menu"].configure(bg='gray8',fg='green')


		#The widget where the user can choose multiple technical indicators
		self.tI=tk.Listbox(self,selectmode=MULTIPLE)
		self.tI.insert(END, "MacD")
		for item in ["Stochastic Oscillator","EMA","RSI","Directional Indicator","CCI","William %R","OBV","Chaikin A/D Line","BBands", "Aroon", "ADX"]:
			self.tI.insert(END,item)
		self.tI.configure(font='nimbussans1 8',bg='gray8',fg='green',width="40", height='12')

		#The widget where the user chooses whether to include price in the training
		self.incP = tk.StringVar(self)
		self.incP.set("Include Price in Training (Classification)")
		self.chooseP = tk.OptionMenu(self, self.incP, "Yes", "No")
		self.chooseP.configure(font = 'nimbussans1 8',bg='gray8',fg='green',width="40")
		self.chooseP["menu"].configure(bg='gray8',fg='green')

		#The widget where the user chooses what type of class they want a rpediction for
		self.typeClass = tk.StringVar(self)
		self.typeClass.set("Type Of Class (Classification)")
		self.chooseClass = tk.OptionMenu(self,self.typeClass,"Open To Close","Close To Close")
		self.chooseClass.configure(font = 'nimbussans1 8',bg='gray8',fg='green',width="40")
		self.chooseClass["menu"].configure(bg='gray8',fg='green')

		#The progress bar
		self.style = ttk.Style()
		self.style.theme_use('clam')
		self.style.configure("green.Horizontal.TProgressbar", foreground='green', background='green', troughcolor='black')
		self.pbar = ttk.Progressbar(self, style="green.Horizontal.TProgressbar", mode='indeterminate')

		# Render Widgets Placement
		self.title.pack()
		self.searchBox.pack()
		self.chooseAlg.pack()
		self.chooseTime.pack()
		self.chooseTDF.pack()
		self.chooseClass.pack()
		self.chooseP.pack()
		self.tI.pack()
		self.predButton.pack()
		self.pbar.pack()

	#If the user chooses to do a tDF, the unnecessary options are disabled
	def handleTDF(self,*args):
		if self.tDF.get() == "Yes":
			self.chooseP.config(state=DISABLED)
			self.tI.config(state=DISABLED)
		else:
			self.chooseP.config(state=NORMAL)
			self.tI.config(state=NORMAL)

	#When the user clicks on the text entrybox, the "Specify a stock symbol" goes away
	def clear_entry(self, event):
		self.searchBox.delete(0, END)

	#Checking the form and ensuring the user chose whats needed from them
	def checkInp(self):
		#Comparing the values with the default values, if equal the user did not choose a necessary option
		if(self.alg.get() == "Choose Algorithm"):
			tkinter.messagebox.showerror("Error","Did not select an algorithm")
			return
		if(self.time.get() == "Time Interval"):
			tkinter.messagebox.showerror("Error","Did not select a time interval")
			return
		if(self.tDF.get() == "Trend Deterministic Transformation (Classification)" and not "Regression" in self.alg.get()):
			tkinter.messagebox.showerror("Error","Did not select whether to do a trend deterministic transformation or not")
			return
		if(self.incP.get() == "Include Price in Training (Classification)" and self.tDF.get()!="Yes"):
			tkinter.messagebox.showerror("Error","Did not select if you would like the price to be included in the training")
			return
		if(self.typeClass.get() == "Type Of Class (Classification)" and not "Regression" in self.alg.get()):
			tkinter.messagebox.showerror("Error","Did not choose the type of class")
			return
		if not self.tI.curselection() and self.tDF.get()!="Yes":
			tkinter.messagebox.showerror("Error","Did not choose any technical indicators to train with")
			return
		if tickers(self.searchBox.get()) is True:
			self.onStart()
		else:
			tkinter.messagebox.showerror("Error","Ticker Unavailable")
			return

	#Multiprocessing in order to have the progress bar move while
	# the algorithm works.
	def onStart(self):
		#Disabling the button so the user can't click it multiple times while the algorithm works
		self.predButton.config(state=DISABLED)
		#The list that is sent to the second process to return the outputs
		self.return_list = Queue()
		#Staring the second process
		self.p1 = Process(target=getData, args=(self.searchBox.get(),self.alg.get(),self.time.get(),self.tDF.get(),self.tI.curselection(),self.incP.get(),self.typeClass.get(),self.return_list),daemon=True)
		self.p1.start()
		self.pbar.start(20)
		#Checking for when the second process ends and an output has been returned
		self.after(80, self.onGetValue)
	def onGetValue(self):
		if(self.p1.is_alive()):
			self.after(80, self.onGetValue)
			return
		else:
			self.pbar.stop()
			self.p1.join()
			#We returned nothing when there was an error, this message is for the two possible errors that occured
			if self.return_list.empty():
				tkinter.messagebox.showerror("API Error"," API Failed to Return Information for The Prices or Technical Indicators.\n Please try again or try another ticker")
				self.predButton.config(state=NORMAL)
			else:
				#The output data from the algorithms is put into a list shared by all the GUI windows
				self.controller.shared_queue.put(self.return_list.get())
				self.controller.shared_queue.put(self.return_list.get())
				self.controller.shared_queue.put(self.return_list.get())
				self.controller.shared_queue.put(self.return_list.get())
				self.controller.shared_queue.put(self.return_list.get())
				self.controller.shared_queue.put(self.return_list.get())
				self.controller.shared_queue.put(self.return_list.get())
				#Removing the prerendered Results frame with a new one with the outputs
				self.controller.frames.pop(Results, None)
				frame = Results(self.controller.container, self.controller)
				self.controller.frames[Results] = frame
				frame.grid(row=0,column=0,sticky="nsew")
				self.predButton.config(state=NORMAL)
				self.controller.show_frame(Results)
#The results window
class Results(tk.Frame):
	def __init__(self, parent, controller):
		self.controller = controller
		tk.Frame.__init__(self,parent, bg="gray10")

		#The title of the current window
		self.title = tk.Label(self,text='Results',fg='green',bg='gray10',font='nimbusmonol  40')

		#The prediction accuracy is printed with up to 4 decimal places
		self.predAccuracy = tk.Label(self,text='Prediction Accuracy: '+"{:.4f}".format((self.controller.shared_queue.get())*100)+"%",fg='green',bg='gray10',font='nimbusmonol  20')

		#The prediction is translated to english and then displayed for the user
		pred = str(self.controller.shared_queue.get())
		if pred == "0" or pred == "1":
			pred = "The price will go down" if pred == 0 else "The price will go up"

		self.prediction = tk.Label(self,text='Prediction: '+ pred,fg='green',bg='gray10',font='nimbusmonol  20')

		#A button that lets the user go back to the MainMenu window and start a new prediction
		self.back = tk.Button(self,text='Go Back',fg='green',bg='gray8',font ='nimbussansl 8',highlightbackground='black', command = lambda: self.controller.show_frame(MainMenu))
		self.back.configure(state = "normal", relief="raised", activeforeground= "white", activebackground="gray8")

		dates = self.controller.shared_queue.get()
		openValue = self.controller.shared_queue.get()
		highValue = self.controller.shared_queue.get()
		lowValue = self.controller.shared_queue.get()
		closeValue = self.controller.shared_queue.get()

		# draw candlestick plot based on the values obtained
		f = Figure(figsize=(5,5), dpi=100, facecolor='#1a1a1a')
		a = f.add_subplot(111)
		# place ticks with a gap of 7
		a.set_xticks(range(0, len(dates), 7))
		a.set_xticklabels(dates[::7], rotation=15)
		a.set_ylabel('Price')
		a.spines['bottom'].set_color('green')
		a.spines['left'].set_color('green')
		a.xaxis.label.set_color('green')
		a.yaxis.label.set_color('green')
		a.tick_params(axis='x', colors='green')
		a.tick_params(axis='y', colors='green')
		candlestick2_ohlc(a, openValue, highValue, lowValue, closeValue, width=0.6, alpha=0.6)

		canvas = FigureCanvasTkAgg(f, self)
		canvas.draw()
		canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

		self.title.pack()
		self.predAccuracy.pack()
		self.prediction.pack()
		self.back.pack()

#The controller class intializes the windows
class Main(tk.Tk):
	def __init__(self, *args, **kwargs):
		tk.Tk.__init__(self,*args,**kwargs)
		self.shared_queue = Queue()
		self.shared_queue.put(0)
		self.shared_queue.put(0)
		array=[0]
		self.shared_queue.put(array)
		self.shared_queue.put(array)
		self.shared_queue.put(array)
		self.shared_queue.put(array)
		self.shared_queue.put(array)
		self.container = tk.Frame(self, bg="black")
		self.container.pack(side="top",fill="both",expand = True)
		self.container.grid_rowconfigure(0,weight=1)
		self.container.grid_columnconfigure(0,weight=1)
		self.frames = {}
		for F in (MainMenu, Results):
			frame = F(self.container, self)
			self.frames[F] = frame
			frame.grid(row=0,column=0, sticky="nsew")
		self.show_frame(MainMenu)
	def show_frame(self,cont):
		frame = self.frames[cont]
		frame.tkraise()
	def get_page(self, page_class):
		return self.frames[page_class]

#Starting the Program
App = Main()
App.mainloop()
