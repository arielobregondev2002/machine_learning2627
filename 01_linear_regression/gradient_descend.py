import pandas as pd
import numpy as np

db = pd.DataFrame({
	"x": [1, 2, 3, 4],
	"y": [3, 5, 7, 9]
})

x = db['x']
y = db['y']

learn_rate = 0.1
def main():
	print(db)
	gradients()

def gradients():

	mw= float(input("m: "))
	bw = float(input("b: "))
	
	for iteration in range(5):
		yw_list = []
		for data in db['x']:
			yw = mw*data + bw
			yw_list.append(yw)
		yw_set = np.array(yw_list)
		b_grad = (2/db['y'].count()) * sum(yw_set - y)
		m_grad =  (2/db['y'].count()) * sum(x*(yw_set - y))
		print("b gradient:", b_grad)
		print("m gradient:", m_grad)
		new_m, new_b = new_parameters(mw, bw,m_grad, b_grad)
		print("New coef:", new_m)
		print("New interception:", new_b)
		y_new = new_m * x + new_b
		print(y_new)
		mw = new_m
		bw = new_b
	
def new_parameters(mw, bw, m_grad, b_grad):
	new_m = mw - learn_rate * m_grad
	new_b = bw - learn_rate * b_grad
	return new_m, new_b
	
	
main()
	
