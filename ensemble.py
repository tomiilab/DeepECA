import pandas as pd
import numpy as np
import os
import sys

#simple average

base_dir="./"
#tfrecord_dir=os.path.join(base_dir,"tfrecord")
pred_dir=os.path.join(base_dir,"multitask/pred") #+data_num
out_dir=os.path.join(base_dir,"contact_pred")
out_dir_ss=os.path.join(base_dir,"ss_pred")
if not os.path.exists(out_dir):
	os.makedirs(out_dir)
if not os.path.exists(out_dir_ss):
	os.makedirs(out_dir_ss)
def main():
	#one or all
	if len(sys.argv)>=2:
		files=[sys.argv[1]]
	else:
		files=[x.split(".")[0] for x in os.listdir(pred_dir) if x.split(".")[1]=="csv"]

	for file in files:
		#check already exists
		if os.path.exists(os.path.join(out_dir,file+".csv")):
			continue
		#read prediction
		for i in range(5):
		#for i in range(1):
			#contact
			tmp=pd.read_csv(os.path.join(pred_dir+str(i),file+".csv"))
			if i==0:
				df=tmp
			else:
				df["pred"]+=tmp["pred"]

			#ss
			tmp=pd.read_csv(os.path.join(pred_dir+str(i),file+".ss.csv"))
			if i==0:
				_pred_ss=tmp.values
			else:
				_pred_ss+=tmp.values
		df["pred"]=df["pred"]/5
		#df["pred"]=df["pred"]/1
		
		#save csv
		df.to_csv(os.path.join(out_dir,file+".csv"),index=False)

		#save ss
		with open(os.path.join(out_dir_ss,file+".ss"),"w") as g:
			g.write(">"+file+"\n")
			num2ss=["H","E","C"]
			ss_final="".join([num2ss[x] for x in np.argmax(_pred_ss,axis=1)])
			print(ss_final)
			g.write(ss_final+"\n")

if __name__ == '__main__':
	main()