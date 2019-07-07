import pandas as pd
import os,sys

base_dir="./dataset"
fasta_dir=os.path.join(base_dir,"fasta")
pred_dir="./contact_pred/"
rr2l_dir="./rr2l"
rrall_dir="./rrall"

if not os.path.exists(rr2l_dir):
	os.makedirs(rr2l_dir)
if not os.path.exists(rrall_dir):
	os.makedirs(rrall_dir)

#for confold
def rr2l(name):
	if 	os.path.exists(os.path.join(rr2l_dir,name)+".rr"):
		print("alredy exists.")
		return
	#read csv
	df=pd.read_csv(os.path.join(pred_dir,name+".csv"))
	df=df.sort_values(by="pred", ascending=False)
	#read fasta
	with open(os.path.join(fasta_dir,name+".fasta")) as g:
		fasta=g.readlines()
		fasta=fasta[1].replace("U","-").replace("B","-").replace("Z","-").replace("X","-").replace("O","-").replace("\n","")
	#import pdb;pdb.set_trace()
	with open(os.path.join(rr2l_dir,name)+".rr","w") as g:
		g.write(fasta+"\n")
		cnt=0
		for k,v in df.iterrows():
			i=int(v["i"])
			j=int(v["j"])
			if (j-i)>=6:
				g.write(str(i)+" "+str(j)+" 0 8 "+str(v["pred"])+"\n")
				cnt+=1
			if cnt>=2*len(fasta):
				break
#for submission
def rrall(name):
	if 	os.path.exists(os.path.join(rrall_dir,name)+".rr"):
		print("alredy exists.")
		return
	#read csv
	df=pd.read_csv(os.path.join(pred_dir,name+".csv"))
	df=df.sort_values(by="pred", ascending=False)
	#read fasta
	with open(os.path.join(fasta_dir,name+".fasta")) as g:
		fasta=g.readlines()
		fasta=fasta[1].replace("U","-").replace("B","-").replace("Z","-").replace("X","-").replace("O","-").replace("\n","")
	#import pdb;pdb.set_trace()
	with open(os.path.join(rrall_dir,name)+".rr","w") as g:
		#write comment
		g.write("PFRMAT RR\n")
		g.write("TARGET "+name+"\n")
		g.write("AUTHOR 1046-2246-1673\n")
		#f.write("SCORE\n")
		g.write("REMARK It's Tomii Lab's submission.\n")
		g.write("METHOD An original contact prediction method using DNN.\n")
		g.write("METHOD It takes MSA as input and predicts contact directly from MSA\n")
		g.write("METHOD which can weight the sequences in MSA at the same time by the same network.\n")
		g.write("METHOD Predicted secondary structure and others are also used as input features.\n")
		g.write("MODEL  1\n")
		#write fasata
		for i in range(len(fasta)//50+1):
			g.write(fasta[i*50:(i+1)*50]+"\n")
		#write prediction
		for k,v in df[:30000].iterrows():
			i=int(v["i"])
			j=int(v["j"])
			if i<j:
				g.write(str(i)+" "+str(j)+" 0 8 "+str(v["pred"])+"\n")
		#write termination
		g.write("END\n")	

def main():
	#one or all
	if len(sys.argv)>=2:
		files=[sys.argv[1]+".fasta"]	
	else:
		files=os.listdir(fasta_dir)

	for file in files:
		name=file.split(".")[0]
		print(name)
		rr2l(name)
		rrall(name)


if __name__ == '__main__':
	main()