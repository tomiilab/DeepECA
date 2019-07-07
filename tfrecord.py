import os,sys
import numpy as np
import pandas as pd
import tensorflow as tf

#dir
base_dir="./dataset"
aln_dir=os.path.join(base_dir,"aln")
fasta_dir=os.path.join(base_dir,"fasta")
contact_dir=os.path.join(base_dir,"contact")
dssp_dir=os.path.join(base_dir,"dssp")
out_dir="./tfrecord"
if not os.path.exists(out_dir):
	os.makedirs(out_dir)

aa=["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y","-"]
aa2num={aa[i]:str(i) for i in range(len(aa))}
def read_fasta(fasta_file):
	#fasta_file="/mnt/hdd1/Data/PSICOV/seq/1a3aA.fasta" for test
	with open(fasta_file) as f:
		fasta=f.readlines()
	fasta=fasta[1].replace("U","-").replace("B","-").replace("Z","-").replace("X","-").replace("O","-").replace("\n","")
	fasta=np.array([ aa2num[ fasta[i] ] for i in range(len(fasta))]).astype(int)
	return fasta

def read_aln(aln_file):
	#aln_file="/mnt/hdd1/Data/PSICOV/aln/1a3aA.aln" for test
	#read aln
	with open(aln_file) as f:
		aln=f.readlines()
	#remove \n
	aln=[aln[i].replace("J","-").replace("U","-").replace("B","-").replace("Z","-").replace("X","-").replace("O","-").replace("\n","").replace("*","-") for i in range(len(aln))]
	#amino acid->int
	m=len(aln) #count of alignment 
	n=len(aln[0]) #length of sequence 
	aln=np.array([[ aa2num[ aln[j][i] ] for i in range(n)] for j in range(m)]).astype(int)
	return aln

def read_contact(data_file):
	data=pd.read_csv(data_file,header=None).values
	return data

def read_dssp(data_file):
	dssp=pd.read_csv(data_file)
	ss2num={"H":0,"B":1,"E":2,"G":3,"I":4,"T":5,"S":6,"-":7}
	ss=np.array([ss2num[x] for x in dssp.ss])
	asa_num=dssp.asa.fillna(-1).values
	return ss,asa_num

def read_data(name):
	#aln
	aln=read_aln(os.path.join(aln_dir,name+".aln"))
	m=aln.shape[0]
	n=aln.shape[1]
	#query
	query=read_fasta(os.path.join(fasta_dir,name+".fasta"))
	#contact #dssp
	if os.path.exists(os.path.join(contact_dir,name+".contact")):
		y=read_contact(os.path.join(contact_dir,name+".contact"))[:n,:n] 
		ss_dssp,asa_num=read_dssp(os.path.join(dssp_dir,name+".csv"))
	else:
		y=np.zeros([n,n]) #dummy
		ss_dssp=np.zeros([n]) #dummy
		asa_num=np.zeros([n]) #dummy
	#mask
	mask=np.array([[1 if y[i,j]!=9  else 0 for j in range(aln.shape[1])] for i in range(aln.shape[1])])
	#gap
	gap=np.array([(aln[i,:]==20).sum()/n for i in range(m)])
	#identity
	identity=[(x==query).sum()/n for x in aln]
	#identity_cons 
	cons=[np.argmax([(aln[:,i]==j).sum() for j in range(20)]) for i in range(n)] 
	identity_cons=[(x==cons).sum()/n for x in aln]
	#tostring
	aln=aln.astype("uint8").tostring()
	query=query.astype("uint8").tostring()
	y=y.astype("uint8").tostring()
	mask=mask.astype("uint8").tostring()
	ss_dssp=ss_dssp.astype("uint8").tostring()
	asa_num=asa_num.astype("int64")
	#import pdb;pdb.set_trace()
	return m,n,aln,query,y,mask,gap,identity,identity_cons,ss_dssp,asa_num

def write_data(name,out_dir):
	#build dataset
	with tf.python_io.TFRecordWriter(os.path.join(out_dir,name)) as writer:
		#read data
		m,n,align,query,y,mask,gap,identity,identity_cons,ss_dssp,asa_num=read_data(name)
		print(m,n)
		#weite TFRecord
		record=tf.train.Example(features=tf.train.Features(feature={
			'name':tf.train.Feature(bytes_list=tf.train.BytesList(value=[name.encode()])),
			'm':tf.train.Feature(int64_list=tf.train.Int64List(value=[m])),
			'n':tf.train.Feature(int64_list=tf.train.Int64List(value=[n])),
			'query':tf.train.Feature(bytes_list=tf.train.BytesList(value=[query])),
			'align':tf.train.Feature(bytes_list=tf.train.BytesList(value=[align])),
			'y':tf.train.Feature(bytes_list=tf.train.BytesList(value=[y])),
			'mask':tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask])),
			'gap':tf.train.Feature(float_list=tf.train.FloatList(value=gap)),
			'identity':tf.train.Feature(float_list=tf.train.FloatList(value=identity)),
			'identity_cons':tf.train.Feature(float_list=tf.train.FloatList(value=identity_cons)),
			'ss_dssp':tf.train.Feature(bytes_list=tf.train.BytesList(value=[ss_dssp])),
			'asa_num':tf.train.Feature(int64_list=tf.train.Int64List(value=asa_num))			
		}))
		writer.write(record.SerializeToString())

if __name__ == '__main__':

	#mkdir
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

	#one or all
	if len(sys.argv)>=2:
		files=[sys.argv[1]+".fasta"]	
	else:
		files=os.listdir(fasta_dir)

	#main
	for k,file in enumerate(files):
		file=file.split(".")[0]
		print(k,file)
		#if os.path.exists(os.path.join(out_dir,file)):
		#	continue
		write_data(file,out_dir)

