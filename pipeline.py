import subprocess
import os
import sys

fasta_dir="./dataset/fasta"

def pipeline(name):
	#run_hhblits
	cmd="python run_hhblits_local.py "+name
	print(cmd)
	subprocess.call(cmd, shell=True)

	#tfrecord
	cmd="python tfrecord.py "+name
	print(cmd)
	subprocess.call(cmd, shell=True)

	#pred contact
	cmd="python run_contact_pred.py "+name
	print(cmd)
	subprocess.call(cmd, shell=True)

	#pred ss
	cmd="python run_ss_pred.py "+name
	print(cmd)
	subprocess.call(cmd, shell=True)

	#pred ensemble
	cmd="python ensemble.py "+name
	print(cmd)
	subprocess.call(cmd, shell=True)

def main():
	#one or all
	if len(sys.argv)>=2:
		targets=[sys.argv[1]]	
	else:
		targets=[x.split(".")[0] for x in os.listdir(fasta_dir)]

	for target in targets:
		pipeline(target)

if __name__ == '__main__':
	main()