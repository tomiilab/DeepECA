import subprocess
import os,sys
#uniprot_path="path/to/uniprot20_2016_02/uniprot20_2016_02"
uniprot_path="/mnt/hdd2/Data/uniref/uniprot20_2016_02/uniprot20_2016_02"

def run_hhblits(file,in_dir,out_dir):
	base_name=file.split(".")[0]

	#hhblits
	in_file=os.path.join(in_dir,file)
	out_file=base_name+".a3m"
	cmd="hhblits -i " + in_file + " -d " + uniprot_path + " -oa3m "
	cmd+=out_file + " -n 3 -diff inf -cov 60"
	print(cmd)
	subprocess.call(cmd, shell=True)

	#egrep
	in_file=base_name+".a3m"
	out_file=os.path.join(out_dir,base_name+".aln")
	cmd="egrep -v \"^>\" "
	cmd+=in_file+" | sed \"s/[a-z]//g\" | sort -u > "
	cmd+=out_file
	print(cmd)
	subprocess.call(cmd, shell=True)

	os.remove(in_file)
	os.remove(os.path.join(in_dir,base_name+".hhr"))

def main():
	base_dir="./dataset/"
	in_dir=os.path.join(base_dir,"fasta")
	out_dir=os.path.join(base_dir,"aln")

	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

	#one or all
	if len(sys.argv)>=2:
		files=[sys.argv[1]+".fasta"]	
	else:
		files=os.listdir(in_dir)

	for file in files:
		base_name=file.split(".")[0]
		print(base_name)
		if os.path.exists(os.path.join(out_dir,base_name+".aln")):
			print("already exists...")
			continue
		run_hhblits(file,in_dir,out_dir)

if __name__ == '__main__':
	main()
