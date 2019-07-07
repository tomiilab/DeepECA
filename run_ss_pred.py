import os,subprocess,sys

model="./multitask.py"

def main():
	#one or all
	if len(sys.argv)>=2:
		name=sys.argv[1]	

		#pred contact
		cmd="python "+model+" --test --data_num 0 --epoch_num 22 --save_ss --target "+name
		print(cmd)
		subprocess.call(cmd, shell=True)

		cmd="python "+model+" --test --data_num 1 --epoch_num 20 --save_ss --target "+name
		print(cmd)
		subprocess.call(cmd, shell=True)

		cmd="python "+model+" --test --data_num 2 --epoch_num 13 --save_ss --target "+name
		print(cmd)
		subprocess.call(cmd, shell=True)

		cmd="python "+model+" --test --data_num 3 --epoch_num 10 --save_ss --target "+name
		print(cmd)
		subprocess.call(cmd, shell=True)

		cmd="python "+model+" --test --data_num 4 --epoch_num 14 --save_ss --target "+name
		print(cmd)
		subprocess.call(cmd, shell=True)

	else:
		#pred contact
		cmd="python "+model+" --test --data_num 0 --epoch_num 22 --save_ss"
		print(cmd)
		subprocess.call(cmd, shell=True)

		cmd="python "+model+" --test --data_num 1 --epoch_num 20 --save_ss"
		print(cmd)
		subprocess.call(cmd, shell=True)

		cmd="python "+model+" --test --data_num 2 --epoch_num 13 --save_ss"
		print(cmd)
		subprocess.call(cmd, shell=True)

		cmd="python "+model+" --test --data_num 3 --epoch_num 10 --save_ss"
		print(cmd)
		subprocess.call(cmd, shell=True)

		cmd="python "+model+" --test --data_num 4 --epoch_num 14 --save_ss"
		print(cmd)
		subprocess.call(cmd, shell=True)

if __name__ == '__main__':
	main()
