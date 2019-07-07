import os,subprocess,sys

model="./multitask.py"

def main():
	#one or all
	if len(sys.argv)>=2:
		name=sys.argv[1]	

		#pred contact
		cmd="python "+model+" --test --data_num 0 --epoch_num 15 --save --target "+name
		print(cmd)
		subprocess.call(cmd, shell=True)

		cmd="python "+model+" --test --data_num 1 --epoch_num 19 --save --target "+name
		print(cmd)
		subprocess.call(cmd, shell=True)

		cmd="python "+model+" --test --data_num 2 --epoch_num 18 --save --target "+name
		print(cmd)
		subprocess.call(cmd, shell=True)

		cmd="python "+model+" --test --data_num 3 --epoch_num 24 --save --target "+name
		print(cmd)
		subprocess.call(cmd, shell=True)

		cmd="python "+model+" --test --data_num 4 --epoch_num 22 --save --target "+name
		print(cmd)
		subprocess.call(cmd, shell=True)

	else:
		#pred contact
		cmd="python "+model+" --test --data_num 0 --epoch_num 15 --save"
		print(cmd)
		subprocess.call(cmd, shell=True)

		cmd="python "+model+" --test --data_num 1 --epoch_num 19 --save"
		print(cmd)
		subprocess.call(cmd, shell=True)

		cmd="python "+model+" --test --data_num 2 --epoch_num 18 --save"
		print(cmd)
		subprocess.call(cmd, shell=True)

		cmd="python "+model+" --test --data_num 3 --epoch_num 24 --save"
		print(cmd)
		subprocess.call(cmd, shell=True)

		cmd="python "+model+" --test --data_num 4 --epoch_num 22 --save"
		print(cmd)
		subprocess.call(cmd, shell=True)

if __name__ == '__main__':
	main()
