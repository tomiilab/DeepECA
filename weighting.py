#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import pandas as pd
import os,sys
import tensorflow as tf

np.random.seed(0)
tf.set_random_seed(0)

aa=["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y","-"]
aa2num={aa[i]:str(i) for i in range(len(aa))}

#setting
n_maxepoch=30
n_classes=2
n_hidden=32
n_hidden_ss=100
n_alignment=30000
n_clip=200
learning_rate=0.0005
beta=1e-3

#flags
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('data_num', 0, """test data set""")
tf.app.flags.DEFINE_integer('epoch_num', 0, """epoch number""")
tf.app.flags.DEFINE_boolean('validation', False, """validation mode""")
tf.app.flags.DEFINE_boolean('test', False, """test mode""")
tf.app.flags.DEFINE_boolean('predict', False, """predict mode""")
tf.app.flags.DEFINE_integer('gpu_num', 1, """gpu number""")
tf.app.flags.DEFINE_string('data_dir', "./training", """data directory""")
tf.app.flags.DEFINE_boolean('save', False, """save result""")
tf.app.flags.DEFINE_boolean('save_ss', False, """save ss result""")
tf.app.flags.DEFINE_string('target', None, """target name""")
tf.app.flags.DEFINE_string('test_dir', "./tfrecord", """target name""")

#save_dir
save_dir=os.path.join("./",__file__.split("/")[-1].split(".")[0]+"/ckpt"+str(FLAGS.data_num))
if not os.path.exists(save_dir):
	os.makedirs(save_dir)

#pred_dir
pred_dir=os.path.join("./",__file__.split("/")[-1].split(".")[0]+"/pred"+str(FLAGS.data_num))
if not os.path.exists(pred_dir):
	os.makedirs(pred_dir)

#target_record
#set gpu_num=1 for validation/test
if not FLAGS.test: #training/validatoin
	#datalist
	datalist_file=[x for x in FLAGS.data_dir]

	#split
	train_files=[datalist[i] for i in range(len(datalist_file)) if i%5!=FLAGS.data_num]
	validation_files=[datalist[i] for i in range(len(datalist_file)) if i%5==FLAGS.data_num]
	train_record=[FLAGS.data_dir+x for x in train_files]
	validation_record=[FLAGS.data_dir+x for x in validation_files]
	
	if not FLAGS.validation:
		target_record=train_record
	else:
		target_record=validation_record
		FLAGS.gpu_num=1
else: #test
	if FLAGS.target!=None:
		test_files=pd.Series([FLAGS.target])
	else:
		test_files=pd.Series(os.listdir(FLAGS.test_dir))
	error_list=[] #os.listdir(pred_dir)
	test_record=[os.path.join(FLAGS.test_dir,x) for x in test_files if x+".csv" not in error_list]
	target_record=test_record
	FLAGS.gpu_num=1
	if test_record==[]:
		sys.exit()

def batch_normalization(input,is_train=True,name=None):
  shape=input.get_shape().as_list()[-1]
  eps = 1e-5
  if name!=None:
    gamma = weight_get_variable(name+"_gamma",[shape])
    beta = weight_get_variable(name+"_beta",[shape])
  else:
    gamma = weight_variable([shape])
    beta = weight_variable([shape])
  pop_mean = tf.Variable(tf.zeros([shape]), trainable=False)
  pop_var = tf.Variable(tf.ones([shape]), trainable=False)
  mean, variance = tf.nn.moments(input, [0,1,2])
  decay = 0.999
  global_mean = tf.assign(pop_mean,pop_mean*decay+mean*(1-decay))
  global_var = tf.assign(pop_var,pop_var*decay+variance*(1-decay))
  if is_train is not None:
  	return gamma * (input - mean) / tf.sqrt(variance + eps) + beta
  else:
  	return gamma * (input - global_mean) / tf.sqrt(global_var + eps) + beta

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def weight_get_variable(name,shape):
	initial = tf.truncated_normal_initializer(stddev=0.1)
	return tf.get_variable(name,shape=shape,initializer=initial)

def bias_get_variable(name,shape):
	initial = tf.constant_initializer(0.1)
	return tf.get_variable(name,shape=shape,initializer=initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv(input,filter,activation=False):
	#input [batch,width,hight,channel]
	#filter [filter_width, filter_height, in_channel, out_channel]
	w_conv=tf.Variable(tf.truncated_normal(filter, stddev=0.1))
	b_conv=tf.Variable(tf.constant(0.1,shape=[filter[3]]))
	if activation==True:
		return tf.nn.relu(conv2d(input, w_conv)+b_conv)
	else:
		return conv2d(input, w_conv)+b_conv

def conv_bn(input,filter,activation=False,is_train=True,name=None):
	#w_conv=tf.Variable(tf.truncated_normal(filter, stddev=0.1))
	#He initialization
	import math
	n=filter[0]*filter[1]*filter[2]
	if name!=None:
		w_conv=tf.get_variable(name,shape=filter,initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/n)))
	else:
		with tf.variable_scope("Conv2d"):
			w_conv=tf.Variable(tf.truncated_normal(filter, stddev=math.sqrt(2.0/n)))
	bn=batch_normalization(tf.nn.conv2d(input,w_conv,strides=[1,1,1,1],padding='SAME'),is_train=is_train,name=name)
	if activation==True:
		return tf.nn.relu(bn)
	else:
		return bn

def residual_block(inpt, output_depth,is_train=True,name=None):
	input_depth = inpt.get_shape().as_list()[3]
	if name!=None:
	    conv1 = conv_bn(inpt,[3,3,input_depth,output_depth],activation=True,is_train=is_train,name=name+"_1")
	    conv2 = conv_bn(conv1,[3,3,output_depth,output_depth],activation=True,is_train=is_train,name=name+"_2")
	else:
		conv1 = conv_bn(inpt,[3,3,input_depth,output_depth],activation=True,is_train=is_train,name=None)
		conv2 = conv_bn(conv1,[3,3,output_depth,output_depth],activation=True,is_train=is_train,name=None)
	res = conv2 + inpt
	return res

def resnet(inpt,n_block,n_filter,is_train=True,name=None):
	conv=inpt
	for i in range(n_block):
		if name!=None:
			conv=residual_block(conv,n_filter,is_train=is_train,name=name+"_"+str(i))		
		else:
			conv=residual_block(conv,n_filter,is_train=is_train,name=None)
	return conv

def read_data(filename_queue,is_train):
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(
		serialized_example,
		features={
			'name':tf.FixedLenFeature([],tf.string),
			'm':tf.FixedLenFeature([],tf.int64),
			'n':tf.FixedLenFeature([],tf.int64),
			'query':tf.FixedLenFeature([],tf.string),
			'align':tf.FixedLenFeature([],tf.string),
			'y':tf.FixedLenFeature([],tf.string),
			'mask':tf.FixedLenFeature([],tf.string),
			'gap':tf.VarLenFeature(tf.float32),
			'identity':tf.VarLenFeature(tf.float32),
			'identity_cons':tf.VarLenFeature(tf.float32),
			'ss_dssp':tf.FixedLenFeature([],tf.string),
			'asa_num':tf.VarLenFeature(tf.int64),
		}
	)
	name=features["name"]
	m=tf.cast(features["m"],tf.int32)
	n=tf.cast(features["n"],tf.int32)
	align=tf.reshape(tf.decode_raw(features["align"],tf.uint8),tf.stack([m,n]))
	query=tf.decode_raw(features["query"],tf.uint8)
	y=tf.reshape(tf.decode_raw(features["y"],tf.uint8),tf.stack([n,n]))
	mask=tf.reshape(tf.decode_raw(features["mask"],tf.uint8),tf.stack([n,n]))
	gap=features["gap"].values
	identity=features["identity"].values
	identity_cons=features["identity_cons"].values
	ss_dssp=tf.decode_raw(features["ss_dssp"],tf.uint8)
	asa_num=tf.cast(features["asa_num"].values,tf.int32)
	gap=features["gap"].values
	identity=features["identity"].values
	identity_cons=features["identity_cons"].values
	
	#clip
	def clipping(align,query,ss_dssp,asa_num,y,mask):	
		begin=tf.random_uniform([],maxval=tf.shape(align)[1]-n_clip,dtype=tf.int32)
		align=align[:,begin:begin+n_clip]
		query=query[begin:begin+n_clip]
		ss_dssp=ss_dssp[begin:begin+n_clip]
		asa_num=asa_num[begin:begin+n_clip]
		y=y[begin:begin+n_clip,begin:begin+n_clip]
		mask=mask[begin:begin+n_clip,begin:begin+n_clip]
		return align,query,ss_dssp,asa_num,y,mask
	align,query,ss_dssp,asa_num,y,mask=tf.cond((n>n_clip)&(is_train),lambda:clipping(align,query,ss_dssp,asa_num,y,mask)
		,lambda:(align,query,ss_dssp,asa_num,y,mask))

	#sampling	
	def sampling(align,gap,identity,identity_cons):
		idx=tf.random_uniform([n_alignment],maxval=m,dtype=tf.int32)
		align=tf.gather_nd(align,tf.expand_dims(idx,1))
		gap=tf.gather_nd(gap,tf.expand_dims(idx,1))
		identity=tf.gather_nd(identity,tf.expand_dims(idx,1))
		identity_cons=tf.gather_nd(identity_cons,tf.expand_dims(idx,1))
		return align,gap,identity,identity_cons
	align,gap,identity,identity_cons=tf.cond((m>n_alignment)&(is_train),lambda:sampling(align,gap,identity,identity_cons),lambda:(align,gap,identity,identity_cons))
	
	return name,align,query,y,mask,gap,identity,identity_cons,ss_dssp,asa_num

def input(filelist,is_train):
	filename_queue = tf.train.string_input_producer(filelist,num_epochs=None) #,shuffle=False)
	name,align,query,y,mask,gap,identity,identity_cons,ss_dssp,asa_num=read_data(filename_queue,is_train)
	return name,align,query,y,mask,gap,identity,identity_cons,ss_dssp,asa_num

with tf.device('/gpu:0'):
	is_train=tf.placeholder(tf.bool)
	keep_prob=tf.placeholder(tf.float32)

	name,aln,query,y,mask,gap,identity,cons_identity,ss_dssp,asa_dssp,asa_num=input(target_record,is_train)

	#m,n
	m=tf.shape(aln)[0]
	n=tf.shape(aln)[1]

	#pos
	pos_i=tf.ones((n,n),tf.int32)*tf.cast(tf.range(n),tf.int32) #n,n
	pos_j=tf.transpose(pos_i,[1,0])
	#dist
	dist=pos_j-pos_i

	#features
	m_=tf.ones([m,1],dtype=tf.float32)*tf.cast(m,tf.float32)/100000
	identity_mean=tf.ones([m,1],dtype=tf.float32)*tf.reduce_mean(identity)
	gap_mean=tf.ones([m,1],dtype=tf.float32)*tf.reduce_mean(gap)
	cons_mean=tf.ones([m,1],dtype=tf.float32)*tf.reduce_mean(cons_identity)
	h_aln=tf.concat([tf.expand_dims(identity,1),tf.expand_dims(gap,1),
		tf.expand_dims(cons_identity,1),m_,identity_mean,gap_mean,cons_mean],axis=1)
	#calc weight
	n_aln=7
	#aln1
	w_aln1=weight_get_variable("w_aln1",[7,n_aln])
	b_aln1=bias_get_variable("b_aln1",[n_aln])
	h_aln=tf.nn.relu(tf.matmul(h_aln,w_aln1)+b_aln1)
	#aln2
	w_aln2=weight_get_variable("w_aln2",[n_aln,n_aln])
	b_aln2=bias_get_variable("b_aln2",[n_aln])
	h_aln=tf.nn.relu(tf.matmul(h_aln,w_aln2)+b_aln2)
	#dropout
	h_aln=tf.nn.dropout(h_aln,keep_prob)
	#aln3
	w_aln3=weight_get_variable("w_aln3",[n_aln,1])
	b_aln3=bias_get_variable("b_aln3",[1])
	p=tf.cast(tf.nn.sigmoid(tf.matmul(h_aln,w_aln3)+b_aln3),tf.float32)
	p=p*tf.cast(m,tf.float32)/tf.reduce_sum(p) #psum=m

	#one hot
	x=tf.one_hot(aln,depth=21,dtype=tf.float32)*tf.expand_dims(p,2) #(m,n,21)
	print("x",x)
	#fa
	fa=tf.reduce_sum(x,0)+1 #(n,21)
	fa/=tf.cast(m+1,tf.float32)
	fa2=tf.reshape(fa,[n*21,1])
	fafb=tf.matmul(fa2,tf.transpose(fa2)) #(n*21,1)(1,n*21)->(n*21,n*21)
	fafb=tf.reshape(tf.transpose(tf.reshape(fafb,[n,21,n,21]),[0,2,1,3]),[n,n,441])
	print("fafb",fafb)	
	#fab
	fab=x #(m,n,21)
	fab=tf.reshape(fab,[m,n*21])
	fab=tf.matmul(tf.transpose(fab),fab)
	fab+=1
	fab/=tf.cast(m+1,tf.float32)
	fab=tf.reshape(tf.transpose(tf.reshape(fab,[n,21,n,21]),[0,2,1,3]),[n,n,441])
	print("fab",fab)	
	#cmat
	cmat=fab-fafb
	mean,variance=tf.nn.moments(cmat,[0,1])
	cmat=(cmat-mean)/tf.sqrt(variance+1e-8)
	cmat=tf.expand_dims(cmat,0)
	print("cmat",cmat)
	cmat=conv_bn(cmat,filter=[1,1,441,128],activation=True,is_train=is_train,name="Conv2d_1")

	#cmat->h
	h=cmat
	#resnet
	#h=conv_bn(h,filter=[1,1,441,128],activation=True,is_train=is_train,name="Conv2d_2")
	h=resnet(h,30,128,is_train=is_train,name="Conv2d_res")

	#contact
	#h_con=resnet(h,10,128,is_train=is_train,name="Conv2d_res_con")
	h_con=tf.reshape(h,[n*n,128])
	w_fc=weight_get_variable("w_fc",[128,1])
	b_fc=bias_get_variable("b_fc",[1])
	pred=tf.nn.sigmoid(tf.matmul(h_con,w_fc)+b_fc)
	pred=tf.concat([1-pred,pred],axis=1) #(n*n,2)
	#true 
	true=tf.to_float(tf.one_hot(tf.reshape(y,[n*n]), depth=n_classes)) #(n*n,2)
	mask_=tf.to_float(tf.expand_dims(tf.reshape(mask,[n*n]),1))
	#loss
	loss=-tf.reduce_sum(true*tf.log(tf.clip_by_value(pred,1e-10,1.0))*mask_)
	pred1=pred

	#loss
	optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

	#result short L/10,L/5,L/2,L middle L/10,L/5,L/2,L long L/10,L/5,L/2,L
	def calc_score(pred,dist,mask_):
		dist=tf.reshape(dist,[n*n])
		mask_=tf.reshape(mask_,[n*n])
		scores=[]
		count=[]
		total=[]
		for i,distance in enumerate([0,1,2]): #distance
			if i==0: #short
				idx=tf.where((dist>=6)&(dist<=11)&(mask_>=1)) #距離条件に合ったindexを取得 #n*n,1
			elif i==1: #middle
				idx=tf.where((dist>=12)&(dist<=23)&(mask_>=1))
			else:
				idx=tf.where((dist>=24)&(mask_>=1))
			sel_pred=tf.gather_nd(pred[:,1],idx) #距離条件に合う予測のみ選択
			sel_true=tf.gather_nd(true[:,1],idx)
			top_pred,top_idx=tf.nn.top_k(sel_pred,tf.minimum(n,tf.shape(sel_pred)[0])) #ソートしてtop Lを選択
			top_true=tf.gather_nd(sel_true,tf.expand_dims(top_idx,1)) #top Lを選択
			for j,topk in enumerate([10,5,2,1]): #top k #予測がn個なくてもnで割る
				scores.append(tf.reduce_sum(tf.to_float(top_true[:n//topk]))/tf.to_float(n//topk))
				count.append(tf.reduce_sum(tf.to_float(top_true[:n//topk])))
				total.append(n//topk)
		return scores,count,total
	scores1,count1,total1=calc_score(pred1,dist,mask_)	
	#scores2,count2,total2=calc_score(pred2,dist,mask_)

	init=tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
	saver = tf.train.Saver(max_to_keep=None)

def main():
	#session
	#myconfig=tf.ConfigProto(intra_op_parallelism_threads=4,allow_soft_placement=True) #,log_device_placement=True)
	myconfig=tf.ConfigProto(allow_soft_placement=True)
	with tf.Session(config=myconfig) as sess:

		#init
		sess.run(init)
		#restore
		if FLAGS.epoch_num != 0:
			saver.restore(sess, os.path.join(save_dir,"model-"+str(FLAGS.epoch_num)))
		#log
		from datetime import datetime
		timeid=datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
		logid=os.path.join("./",__file__.split("/")[-1])+str(FLAGS.data_num)
		if FLAGS.validation:
			f=open(logid+"_"+timeid+"validation.txt","w")
		elif FLAGS.test:
			f=open(logid+"_"+timeid+"test.txt","w")
		else:
			f=open(logid+"_"+timeid+".txt","w")

		# Start input enqueue threads.
		coord=tf.train.Coordinator()
		threads=tf.train.start_queue_runners(sess=sess,coord=coord)

		try:
			if FLAGS.validation:
				#is_train=tf.constant(False)
				n_epoch=FLAGS.epoch_num
				n_total=validation_files.shape[0]
				cnt=1
				total_loss=0
				total_scores=np.zeros([3,4])
				total_total=np.zeros([3,4])
				total_count=np.zeros([3,4])
				#total_scores2=np.zeros([3,4])
				#total_total2=np.zeros([3,4])
				#total_count2=np.zeros([3,4])
				total_accuracy_ss=0
				total_accuracy_asa=0				
				while not coord.should_stop():
					_m,_n,_loss,_name,_scores1,_count1,_total1=sess.run([m,n,loss,name,scores1,count1,total1],feed_dict={is_train:False,keep_prob:1.0})
					#_m,_n,_loss,_name,_scores1,_count1,_total1,_scores2,_count2,_total2,_loss_ss,_accuracy_ss,_loss_asa=sess.run([m,n,loss2,name,scores1,count1,total1,scores2,count2,total2,loss_ss,accuracy_ss,loss_asa],feed_dict={is_train:False,keep_prob:1.0})
					print(n_epoch,cnt,_name.decode(),_m,_n,_loss)
					#print(_loss_ss)
					#print(_accuracy_ss)
					#print(_loss_asa)
					total_loss+=_loss
					#total_accuracy_ss+=_accuracy_ss
					#total_accuracy_asa+=_loss_asa					
					#calc score
					for i,distance in enumerate([0,1,2]): #distance
						for j,topk in enumerate([10,5,2,1]): #top k
							total_scores[i,j]+=_scores1[i*4+j]
							total_total[i,j]+=_total1[i*4+j]
							total_count[i,j]+=_count1[i*4+j]
							print(n_epoch,distance,topk,_scores1[i*4+j])
					#for i,distance in enumerate([0,1,2]): #distance
					#	for j,topk in enumerate([10,5,2,1]): #top k
					#		total_scores2[i,j]+=_scores2[i*4+j]
					#		total_total2[i,j]+=_total2[i*4+j]
					#		total_count2[i,j]+=_count2[i*4+j]
					#		print(n_epoch,distance,topk,_scores2[i*4+j])
					print("")
					#epoch			
					if cnt%n_total==0:
						f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+" "+str(n_epoch)+" "+str(total_loss/n_total)+"\n")
						#display score
						#平均の正解率
						for i,distance in enumerate([0,1,2]): #distance
							for j,topk in enumerate([10,5,2,1]): #top k
								print(n_epoch,distance,topk,total_scores[i,j]/n_total)
								print(n_epoch,distance,topk,total_scores[i,j]/n_total,file=f)
						#for i,distance in enumerate([0,1,2]): #distance
						#	for j,topk in enumerate([10,5,2,1]): #top k
						#		print(n_epoch,distance,topk,total_scores2[i,j]/n_total)
						#		print(n_epoch,distance,topk,total_scores2[i,j]/n_total,file=f)
						#print(total_accuracy_ss/cnt)
						#print(total_accuracy_ss/cnt,file=f)
						#print(total_accuracy_asa/cnt)
						#print(total_accuracy_asa/cnt,file=f)
						print("")
						print("",file=f)
						f.flush()					
						#to next epoch
						n_epoch+=1
						if os.path.exists(os.path.join(save_dir,"model-"+str(n_epoch)+".index")): #check saved file exists
							saver.restore(sess, os.path.join(save_dir,"model-"+str(n_epoch)))
						else:
							f.close()
							break
						cnt=0
						total_loss=0
						total_scores=np.zeros([3,4])
						total_total=np.zeros([3,4])
						total_count=np.zeros([3,4])
						#total_scores2=np.zeros([3,4])
						#total_total2=np.zeros([3,4])
						#total_count2=np.zeros([3,4])
						total_accuracy_ss=0
						total_accuracy_asa=0
					cnt+=1
			elif FLAGS.test:
				n_total=test_files.shape[0]
				n_epoch=FLAGS.epoch_num 
				total_loss=0
				cnt=1
				total_scores=np.zeros([3,4])
				total_correct=np.zeros([3,4])
				total_total=np.zeros([3,4])
				total_accuracy_ss=0
				total_correct_ss=0
				total_count_ss=0
				#total_accuracy_asa=0					
				while not coord.should_stop():
					_name,_y,_mask,_pred,_dist,_m,_n,_loss=sess.run([name,y,mask,pred,dist,m,n,loss],feed_dict={is_train:False,keep_prob:1.0})
					#_name,_y,_mask,_pred,_dist,_m,_n,_loss,_query,_gap,_identity=sess.run([name,y,mask,pred,dist,m,n,loss,query,gap,identity],feed_dict={is_train:False,keep_prob:1.0})
					print(n_epoch,cnt,_name.decode(),_m,_n,_loss)
					#print(_accuracy_ss)
					#print(_loss_asa)
					total_loss+=_loss
					#total_accuracy_ss+=_accuracy_ss
					#total_correct_ss+=_correct_ss.sum()
					#total_count_ss+=_correct_ss.shape[0]
					#total_accuracy_asa+=_loss_asa							
					#result
					tmp=pd.DataFrame([_y.reshape(-1),_pred[:,1],_dist.reshape(-1),_mask.reshape(-1)]).T
					tmp.columns=["true","pred","dist","mask"]
					tmp=tmp[tmp["mask"]==1]
					tmp["dist"][tmp.dist<=5]=-1
					tmp["dist"][(tmp.dist>=6)&(tmp.dist<=11)]=0
					tmp["dist"][(tmp.dist>=12)&(tmp.dist<=23)]=1
					tmp["dist"][tmp.dist>=24]=2
					for i,distance in enumerate([0,1,2]): #distance
						result=tmp[tmp.dist==distance].sort_values(by="pred", ascending=False)
						for j,topk in enumerate([10,5,2,1]): #top k
							total_total[i,j]+=int(_y.shape[1]/topk) #n/top k
							total_correct[i,j]+=result.iloc[:int(_y.shape[1]/topk),:]["true"].sum()
							total_scores[i,j]+=(result.iloc[:int(_y.shape[1]/topk),:]["true"].sum())/int(_y.shape[1]/topk)
							print(n_epoch,distance,topk,(result.iloc[:int(_y.shape[1]/topk),:]["true"].sum())/int(_y.shape[1]/topk))
					#save prediction
					if FLAGS.save:
						#copy
						tmp=_pred[:,1]
						#if not exstis save
						with open(os.path.join(pred_dir,_name.decode()+".csv"),"w") as g:
							g.write("i,j,pred\n")
							#for idx in idxs:
							for idx in range(tmp.shape[0]):
								i=int(idx/_n)+1 #0->(1,1) 1->(1,2)
								j=int(idx%_n)+1
								if (i<j):
									g.write(str(i)+","+str(j)+","+str(tmp[idx])+"\n")
						#save ss
						#with open(os.path.join(fasta_dir,_name.decode()+".fasta")) as g:
						#	fasta=g.readlines()
						#	fasta=fasta[1].replace("U","-").replace("B","-").replace("Z","-").replace("X","-").replace("O","-").replace("\n","")
						#with open(os.path.join(pred_dir,_name.decode()+".ss"),"w") as g:
						#	g.write(">"+_name.decode()+"\n")
							#g.write(fasta+"\n")
							#num2ss=["H","B","E","G","I","T","S","-"]
							#ss_final="".join([num2ss[x] for x in np.argmax(_pred_ss,axis=1)]).replace("T","C").replace("S","C").replace("-","C").replace("I","C").replace("G","H").replace("B","E")
						#	num2ss=["H","E","C"]
						#	ss_final="".join([num2ss[x] for x in np.argmax(_pred_ss,axis=1)])
						#	print(ss_final)
						#	g.write(ss_final+"\n")
						#ss csv
						#pd.DataFrame(_pred_ss).to_csv(os.path.join(pred_dir,_name.decode()+".ss.csv"),index=False)
					#epoch finished
					if cnt%n_total==0:
						#display score
						for i,distance in enumerate([0,1,2]): #distance
							for j,topk in enumerate([10,5,2,1]): #top k
								print(n_epoch,distance,topk,total_scores[i,j]/n_total)
								print(n_epoch,distance,topk,total_scores[i,j]/n_total,file=f)
						#print(total_accuracy_ss/cnt)
						#print(total_accuracy_ss/cnt,file=f)
						#print(total_correct_ss/float(total_count_ss))
						#print(total_correct_ss/float(total_count_ss),file=f)
						#print(total_accuracy_asa/cnt)
						#print(total_accuracy_asa/cnt,file=f)
						print("")
						print("",file=f)
						f.flush()		
						#import pdb;pdb.set_trace()			
						#to next epoch
						n_epoch+=1
						if os.path.exists(os.path.join(save_dir,"model-"+str(n_epoch)+".index")): #check saved file exists
							f.close()
							break
							#saver.restore(sess, os.path.join(save_dir,"model-"+str(n_epoch)))
						else:
							f.close()
							break
						cnt=0
						total_loss=0
						total_correct=np.zeros([3,4])
						total_total=np.zeros([3,4])
						total_scores=np.zeros([3,4])
						total_accuracy_ss=0
						total_correct_ss=0
						total_count_ss=0
						#total_accuracy_asa=0
					cnt+=1
			else:
				n_epoch=FLAGS.epoch_num+1
				n_total=train_files.shape[0]
				total_loss=0
				cnt=1
				while not coord.should_stop():
					#_,_name,_m,_n,_loss,_loss_ss,_accuracy_ss,_loss_asa=sess.run([optimizer,name,m,n,loss,loss_ss,accuracy_ss,loss_asa],feed_dict={is_train:True,keep_prob:0.5})
					_,_name,_m,_n,_loss=sess.run([optimizer,name,m,n,loss],feed_dict={is_train:True,keep_prob:0.5})
					print(n_epoch,cnt,_name.decode(),_m,_n,_loss)
					total_loss+=_loss
					#print(_accuracy_ss)
					#_,_name,_m,_n,_loss_ss,_accuracy_ss,_true,_pred=sess.run([optimizer_ss,name,m,n,loss_ss,accuracy_ss,asa_num,pred_asa],feed_dict={is_train:True,keep_prob:0.5})
					#print(n_epoch,cnt,_name.decode(),_m,_n,_loss)
					#print(_true[:10,0].astype(int),_pred[:5,0].astype(int))
					#_,_name,_m,_n,_loss_asa,_true,_pred=sess.run([optimizer_asa,name,m,n,loss_asa,asa_num,pred_asa],feed_dict={is_train:True,keep_prob:0.5})
					#print(n_epoch,cnt,_name.decode(),_m,_n,_loss)
					#print(_true[:10,0].astype(int),_pred[:5,0].astype(int))
					#epoch
					if cnt>=n_total:
						saver.save(sess, os.path.join(save_dir,'model'),global_step=n_epoch)
						f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+" "+str(n_epoch)+" "+str(total_loss/n_total)+"\n")
						f.flush()
						n_epoch+=1
						cnt=0
						total_loss=0
					cnt+=1*FLAGS.gpu_num
		except tf.errors.OutOfRangeError:
			import pdb;pdb.set_trace()
			prinrt("Done training -- epoch limit reached")
		finally:
			coord.request_stop()

		coord.join(threads)

		#log close
		f.close()

if __name__ == '__main__':
	main()

