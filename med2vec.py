#################################################################
# Code written by Edward Choi (mp2893@gatech.edu)
# For bug report, please contact author using the email address
#################################################################

import sys, random
import numpy as np
import cPickle as pickle
from collections import OrderedDict
import argparse

import theano
import theano.tensor as T
from theano import config

def numpy_floatX(data):
	return np.asarray(data, dtype=config.floatX)

def unzip(zipped):
	new_params = OrderedDict()
	for k, v in zipped.iteritems():
		new_params[k] = v.get_value()
	return new_params

def init_params(options):
	params = OrderedDict()

	numXcodes = options['numXcodes']
	numYcodes = options['numYcodes']
	embDimSize= options['embDimSize']
	demoSize = options['demoSize']
	hiddenDimSize = options['hiddenDimSize']

	params['W_emb'] = np.random.uniform(-0.01, 0.01, (numXcodes, embDimSize)).astype(config.floatX) #emb matrix needs an extra dimension for the time
	params['b_emb'] = np.zeros(embDimSize).astype(config.floatX)
	params['W_hidden'] = np.random.uniform(-0.01, 0.01, (embDimSize+demoSize, hiddenDimSize)).astype(config.floatX) #emb matrix needs an extra dimension for the time
	params['b_hidden'] = np.zeros(hiddenDimSize).astype(config.floatX)
	if numYcodes > 0:
		params['W_output'] = np.random.uniform(-0.01, 0.01, (hiddenDimSize, numYcodes)).astype(config.floatX) #emb matrix needs an extra dimension for the time
		params['b_output'] = np.zeros(numYcodes).astype(config.floatX)
	else:
		params['W_output'] = np.random.uniform(-0.01, 0.01, (hiddenDimSize, numXcodes)).astype(config.floatX) #emb matrix needs an extra dimension for the time
		params['b_output'] = np.zeros(numXcodes).astype(config.floatX)

	return params

def load_params(options):
	params = np.load(options['modelFile'])
	return params

def init_tparams(params):
	tparams = OrderedDict()
	for k, v in params.iteritems():
		tparams[k] = theano.shared(v, name=k)
	return tparams

def build_model(tparams, options):
	x = T.matrix('x', dtype=config.floatX)
	d = T.matrix('d', dtype=config.floatX)
	y = T.matrix('y', dtype=config.floatX)
	mask = T.vector('mask', dtype=config.floatX)

	logEps = options['logEps']

	emb = T.maximum(T.dot(x, tparams['W_emb']) + tparams['b_emb'],0)
	if options['demoSize'] > 0: emb = T.concatenate((emb, d), axis=1)
	visit = T.maximum(T.dot(emb, tparams['W_hidden']) + tparams['b_hidden'],0)
	results = T.nnet.softmax(T.dot(visit, tparams['W_output']) + tparams['b_output'])
	
	mask1 = (mask[:-1] * mask[1:])[:,None]
	mask2 = (mask[:-2] * mask[1:-1] * mask[2:])[:,None]
	mask3 = (mask[:-3] * mask[1:-2] * mask[2:-1] * mask[3:])[:,None]
	mask4 = (mask[:-4] * mask[1:-3] * mask[2:-2] * mask[3:-1] * mask[4:])[:,None]
	mask5 = (mask[:-5] * mask[1:-4] * mask[2:-3] * mask[3:-2] * mask[4:-1] * mask[5:])[:,None]

	t = None
	if options['numYcodes'] > 0: t = y
	else: t = x

	forward_results =  results[:-1] * mask1
	forward_cross_entropy = -(t[1:] * T.log(forward_results + logEps) + (1. - t[1:]) * T.log(1. - forward_results + logEps))

	forward_results2 =  results[:-2] * mask2
	forward_cross_entropy2 = -(t[2:] * T.log(forward_results2 + logEps) + (1. - t[2:]) * T.log(1. - forward_results2 + logEps))

	forward_results3 =  results[:-3] * mask3
	forward_cross_entropy3 = -(t[3:] * T.log(forward_results3 + logEps) + (1. - t[3:]) * T.log(1. - forward_results3 + logEps))

	forward_results4 =  results[:-4] * mask4
	forward_cross_entropy4 = -(t[4:] * T.log(forward_results4 + logEps) + (1. - t[4:]) * T.log(1. - forward_results4 + logEps))

	forward_results5 =  results[:-5] * mask5
	forward_cross_entropy5 = -(t[5:] * T.log(forward_results5 + logEps) + (1. - t[5:]) * T.log(1. - forward_results5 + logEps))

	backward_results =  results[1:] * mask1
	backward_cross_entropy = -(t[:-1] * T.log(backward_results + logEps) + (1. - t[:-1]) * T.log(1. - backward_results + logEps))

	backward_results2 =  results[2:] * mask2
	backward_cross_entropy2 = -(t[:-2] * T.log(backward_results2 + logEps) + (1. - t[:-2]) * T.log(1. - backward_results2 + logEps))

	backward_results3 =  results[3:] * mask3
	backward_cross_entropy3 = -(t[:-3] * T.log(backward_results3 + logEps) + (1. - t[:-3]) * T.log(1. - backward_results3 + logEps))

	backward_results4 =  results[4:] * mask4
	backward_cross_entropy4 = -(t[:-4] * T.log(backward_results4 + logEps) + (1. - t[:-4]) * T.log(1. - backward_results4 + logEps))

	backward_results5 =  results[5:] * mask5
	backward_cross_entropy5 = -(t[:-5] * T.log(backward_results5 + logEps) + (1. - t[:-5]) * T.log(1. - backward_results5 + logEps))

	visit_cost1 = (forward_cross_entropy.sum(axis=1).sum(axis=0) + backward_cross_entropy.sum(axis=1).sum(axis=0)) / (mask1.sum() + logEps)
	visit_cost2 = (forward_cross_entropy2.sum(axis=1).sum(axis=0) + backward_cross_entropy2.sum(axis=1).sum(axis=0)) / (mask2.sum() + logEps)
	visit_cost3 = (forward_cross_entropy3.sum(axis=1).sum(axis=0) + backward_cross_entropy3.sum(axis=1).sum(axis=0)) / (mask3.sum() + logEps)
	visit_cost4 = (forward_cross_entropy4.sum(axis=1).sum(axis=0) + backward_cross_entropy4.sum(axis=1).sum(axis=0)) / (mask4.sum() + logEps)
	visit_cost5 = (forward_cross_entropy5.sum(axis=1).sum(axis=0) + backward_cross_entropy5.sum(axis=1).sum(axis=0)) / (mask5.sum() + logEps)

	windowSize = options['windowSize']
	visit_cost = visit_cost1
	if windowSize == 2:
		visit_cost = visit_cost1 + visit_cost2
	elif windowSize == 3:
		visit_cost = visit_cost1 + visit_cost2 + visit_cost3
	elif windowSize == 4:
		visit_cost = visit_cost1 + visit_cost2 + visit_cost3 + visit_cost4
	elif windowSize == 5:
		visit_cost = visit_cost1 + visit_cost2 + visit_cost3 + visit_cost4 + visit_cost5

	iVector = T.vector('iVector', dtype='int32')
	jVector = T.vector('jVector', dtype='int32')
	preVec = T.maximum(tparams['W_emb'],0)
	norms = (T.exp(T.dot(preVec, preVec.T))).sum(axis=1)
	emb_cost = -T.log((T.exp((preVec[iVector] * preVec[jVector]).sum(axis=1)) / norms[iVector]) + logEps)

	total_cost = visit_cost + T.mean(emb_cost) + options['L2_reg'] * (tparams['W_emb'] ** 2).sum()

	if options['demoSize'] > 0 and options['numYcodes'] > 0: return x, d, y, mask, iVector, jVector, total_cost
	elif options['demoSize'] == 0 and options['numYcodes'] > 0: return x, y, mask, iVector, jVector, total_cost
	elif options['demoSize'] > 0 and options['numYcodes'] == 0: return x, d, mask, iVector, jVector, total_cost
	else: return x, mask, iVector, jVector, total_cost

def adadelta(tparams, grads, x, mask, iVector, jVector, cost, options, d=None, y=None):
	zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_grad' % k) for k, p in tparams.iteritems()]
	running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rup2' % k) for k, p in tparams.iteritems()]
	running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rgrad2' % k) for k, p in tparams.iteritems()]

	zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
	rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

	if options['demoSize'] > 0 and options['numYcodes'] > 0:
		f_grad_shared = theano.function([x, d, y, mask, iVector, jVector], cost, updates=zgup + rg2up, name='adadelta_f_grad_shared')
	elif options['demoSize'] == 0 and options['numYcodes'] > 0:
		f_grad_shared = theano.function([x, y, mask, iVector, jVector], cost, updates=zgup + rg2up, name='adadelta_f_grad_shared')
	elif options['demoSize'] > 0 and options['numYcodes'] == 0:
		f_grad_shared = theano.function([x, d, mask, iVector, jVector], cost, updates=zgup + rg2up, name='adadelta_f_grad_shared')
	else:
		f_grad_shared = theano.function([x, mask, iVector, jVector], cost, updates=zgup + rg2up, name='adadelta_f_grad_shared')

	updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
	ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
	param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

	f_update = theano.function([], [], updates=ru2up + param_up, on_unused_input='ignore', name='adadelta_f_update')

	return f_grad_shared, f_update

def load_data(xFile, dFile, yFile):
	seqX = np.array(pickle.load(open(xFile, 'rb')))
	seqD = []
	if len(dFile) > 0: seqD = np.asarray(pickle.load(open(dFile, 'rb')), dtype=config.floatX)
	seqY = []
	if len(yFile) > 0: seqY = np.array(pickle.load(open(yFile, 'rb')))
	return seqX, seqD, seqY

def pickTwo(codes, iVector, jVector):
	for first in codes:
		for second in codes:
			if first == second: continue
			iVector.append(first)
			jVector.append(second)
	
def padMatrix(seqs, labels, options):
	n_samples = len(seqs)
	iVector = []
	jVector = []
	numXcodes = options['numXcodes']
	numYcodes = options['numYcodes']

	if numYcodes > 0:
		x = np.zeros((n_samples, numXcodes)).astype(config.floatX)
		y = np.zeros((n_samples, numYcodes)).astype(config.floatX)
		mask = np.zeros((n_samples,)).astype(config.floatX)
		for idx, (seq, label) in enumerate(zip(seqs, labels)):
			if not seq[0] == -1:
				x[idx][seq] = 1.
				y[idx][label] = 1.
				pickTwo(seq, iVector, jVector)
				mask[idx] = 1.
		return x, y, mask, iVector, jVector
	else:
		x = np.zeros((n_samples, numXcodes)).astype(config.floatX)
		mask = np.zeros((n_samples,)).astype(config.floatX)
		for idx, seq in enumerate(seqs):
			if not seq[0] == -1:
				x[idx][seq] = 1.
				pickTwo(seq, iVector, jVector)
				mask[idx] = 1.
		return x, mask, iVector, jVector

def train_med2vec(seqFile='seqFile.txt', 
				demoFile='demoFile.txt',
				labelFile='labelFile.txt',
				outFile='outFile.txt',
				modelFile='modelFile.txt',
				L2_reg=0.001,
				numXcodes=20000, 
				numYcodes=20000, 
				embDimSize=1000,
				hiddenDimSize=2000,
				batchSize=100,
				demoSize=2,
				logEps=1e-8,
				windowSize=1,
				verbose=False,
				maxEpochs=1000):

	options = locals().copy()
	print 'initializing parameters'
	params = init_params(options)
	#params = load_params(options)
	tparams = init_tparams(params)

	print 'building models'
	f_grad_shared = None
	f_update = None
	if demoSize > 0 and numYcodes > 0:
		x, d, y, mask, iVector, jVector, cost = build_model(tparams, options)
		grads = T.grad(cost, wrt=tparams.values())
		f_grad_shared, f_update = adadelta(tparams, grads, x, mask, iVector, jVector, cost, options, d=d, y=y)
	elif demoSize == 0 and numYcodes > 0:
		x, y, mask, iVector, jVector, cost = build_model(tparams, options)
		grads = T.grad(cost, wrt=tparams.values())
		f_grad_shared, f_update = adadelta(tparams, grads, x, mask, iVector, jVector, cost, options, y=y)
	elif demoSize > 0 and numYcodes == 0:
		x, d, mask, iVector, jVector, cost = build_model(tparams, options)
		grads = T.grad(cost, wrt=tparams.values())
		f_grad_shared, f_update = adadelta(tparams, grads, x, mask, iVector, jVector, cost, options, d=d)
	else:
		x, mask, iVector, jVector, cost = build_model(tparams, options)
		grads = T.grad(cost, wrt=tparams.values())
		f_grad_shared, f_update = adadelta(tparams, grads, x, mask, iVector, jVector, cost, options)

	print 'loading data'
	seqs, demos, labels = load_data(seqFile, demoFile, labelFile)
	n_batches = int(np.ceil(float(len(seqs)) / float(batchSize)))

	print 'training start'
	for epoch in xrange(maxEpochs):
		iteration = 0
		costVector = []
		for index in random.sample(range(n_batches), n_batches):
			batchX = seqs[batchSize*index:batchSize*(index+1)]
			batchY = []
			batchD = []
			if demoSize > 0 and numYcodes > 0:
				batchY = labels[batchSize*index:batchSize*(index+1)]
				x, y, mask, iVector, jVector = padMatrix(batchX, batchY, options)
				batchD = demos[batchSize*index:batchSize*(index+1)]
				cost = f_grad_shared(x, batchD, y, mask, iVector, jVector)
			elif demoSize == 0 and numYcodes > 0:
				batchY = labels[batchSize*index:batchSize*(index+1)]
				x, y, mask, iVector, jVector = padMatrix(batchX, batchY, options)
				cost = f_grad_shared(x, y, mask, iVector, jVector)
			elif demoSize > 0 and numYcodes == 0:
				x, mask, iVector, jVector = padMatrix(batchX, batchY, options)
				batchD = demos[batchSize*index:batchSize*(index+1)]
				cost = f_grad_shared(x, batchD, mask, iVector, jVector)
			else:
				x, mask, iVector, jVector = padMatrix(batchX, batchY, options)
				cost = f_grad_shared(x, mask, iVector, jVector)
			costVector.append(cost)
			f_update()
			if (iteration % 10 == 0) and verbose: print 'epoch:%d, iteration:%d/%d, cost:%f' % (epoch, iteration, n_batches, cost)
			iteration += 1
		print 'epoch:%d, mean_cost:%f' % (epoch, np.mean(costVector))
		tempParams = unzip(tparams)
		np.savez_compressed(outFile + '.' + str(epoch), **tempParams)

def parse_arguments(parser):
	parser.add_argument('seq_file', type=str, metavar='<visit_file>', help='The path to the Pickled file containing visit information of patients')
	parser.add_argument('n_input_codes', type=int, metavar='<n_input_codes>', help='The number of unique input medical codes')
	parser.add_argument('out_file', type=str, metavar='<out_file>', help='The path to the output models. The models will be saved after every epoch')
	parser.add_argument('--label_file', type=str, default='', help='The path to the Pickled file containing grouped visit information of patients. If you are not using a grouped output, do not use this option')
	parser.add_argument('--n_output_codes', type=int, default=0, help='The number of unique output medical codes (the number of unique grouped codes). If you are not using a grouped output, do not use this option')
	parser.add_argument('--demo_file', type=str, default='', help='The path to the Pickled file containing demographic information of patients. If you are not using patient demographic information, do not use this option')
	parser.add_argument('--demo_size', type=int, default=0, help='The size of the demographic information vector. If you are not using patient demographic information, do not use this option')
	parser.add_argument('--cr_size', type=int, default=200, help='The size of the code representation (default value: 200)')
	parser.add_argument('--vr_size', type=int, default=200, help='The size of the visit representation (default value: 200)')
	parser.add_argument('--batch_size', type=int, default=1000, help='The size of a single mini-batch (default value: 1000)')
	parser.add_argument('--n_epoch', type=int, default=10, help='The number of training epochs (default value: 10)')
	parser.add_argument('--L2_reg', type=float, default=0.001, help='L2 regularization for the code representation matrix W_c (default value: 0.001)')
	parser.add_argument('--window_size', type=int, default=1, choices=[1,2,3,4,5], help='The size of the visit context window (range: 1,2,3,4,5), (default value: 1)')
	parser.add_argument('--log_eps', type=float, default=1e-8, help='A small value to prevent log(0) (default value: 1e-8)')
	parser.add_argument('--verbose', action='store_true', help='Print output after every 10 mini-batches')
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	args = parse_arguments(parser)

	train_med2vec(seqFile=args.seq_file, demoFile=args.demo_file, labelFile=args.label_file, outFile=args.out_file, numXcodes=args.n_input_codes, numYcodes=args.n_output_codes, embDimSize=args.cr_size, hiddenDimSize=args.vr_size, batchSize=args.batch_size, maxEpochs=args.n_epoch, L2_reg=args.L2_reg, demoSize=args.demo_size, windowSize=args.window_size, logEps=args.log_eps, verbose=args.verbose)
