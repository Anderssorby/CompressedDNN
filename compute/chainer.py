# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .base import ComputationInterface
# from chainer import computational_graph
from chainer import cuda
# from chainer import optimizers
# from chainer import serializers
from chainer import Variable
import numpy as np
from scipy import linalg as LA
import shelve
import matplotlib.pyplot as plt
from utils.transform import Transform

xp = cuda.cupy


class Chainer(ComputationInterface):
    def calc_eigs(self, model_wrapper, exp_type, epoch_stage=15):

        # print('load Model exp(%d),epoch(%d)' % (exp_type, epoch_stage))

        # if exp_type == 1:
        #     model_tmp = ResNet2.ResNet()
        #     model = res_analysis2.ResNet_ana()
        #     serializers.load_hdf5('../results/ResNet_2017-06-20_17-42-35_149794815589/epoch-20.model',model_tmp)
        #     serializers.load_hdf5('../results/ResNet_2017-06-20_17-42-35_149794815589/epoch-20.model',model)
        #     model_name = 'ResNet'
        # elif exp_type == 2:
        #     model = res_analysis2.ResNet_ana(wbase=32)
        #     serializers.load_hdf5('../results/ResNet2_2017-07-08_20-24-33_149951307357/epoch-10.model',model)
        #     model_name = 'ResNet2'
        # elif exp_type == 3:
        #     model = vgg2.VGG_mod()
        #     #serializers.load_hdf5('../results/VGG2_2017-07-14_20-46-59_150003281952/epoch-15.model',model)
        #     serializers.load_hdf5('../results/VGG2_2017-07-14_20-46-59_150003281952/epoch-%d.model' % epoch_stage,model)
        #     model_name = 'VGG'
        #     width_list = np.array([64, 64, 128, 128, 256, 256, 256, 256, 1024, 1024])
        #
        # elif exp_type >= 4 and exp_type <=14:
        #     #After thresholding, and re-training
        #     ThreshList = (0.005,0.01,0.025,0.05,0.1,0.2,0.3,0.5,0.7)
        #     fold_names = ('VGG3_2017-09-01_12-28-12_150423649213', #4
        #                     'VGG3_2017-09-01_14-49-32_15042449727', #5
        #                     'VGG3_2017-09-01_17-09-29_150425336996', #6
        #                     'VGG3_2017-10-20_16-08-21_150848330121', #7
        #                     'VGG3_2017-10-22_09-35-55_150863255517',#''VGG3_2017-10-20_18-08-30_150849051095', #8
        #                     'VGG3_2017-10-20_20-00-02_150849720232', #9
        #                     'VGG3_2017-10-20_23-17-18_15085090390', #10
        #                     'VGG3_2017-10-21_00-45-03_15085143038', #11
        #                     'VGG3_2017-10-21_02-10-31_150851943128', #12 # This would be nice
        #                     'VGG3_2017-09-01_19-23-12_150426139203', #13
        #                     'VGG3_2017-09-01_21-21-24_150426848419') #14
        #     for line in open('./results/%s/log.txt' % fold_names[exp_type-4]):
        #         if 'width list:[' in line:
        #                 width_list = np.array(line[line.find(':[')+2:-2].split()).astype(int)
        #                 break
        #     print(width_list)
        #     model = vgg3.VGG(width_list=width_list)
        #     serializers.load_hdf5('./results/%s/epoch-%d.model' % (fold_names[exp_type-4],epoch_stage),model)
        #     model_name = 'VGG3(%g)' % ThreshList[exp_type-4]
        # elif exp_type >= 15 and exp_type <=19:
        #     #After thresholding, and re-training
        #     ThreshList = (0.005,0.01,0.025,0.05,0.1,0.2,0.3,0.5,0.7)
        #     fold_names = ('VGG3_2017-10-22_09-35-55_150863255517','VGG3_randomlabel_2018-02-03_19-03-01_151765218176') #15,16
        #     for line in open('./results/%s/log.txt' % fold_names[exp_type-15]):
        #         if 'width list:[' in line:
        #                 width_list = np.array(line[line.find(':[')+2:-2].split()).astype(int)
        #                 break
        #     print(width_list)
        #     model = vgg3.VGG(width_list=width_list)
        #     serializers.load_hdf5('./results/%s/epoch-%d.model' % (fold_names[exp_type-15],epoch_stage),model)
        #     model_name = 'VGG3(exp:%d)' % exp_type
        model = model_wrapper.model
        model_name = model_wrapper.model_name

        b_usingGPU = False
        if self.args.gpu >= 0:
            print("Getting GPU")
            cuda.get_device(self.args.gpu).use()
            model.to_gpu()
            b_usingGPU = True

        # my_res_cifar.get_model_optimizer(args)
        # dataset = load_dataset(self.args.datadir)

        x_train, y_train, x_test, y_test = model_wrapper.dataset

        N_train = len(x_train)
        N_test = len(x_test)

        trans = Transform(self.args)
        cov = []
        mean_vec = []
        full_mean = []
        y_pred = []
        y_test = []
        cov_class = []
        mean_vec_class = []
        Test_Num = 0  # N_test/batchsize #np.min([N_test,5000])

        b_UseTrainingData = False  # True
        if b_UseTrainingData:
            NN_test = np.min((N_train, 200000))
            batchsize = 1000
            perm = np.random.permutation(N_train)
        else:
            NN_test = np.min((N_test, 10000))
            batchsize = 20
            perm = np.random.permutation(N_test)

        for i in range(0, NN_test, batchsize):  # Test_Num):
            Test_Num = Test_Num + 1
            if np.mod(i, 100) == 0:
                print('%d' % i)
            ind_tmp = perm[i:i + batchsize]

            if b_UseTrainingData:
                x = np.asarray(x_train[ind_tmp], dtype=np.float32)
                x = x.transpose((0, 3, 1, 2))
                t = np.asarray(y_train[ind_tmp], dtype=np.int32)
                x = Variable(xp.asarray(x), volatile='on')
                t = Variable(xp.asarray(t), volatile='on')
            else:
                for (jjind, jj) in enumerate(ind_tmp):
                    if jjind == 0:
                        aug = trans(x_test[jj])
                    else:
                        aug = np.concatenate((aug, trans(x_test[jj])), axis=0)

                x = xp.asarray(aug, dtype=xp.float32)
                # print(x.shape)
                x = x.transpose((0, 3, 1, 2))
                # t = np.asarray(np.repeat(te_labels[perm[i:i + batchsize]], len(aug)), dtype=np.int32)
                # x = chainer.Variable(np.asarray(te_data[perm[i:i + batchsize]]),volatile = 'on')
                # t = chainer.Variable(np.asarray(te_labels[ind_tmp],dtype=np.int32),volatile = 'on')
                t = xp.asarray(xp.repeat(y_test[ind_tmp], len(aug)), dtype=xp.int32)
                # x.data = x.data.reshape((len(x.data), 3, 32, 32))
                # x.data = x.data.reshape((len(x.data), -1))
                # volatile = 'on' #if train else 'on'
                # x = Variable(np.asarray(x), volatile='on')
                # t = Variable(np.asarray(t), volatile='on')
                # (hout,yout) = model(x,t,b_Analysis=True)
            model.train = False
            hout = model(x, t, b_Anal=True)
            # TODO
            # hout = model_wrapper.get_nth_layer_output(n=1)
            for (jj, (name, hh)) in enumerate(hout):
                hsize = hh.data.shape  # (sample num, channel num, width, height)
                tmp_mean_all = xp.mean(hh.data, axis=0)
                tmp_mean_all = tmp_mean_all.reshape(tmp_mean_all.shape[0], -1)
                tmp_mean = xp.array(tmp_mean_all)

                hhdataorig = xp.copy(hh.data)
                b_colapse = True
                if b_colapse:
                    hh.data = hh.data.swapaxes(0, 1).reshape(hsize[1], -1)
                else:
                    hh.data = hh.data.reshape((hsize[0], -1)).swapaxes(0, 1)  # (sample num, channel-num*width*height)^T
                # calculating centered covariance
                if i == 0:
                    # Mcross = xp.outer(xp.mean(hh.data,axis=1),xp.mean(hh.data,axis=1))
                    print('first loop')
                    print(hh.data.shape)
                    cov.append(xp.dot(hh.data, hh.data.T) / (np.prod(hsize) / hsize[1]))  # xp.cov(hh.data))
                    mean_vec.append(xp.mean(hh.data, axis=1))
                    full_mean.append(tmp_mean)

                    tmp_list = []
                    tmp_list2 = []
                    num_class = len(np.unique(y_test))
                    print('numclass: %d' % num_class)
                    class_samp_num = np.zeros(num_class)
                    print(t)
                    for ii in range(num_class):
                        num_classii = len(xp.where(xp.array(t) == ii)[0])
                        class_samp_num[ii] = class_samp_num[ii] + num_classii
                        print('class sample size:%d' % num_classii)
                        print(xp.array(t) == ii)
                        if num_classii == 0:
                            tmp_list.append(xp.zeros((hsize[1], hsize[1])))
                            tmp_list2.append(xp.zeros(tmp_mean_all.shape))
                        else:
                            hhdata_tmp = hhdataorig[xp.where(xp.array(t) == ii)[0], :]
                            tmp_mean = xp.mean(hhdata_tmp, axis=0)
                            hsizetmp = hhdata_tmp.shape
                            hhdata_tmp = hhdata_tmp.swapaxes(0, 1).reshape(hsizetmp[1], -1)
                            tmp_list.append(xp.dot(hhdata_tmp, hhdata_tmp.T) / (np.prod(hsizetmp) / hsizetmp[1]))
                            tmp_list2.append(tmp_mean.reshape(tmp_mean.shape[0], -1) * num_classii)

                    cov_class.append(tmp_list)
                    mean_vec_class.append(tmp_list2)
                else:
                    print('loop: %d' % i)
                    tmpvec = xp.mean(hh.data, axis=1)
                    # Mcross = xp.outer(tmpvec,tmpvec)
                    cov[jj] = cov[jj] + xp.dot(hh.data, hh.data.T) / (
                        np.prod(hsize) / hsize[1])  # - Mcross # xp.cov(hh.data)
                    mean_vec[jj] = mean_vec[jj] + tmpvec
                    full_mean[jj] = full_mean[jj] + tmp_mean

                    for ii in range(num_class):
                        num_classii = len(xp.where(xp.array(t) == ii)[0])
                        if num_classii > 0:
                            class_samp_num[ii] = class_samp_num[ii] + num_classii
                            hhdata_tmp = hhdataorig[xp.where(xp.array(t) == ii)[0], :]
                            tmp_mean = xp.mean(hhdata_tmp, axis=0)
                            hsizetmp = hhdata_tmp.shape
                            hhdata_tmp = hhdata_tmp.swapaxes(0, 1).reshape(hsizetmp[1], -1)
                            cov_class[jj][ii] = xp.array(cov_class[jj][ii] + xp.dot(hhdata_tmp, hhdata_tmp.T) / (
                                np.prod(hsizetmp) / hsizetmp[1]))
                            # xp.cov(hhdata_tmp)*num_classii
                            tmp_mean = tmp_mean.reshape(tmp_mean.shape[0], -1)
                            mean_vec_class[jj][ii] = xp.array(mean_vec_class[jj][ii] + tmp_mean * num_classii)


                            # (ll,xout) = hout[-1]
                            # for ite = range(0,36):
                            #     pred = xout.data[36*i:36*i+1]
                            #     pred = pred.mean(axis=0)
                            #     # #acc = int(pred.argmax() == t.data[0])
                            #     te_index.append(ind_tmp[ite])
                            #     y_pred.append(pred.argmax())
                            #     y_test.append(te_labels[ind_tmp[ite]])
                            # accuracy = F.accuracy(x, t)

        w = []
        for (jj, (name, hh)) in enumerate(hout):
            mean_vec[jj] = mean_vec[jj] / Test_Num
            full_mean[jj] = full_mean[jj] / Test_Num
            # Mcross = xp.outer(mean_vec[jj],mean_vec[jj])
            # cov[jj] = cov[jj]/Test_Num - Mcross
            tmpdata = full_mean[jj]
            Mcross = xp.zeros((tmpdata.shape[0], tmpdata.shape[0]))
            for ii in range(0, tmpdata.shape[1]):
                Mcross = Mcross + xp.outer(tmpdata[:, ii], tmpdata[:, ii])
            Mcross = Mcross / tmpdata.shape[1]
            cov[jj] = cov[jj] / Test_Num - Mcross
            if b_usingGPU:
                tmp_covma = cuda.to_cpu(cov[jj])
            else:
                tmp_covma = cov[jj]
            # cov[jj] = cov[jj]/Test_Num
            w.append(LA.eigvalsh(tmp_covma))

            for ii in range(num_class):
                cov_class[jj][ii] = cov_class[jj][ii] / class_samp_num[ii]
                mean_vec_class[jj][ii] = mean_vec_class[jj][ii] / class_samp_num[ii]

        result_dir = './result'
        if b_colapse:
            datastore = shelve.open('%s/cifar_%s(%d|%d_epoch)_result' % (result_dir, model_name, exp_type, epoch_stage))
        else:
            datastore = shelve.open(
                '%s/cifar_%s(%d|%d_epoch)_wide_result' % (result_dir, model_name, exp_type, epoch_stage))
        datastore[str('wlist')] = w
        datastore[str('covlist')] = cov  # cuda.to_cpu(cov)
        datastore[str('meanveclist')] = mean_vec  # cuda.to_cpu(mean_vec)
        datastore[str('fullmeanlist')] = full_mean  # cuda.to_cpu(mean_vec)
        # datastore[str('num_params')] = count_VGG_params(model)
        # datastore[str('width_list')] = width_list
        datastore[str('cov_class')] = cov_class
        datastore[str('mean_vec_class')] = mean_vec_class
        datastore.close()

        for (jj, wj) in enumerate(w):
            plt.clf()
            plt.plot(np.log(wj))
            if b_colapse:
                plt.savefig('./result/%s(%d|%d-epoch)_eig_fig_%d.jpg' % (model_name, exp_type, epoch_stage, jj))
            else:
                plt.savefig('./result/%s(%d|%d-epoch)_wide_eig_fig_%d.jpg' % (model_name, exp_type, epoch_stage, jj))
