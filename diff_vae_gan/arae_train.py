import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import math, random, sys
import numpy as np
import argparse
from collections import deque
import cPickle as pickle

from fast_jtnn import *
import rdkit

###### hyperparameter setting 
similarity_threshold = 0.20
similarity_weight = -1e-3 #-1e-2 negative 
drd_weight = 0.0 ##-1e-3  negative 
drd_threshold = 0
qed_weight = 0 ##-1e-3 
qed_threshold = 0.5
molecule_size_weight = 1e-4 ## 1e-3 positive
molecule_size_threshold = 0 ## don't use 
alpha = 0.5
###### hyperparameter setting 

prefix = "sim_1e3_020_size_1e4_0" 
##  sim_1e2_020, sim_1e3_020, 
##  sim_1e2_020_drd_1e2_0, sim_1e2_020_drd_1e3_0

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True)
parser.add_argument('--ymols', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--save_dir', type=str, default=None)
parser.add_argument('--load_epoch', type=int, default=-1)

parser.add_argument('--hidden_size', type=int, default=300)
parser.add_argument('--rand_size', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--depthT', type=int, default=6)
parser.add_argument('--depthG', type=int, default=3)
parser.add_argument('--share_embedding', action='store_true')
parser.add_argument('--use_molatt', action='store_true')

parser.add_argument('--diter', type=int, default=5)
parser.add_argument('--beta', type=float, default=10)
parser.add_argument('--disc_hidden', type=int, default=300)
parser.add_argument('--gan_batch_size', type=int, default=10)
parser.add_argument('--gumbel', action='store_true')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--gan_lrG', type=float, default=1e-4)
parser.add_argument('--gan_lrD', type=float, default=1e-4)
parser.add_argument('--kl_lambda', type=float, default=1.0)
parser.add_argument('--clip_norm', type=float, default=50.0)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--anneal_rate', type=float, default=0.9)
  
args = parser.parse_args()
print args

vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
vocab = Vocab(vocab)

model = DiffVAE(vocab, args).cuda()
GAN = ScaffoldGAN(model, args.disc_hidden, beta=args.beta, gumbel=args.gumbel).cuda()

if args.load_epoch >= 0:
    GAN.load_state_dict(torch.load(args.save_dir + "/gan.iter-" + str(args.load_epoch)))
    model.load_state_dict(torch.load(args.save_dir + "/model.iter-" + str(args.load_epoch)))
    GAN.reset_netG(model)
else:
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

    for param in GAN.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

print "Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,)
print "GAN #Params: %dK" % (sum([x.nelement() for x in GAN.parameters()]) / 1000,)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
optimizerG = optim.Adam(model.parameters(), lr=args.gan_lrG, betas=(0, 0.9)) #generator is model parameter!
optimizerD = optim.Adam(GAN.netD.parameters(), lr=args.gan_lrD, betas=(0, 0.9))

scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)
scheduler.step()

PRINT_ITER = 20 
param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

assert args.gan_batch_size <= args.batch_size
num_epoch = (args.epoch - args.load_epoch - 1) * (args.diter + 1) * 10

x_loader = PairTreeFolder2(args.train, vocab, args.gan_batch_size, num_workers=4, y_assm=False, replicate=num_epoch)
x_loader = iter(x_loader)
y_loader = MolTreeFolder(args.ymols, vocab, args.gan_batch_size, num_workers=4, assm=False, replicate=num_epoch)
y_loader = iter(y_loader)

for epoch in xrange(args.load_epoch + 1, args.epoch):
    meters = np.zeros(7)
    main_loader = PairTreeFolder2(args.train, vocab, args.batch_size, num_workers=4)
    alpha = 1.0 / args.epoch * (epoch+1)
    print("alpha is", alpha) 
    for it, batch in enumerate(main_loader):
        print "\r iterator",it,
        #1. Train encoder & decoder
        model.zero_grad()
        smiles, x_batch, y_batch = batch


        ### 1.1 train MOLER 
        '''
        reward = model.forward_reward(smiles, x_batch,\
                similarity_threshold=similarity_threshold, similarity_weight=similarity_weight, \
                drd_threshold = drd_threshold, drd_weight = drd_weight, \
                qed_threshold = qed_threshold, qed_weight = qed_weight, \
                molecule_size_weight = molecule_size_weight, molecule_size_threshold = molecule_size_threshold) 
        if reward is not None:
            reward.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            print "reward", reward.data.item(), 
        '''
        ### 1.1 train MOLER 


        ### 1.2 train generative model
        loss, kl_div, wacc, tacc, sacc = model(x_batch, y_batch, args.kl_lambda, alpha)
        print "loss", loss.data.item()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()
        ### 1.2 train generative model


        #2. Train discriminator
        for i in xrange(args.diter):
            GAN.netD.zero_grad()
            _, x_batch, _ = next(x_loader)
            y_batch = next(y_loader)
            d_loss, gp_loss = GAN.train_D(x_batch, y_batch, model)
            optimizerD.step()

        #3. Train generator (ARAE fashion)
        model.zero_grad()
        GAN.zero_grad()
        _, x_batch, _ = next(x_loader)
        y_batch = next(y_loader)
        g_loss = GAN.train_G(x_batch, y_batch, model)
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizerG.step()
        
        meters = meters + np.array([kl_div, wacc * 100, tacc * 100, sacc * 100, d_loss, g_loss, gp_loss])

        if (it + 1) % PRINT_ITER == 0:
            meters /= PRINT_ITER
            print "KL: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, Disc: %.4f, Gen: %.4f, GP: %.4f, PNorm: %.2f, %.2f, GNorm: %.2f, %.2f" % (meters[0], meters[1], meters[2], meters[3], meters[4], meters[5], meters[6], param_norm(model), param_norm(GAN.netD), grad_norm(model), grad_norm(GAN.netD))
            sys.stdout.flush()
            meters *= 0

        if (it > 0) and (it % 500 == 0):
            torch.save(model.state_dict(), args.save_dir + "/model.iter-" + str(epoch) + "_" + str(it) + "_" + prefix)
            torch.save(GAN.state_dict(), args.save_dir + "/gan.iter-" + str(epoch) + "_" + str(it) + "_" + prefix)
            print "save ok"
        
        '''if (it > 0) and (it % 5000 == 0):
            torch.save(model.state_dict(), args.save_dir + "/model.iter-" + str(epoch) +'_' + str(it))
            torch.save(GAN.state_dict(), args.save_dir + "/gan.iter-" + str(epoch) +'_' + str(it))
            print "save", it
        '''

    scheduler.step()

    print "learning rate: %.6f" % scheduler.get_lr()[0]
    '''if args.save_dir is not None:
        torch.save(model.state_dict(), args.save_dir + "/model.iter-" + str(epoch))
        torch.save(GAN.state_dict(), args.save_dir + "/gan.iter-" + str(epoch))
        print "save ok"
    '''

















