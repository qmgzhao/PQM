import sys, os
import re
import time

import whisper
import editdistance
from dataloader import get_dataloader
import argparse
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim import SGD, Adam
from transformers import WhisperTokenizer
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

parser = argparse.ArgumentParser(description = 'Running Whisper experiments')

# set arguments for training and decoding. 
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--modeltype', type=str, default="base.en")
parser.add_argument('--train_json', type=str, default="data/LibriSpeech/train_clean_100.json")
parser.add_argument('--dev_json', type=str, default="data/LibriSpeech/dev.json")
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--nepochs', type=int, default=10)
parser.add_argument('--expdir', type=str, default="exp/origmodel")
parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--decay_pct', type=float, default=1)
parser.add_argument('--warmup_pct', type=float, default=0.0)
parser.add_argument('--log_interval', type=int, default=200)
parser.add_argument('--logfile', type=str, default="log")
parser.add_argument('--accumgrad', type=int, default=1)
parser.add_argument('--loadfrom', type=str, default="")
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--lora_alpha', type=int, default=64)
parser.add_argument('--lora_r', type=int, default=4)
parser.add_argument('--lora_mode', type=int, default=0)
parser.add_argument('--scheduler', type=str, default="warmuplr")
parser.add_argument('--ftrain_qtrain', action="store_true")
parser.add_argument('--lora_pretrain', type=int, default=0)
parser.add_argument('--save_snapshot', action="store_true")
args = parser.parse_args()

def logging(s, logfile, logging_=True, log_=True):
    print(s)
    if log_:
        with open(logfile, 'a+') as f_log:
            f_log.write(s + '\n')

# trainer
logging(f"batch_size: {args.batch_size}", args.logfile)
# lora
logging(f"lora_pretrain: {args.lora_pretrain}", args.logfile)
logging(f"lora_mode: {args.lora_mode}, lora_r: {args.lora_r}, lora_alpha: {args.lora_alpha}", args.logfile)
# scheduler
logging(f"scheduler: {args.scheduler}, lr: {args.lr}, max_epoch: {args.nepochs}", args.logfile)
# loadfrom
logging(f"loadfrom: {args.loadfrom}", args.logfile)

##################
# Model
##################
torch.manual_seed(args.seed)

if args.loadfrom != "":
    model = torch.load(args.loadfrom)
else:
    model = whisper.load_model(args.modeltype)
model.to(args.device)
model.train()

options = whisper.DecodingOptions(language="en", fp16=False, without_timestamps=True)
tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual, language="en")
decodetask = whisper.decoding.DecodingTask(model, options)
logit_filters = decodetask.logit_filters
sot_sequence = decodetask.sot_sequence
sotlen = len(sot_sequence)

##################
# Lora
##################
if args.lora_mode != 999:  # don't use lora
    import loralib as lora
    import sys
    import whisper
    from lora_replace import replace_attn_layers, replace_conv_layers, replace_attn_conv_layers, replace_attn_conv_layers_encoder, replace_linear_layers, replace_linear_conv_layers

    # replace layer
    if args.lora_pretrain == 0:
        if args.lora_mode == 0:
            replace_attn_layers(model, lora_alpha=args.lora_alpha, lora_r=args.lora_r)
        elif args.lora_mode == 1:
            replace_conv_layers(model, lora_alpha=args.lora_alpha, lora_r=args.lora_r)
        elif args.lora_mode == 2:
            replace_attn_conv_layers(model, lora_alpha=args.lora_alpha, lora_r=args.lora_r)
        elif args.lora_mode == 3:
            replace_attn_conv_layers_encoder(model, lora_alpha=args.lora_alpha, lora_r=args.lora_r)
        elif args.lora_mode == 4:
            replace_linear_layers(model, lora_alpha=args.lora_alpha, lora_r=args.lora_r)
        elif args.lora_mode == 5:
            replace_linear_conv_layers(model, lora_alpha=args.lora_alpha, lora_r=args.lora_r)
    
    model.to(args.device)

    # freeze
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
        else:
            p.requires_grad = True

##################
# Data Loader
##################
trainloader = get_dataloader(args.train_json, args.batch_size, loadtarget=True, tokenizer=tokenizer)
devloader = get_dataloader(args.dev_json, args.batch_size, loadtarget=True, tokenizer=tokenizer)

##################
# Training
##################
criterion = torch.nn.NLLLoss()
optimiser = Adam(model.parameters(), lr=args.lr)

##################
# Start Training
##################
logging("Start of training", args.logfile)
bestacc = 0
for epoch in range(args.nepochs):
    start = time.time()
    totalloss = 0
    for idx, data in enumerate(trainloader):
        uttnames, fbank, tgt = data
        fbank = fbank.to(model.device)
        origtarget = [torch.tensor(list(sot_sequence) + y, dtype=torch.long) for y in tgt]
        target = pad_sequence(origtarget, batch_first=True, padding_value=-100).to(model.device)
        targetmask = target != -100
        
        optimiser.zero_grad()
        
        logits = model(fbank, target * targetmask)
        output = torch.log_softmax(logits[:, sotlen-1:-1], dim=-1)
        loss = F.nll_loss(output.view(-1, output.size(-1)), target[:, sotlen:].reshape(-1))

        loss = loss / args.accumgrad

        loss.backward()
        totalloss += loss.item()

        if (idx + 1) % args.accumgrad == 0:
            # LR scheduler
            if args.scheduler == "warmuplr":
                currentstep = epoch * len(trainloader) + idx + 1
                totalstep = args.nepochs * len(trainloader)
                if currentstep > int(args.decay_pct * totalstep):
                    factor = (totalstep - currentstep) / (totalstep - int(args.decay_pct * totalstep))
                    optimiser.param_groups[0]['lr'] = args.lr * max(0, factor)
                elif currentstep < int(args.warmup_pct * totalstep):
                    factor = currentstep / int(args.warmup_pct * totalstep)
                    optimiser.param_groups[0]['lr'] = args.lr * factor
            elif args.scheduler == "fixlr":
                pass
            elif args.scheduler == "steplr":
                optimiser.param_groups[0]['lr'] = args.lr * (0.9 ** epoch)
            optimiser.step()
            logging("model-parameter update.", args.logfile)
            

        if (idx + 1) % args.log_interval == 0:
            logging("{} / {} steps finished in {} | Loss: {} | lr: {}".format(
                idx, len(trainloader), time.time()-start, totalloss/args.log_interval, optimiser.param_groups[0]['lr']),
                 args.logfile)
            totalloss = 0

    # Validation
    totalvalset = 0
    totalvalacc = 0
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(devloader):
            uttnames, fbank, tgt = data
            fbank = fbank.to(model.device)
            target = [torch.tensor(list(sot_sequence) + y, dtype=torch.long) for y in tgt]
            # target = [torch.tensor(y, dtype=torch.long) for y in tgt]
            target = pad_sequence(target, batch_first=True, padding_value=-100).to(model.device)
            targetmask = target != -100

            # Forward biasing model
            logits = model(fbank, target * targetmask)
            output = torch.log_softmax(logits[:, sotlen-1:-1], dim=-1)

            target = target[:, sotlen:]
            output = output.view(target.size(0), target.size(1), -1).max(dim=-1)[1]
            totalvalacc += ((output == target) * targetmask[:, sotlen:]).sum()
            totalvalset += targetmask[:, sotlen:].sum()

            # result = whisper.decode(model, fbank, options)
            if (idx + 1) % args.accumgrad == 0:
                logging("{} out of {} finished | time elapsed {} | ACC: {}".format(
                    idx, len(devloader), time.time()-start, totalvalacc/totalvalset), args.logfile)
        logging("[epoch {}] Total ACC: {}".format(epoch+1, totalvalacc/totalvalset), args.logfile)

        totalacc = totalvalacc / totalvalset
    if totalacc > bestacc:
        torch.save(model, os.path.join(args.expdir, "model.acc.best"))
        bestacc = totalacc
        logging("Saving best model at epoch {}".format(epoch+1), args.logfile)

    if args.save_snapshot:
        torch.save(model, os.path.join(args.expdir, "snapshot.ep.{}".format(epoch+1)))
