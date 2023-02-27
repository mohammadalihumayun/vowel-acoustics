# import libraries

import python_speech_features as psf
#from librosa import feature as lbf
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from scipy import stats

from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# import dataset audio and descriptions

ind_vow_aud=np.load('/content/drive/MyDrive/paudio.npy',allow_pickle=True)
ind_descrs=np.load('/content/drive/MyDrive/description.npy')
ind_samprt=np.load('/content/drive/MyDrive/samp_rates.npy')
srate_diction={'file':'srate'}
for x in ind_samprt:
  srate_diction[x[1][:-4]]=int(x[0])
#
ind_vow_des=np.array([[x[1][-19:],int(srate_diction[x[1][-19:]]),x[-1],x[1][2:5]] for x in ind_descrs])
#
dialects=np.unique(ind_vow_des[:,-1])
sents=np.unique(ind_vow_des[:,0])
spkrs=np.unique([x[:8] for x in sents])
#
spk_all=[x[:8] for x in ind_vow_des[:,0]]
spk_seg=[[x for x in spkrs if x[:3]==d] for d in dialects]

#input_feats=compute_feats('logf',ind_vow_aud,ind_vow_des[:,1],0.025,1200,128) #ncp
#np.save('/content/drive/MyDrive/prd/ind_vwl_mid_lgfbank_25ms_128',input_feats)
input_feats=np.load('/content/drive/MyDrive/prd/ind_vwl_mid_lgfbank_25ms_128.npy')

energies=np.array([sum(x) for x in input_feats])

input_feats_orig=np.load('/content/drive/MyDrive/prd/ind_vwl_mid_lgfbank_25ms_128.npy')

input_feats=compute_feats('mfcc',ind_vow_aud,ind_vow_des[:,1],0.025,1200,26,13) # mfcc

np.save('/content/drive/MyDrive/prd/ind_vwl_mid_mfcc13',input_feats)

from scipy.spatial.distance import pdist

accdistfeat=[]
for sp in np.unique(spk_all):
     phmean=[]
     for ph in ['AE', 'AH', 'IH', 'IY', 'UW']:
         phmean.append(np.mean(input_feats[(np.array(spk_all)==sp)&(ind_vow_des[:,2]==ph)][:,1:],axis=0))
     accdistfeat.append(pdist(phmean))

accdistlbl=[x[:3] for x in np.unique(spk_all)]
accdistfeat=np.array(accdistfeat)
accdistlbl=np.array(accdistlbl)

spk_tst=test_speakers(12)
tst_idx=np.in1d(np.unique(spk_all),spk_tst)
trn_idx=np.invert(tst_idx)

svc_inst=SVC(kernel='linear')#',degree=12)#rbf
    svc_inst.fit(accdistfeat[trn_idx],accdistlbl[trn_idx])
    prd=svc_inst.predict(accdistfeat[tst_idx])

sum(prd==accdistlbl[tst_idx])/len(tst_idx)

def ae_model():
 inputlayer = Input(shape=(128,))
 hli=Dense(64)(inputlayer)
 emb_layer=Dense(32)(hli)
 hlo=Dense(64)(inputlayer)
 outputlayer = Dense(128)(hlo)
 outputlayer_sec = Dense(1)(hlo)
 embedmodel = Model(inputlayer,emb_layer)
 regenmodel = Model(inputlayer,[outputlayer,outputlayer_sec])
 regenmodel.compile(optimizer='Adam', loss='mse')
 #regenmodel.summary()
 return regenmodel,embedmodel

def test_speakers(num):
  spk_tst=[]
  for x in spk_seg:
    spk_tst.extend(x[:num])
  return spk_tst

def speaker_acc(spk_tst,prd):
 acc=0
 for x in spk_tst:
  sp=np.array(spk_all)[tst_sel]==x
  if stats.mode(prd[sp])[0][0]==x[:3]:
    acc+=1
 return acc/len(spk_tst)

## train and save svm prediction for all vowels & compare with AE
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist


#regenmodel,embedmodel=ae_model()
#es = EarlyStopping(monitor='loss', mode='min',verbose=1, patience=1)
#
#regenmodel.fit(input_feats,input_feats,epochs=150,batch_size=128,shuffle=True,callbacks=[es])
#ae_emb=embedmodel.predict(input_feats)
#
for ftr in [43]:
 #input_feats=input_feats_orig[:,ftr:ftr+43]
 predictions=[]
 for vwl in ['AE', 'AH', 'IH', 'IY', 'UW']:
  selection=ind_vow_des[:,2]==vwl# AE=BAT
  #
  #acr=np.mean(input_feats[selection],axis=0).reshape(1,-1)
  #dists=cdist(input_feats[selection],acr, metric='cosine')#correlation, cosine
  #regenmodel,embedmodel=ae_model()
  #es = EarlyStopping(monitor='loss', mode='min',verbose=1, patience=1)
  #regenmodel.fit(input_feats[selection],[input_feats[selection],dists],epochs=150,batch_size=128,shuffle=True,callbacks=[es])
  #ae_emb=embedmodel.predict(input_feats[selection])
  #
  for tst_spk_num in [12]:#,24,36,48
    spk_tst=test_speakers(tst_spk_num)
    tst_idx=np.in1d(spk_all,spk_tst)
    trn_idx=np.invert(tst_idx)
    #
    trn_sel=np.logical_and(selection, trn_idx)
    tst_sel=np.logical_and(selection, tst_idx)
    #
    tst_emb=np.in1d(np.array(spk_all)[selection],spk_tst)
    trn_emb=np.invert(tst_emb)
    #
    
    svc_inst=SVC(kernel='poly',degree=12)#rbf
    #svc_inst=SVC(kernel='rbf')#rbf
    svc_inst.fit(input_feats[trn_sel],ind_vow_des[:,-1][trn_sel])
    prd=svc_inst.predict(input_feats[tst_sel])
    acc=sum(prd==ind_vow_des[:,-1][tst_sel])/sum(tst_sel)
    spk_acc=speaker_acc(spk_tst,prd)
    f=open('/content/drive/MyDrive/feat_accuracies.txt','a')
    f.write('\nmfcc13,poly9_svm,'+vwl+','+str(tst_spk_num)+','+str(acc)+','+str(spk_acc))
    f.close()
    predictions.append(prd)
    #
    '''
    svc_inst=SVC(kernel='poly',degree=12)
    svc_inst.fit(ae_emb[trn_emb],ind_vow_des[:,-1][trn_sel])
    prd=svc_inst.predict(ae_emb[tst_emb])
    acc=sum(prd==ind_vow_des[:,-1][tst_sel])/sum(tst_emb)
    spk_acc=speaker_acc(spk_tst,prd)
    f=open('/content/drive/MyDrive/feat_accuracies.txt','a')
    f.write('\nAE_spatial_emb_32,poly9_svm,'+vwl+','+str(tst_spk_num)+','+str(acc)+','+str(spk_acc))
    f.close()
    
    svc_inst=SVC(kernel='rbf')
    svc_inst.fit(ae_emb[trn_sel],ind_vow_des[:,-1][trn_sel])
    prd=svc_inst.predict(ae_emb[tst_sel])
    acc=sum(prd==ind_vow_des[:,-1][tst_sel])/sum(tst_sel)
    spk_acc=speaker_acc(spk_tst,prd)
    f=open('/content/drive/MyDrive/feat_accuracies.txt','a')
    f.write('\nfull_train_AE_emb_32,'+vwl+','+str(tst_spk_num)+','+str(acc)+','+str(spk_acc))
    f.close()
    print(vwl,acc)
    '''
 np.save('/content/drive/MyDrive/prd/pred_12spk_mfc13',predictions)

pred=np.load('/content/drive/MyDrive/prd/pred_12spk_mfcc_poly12svm.npy',allow_pickle=True)

# computing speaker level prediction accuracy 
import statistics
from collections import Counter

spk_tst=test_speakers(12)
pred=np.load('/content/drive/MyDrive/prd/pred_12spk_poly12svm.npy',allow_pickle=True)
pred_mfc=np.load('/content/drive/MyDrive/prd/pred_12spk_mfcc_poly12svm.npy',allow_pickle=True)
pred_lgf0=np.load('/content/drive/MyDrive/prd/pred_12spk_logf0.npy',allow_pickle=True)
pred_lgf43=np.load('/content/drive/MyDrive/prd/pred_12spk_logf43.npy',allow_pickle=True)
pred_lgf86=np.load('/content/drive/MyDrive/prd/pred_12spk_logf86.npy',allow_pickle=True)
pred_mfc13=np.load('/content/drive/MyDrive/prd/pred_12spk_mfc13.npy',allow_pickle=True)

tst_idx=np.in1d(spk_all,spk_tst)
pred_mfc=pred_mfc[4*np.arange(5)]# for 12 speakers only

# vowel specific speaker level prediction accuracy
predictions=[]
id=0
for vwl in ['AE', 'AH', 'IH', 'IY', 'UW']:
 selection=ind_vow_des[:,2]==vwl# AE=BAT
 tst_sel=np.logical_and(selection, tst_idx)
 prdv=[]
 acc=0
 for x in spk_tst:
  sp=np.array(spk_all)[tst_sel]==x
  lmt=30
  #cmn= np.argsort(-1*energies[tst_sel][sp])#pred[id][sp]==pred_mfc[id][sp]#np.logical_and(pred[id][sp]==pred_mfc[id][sp] , pred[id][sp]==pred_lgf0[id][sp])
  #prdsel=np.concatenate((pred[id][sp][cmn][:lmt],pred_lgf0[id][sp][cmn][:lmt],pred_mfc[id][sp][cmn][:lmt]),axis=0)#[np.argsort(-1*energies[tst_sel][sp])][:50]
  prdsel=pred_mfc13[id][sp]
  #print(len(prdsel),len(energies[tst_sel][sp]))
  prdv.append(stats.mode(prdsel)[0][0])
  if stats.mode(prdsel)[0][0]==x[:3]:
    acc+=1
 print(vwl,acc/len(spk_tst))
 predictions.append(prdv) 
 id+=1

from sklearn.metrics import confusion_matrix
lbl=np.array([x[:3] for x in np.array(spk_tst).flatten()])
#
vwl=['AE', 'AH', 'IH', 'IY', 'UW']
for k in range(5):
 plt.figure(k)
 cmat=confusion_matrix(lbl,predictions[k])
 plt.imshow(cmat,cmap='Blues')
 plt.xticks(list(range(5)),np.unique(lbl))
 plt.yticks(list(range(5)),np.unique(lbl))
 for i in range(5):
  for j in range(5):
    if int(cmat[i][j])>7:
      plt.text(j,i,int(cmat[i][j]),ha='center',va='center',color='w')
    else:
      plt.text(j,i,int(cmat[i][j]),ha='center',va='center')
 plt.title(vwl[k])
 plt.savefig(vwl[k],dpi=300)

#sum(predictions[3]==lbl)/len(lbl)
from sklearn.metrics import precision_recall_fscore_support as prf
lbl=np.array([x[:3] for x in np.array(spk_tst).flatten()])
#
fscr=[]
for p in predictions:
  fscr.append(prf(lbl,p)[2])
fscr=np.array(fscr)
#
for i in range(5):
  plt.bar([(x*7)+i+1 for x in list(range(5))],fscr[:,i],label=np.unique(lbl)[i])
plt.legend(bbox_to_anchor=(0.95, 0., 0.0, .95),ncol=1)
plt.savefig('/content/f_score_12spk_svmpoly12',dpi=300)

vwl=['AE', 'AH', 'IH', 'IY', 'UW']
print('fscr_12_speaker_logfbnk128_svmpoly_degree12')
print(np.unique(lbl))
for i in range(len(vwl)):
  print(vwl[i],fscr[i])

def compute_feats(type,audios,srates,frmsize,fftsize,filts,ceps=None):# mfcc logfbank fbank
 features=[]
 for x,y in zip(audios,srates):
   samps=int(frmsize*(int(y)//2))
   if type=='fbnk':
    ft=psf.fbank(x[(len(x)//2)-samps:(len(x)//2)+samps], samplerate=int(y), winlen=frmsize, winstep=frmsize, nfilt=filts, nfft=fftsize)[0]#nfft=512 ,numcep=26
   elif type=='logf':
    ft=psf.logfbank(x[(len(x)//2)-samps:(len(x)//2)+samps], samplerate=int(y), winlen=frmsize, winstep=frmsize, nfilt=filts, nfft=fftsize)#nfft=512 ,numcep=26
   elif type=='mfcc':
    ft=psf.mfcc(x[(len(x)//2)-samps:(len(x)//2)+samps], samplerate=int(y), winlen=frmsize, winstep=frmsize, nfilt=filts, nfft=fftsize, numcep=ceps)#nfft=512 ,numcep=26
   features.append(ft)#[len(ft)//2]
 return np.squeeze(features) 
 #
def classify(features,classes,krl):
 trn_tst_idx=np.arange(len(features))
 np.random.shuffle(trn_tst_idx)
 svc_inst=SVC(kernel=krl)
 svc_inst.fit(features[trn_tst_idx][:len(trn_tst_idx)//5],classes[trn_tst_idx][:len(trn_tst_idx)//5])
 prd=svc_inst.predict(features[trn_tst_idx][len(trn_tst_idx)//5:])
 acc=sum(prd==classes[trn_tst_idx][len(trn_tst_idx)//5:])/len(trn_tst_idx)
 print(acc)
 return acc

# compare psf features and their time frames and frequency filters

for flt in [256]:#[32,128,64]:
  for vwl in ['AE', 'AH', 'IH', 'IY', 'UW']:
    for ftype in ['logf','mfcc']:#'fbnk' #logf mfcc
      for twr in [0.5,1,2]:#
        selection=ind_vow_des[:,2]==vwl# AE=BAT
        input_feats=compute_feats(ftype,ind_vow_aud[selection],ind_vow_des[:,1][selection],0.025*twr,int(1200*twr),flt,flt) #ncp
        #
        for d in dialects:
          plt.plot(np.mean(input_feats[ind_vow_des[selection][:,-1]==d],axis=0),label=d)
        plt.title(vwl+'_mean_'+ftype+'_'+str(flt)+'_pt'+str(int(twr*10)))
        plt.legend()
        plt.savefig('/content/drive/MyDrive/plots/'+vwl+'_mean_'+ftype+'_'+str(flt)+'_pt'+str(int(twr*10)),dpi=300)
        plt.close()
        for d in dialects:
          plt.plot(np.var(input_feats[ind_vow_des[selection][:,-1]==d],axis=0),label=d)
        plt.title(vwl+'_var_'+ftype+'_'+str(flt)+'_pt'+str(int(twr*10)))
        plt.legend()
        plt.savefig('/content/drive/MyDrive/plots/'+vwl+'_var_'+ftype+'_'+str(flt)+'_pt'+str(int(twr*10)),dpi=300)
        plt.close()
        #
        plt.bar(np.arange(flt),np.var(input_feats,axis=0))
        plt.title(vwl+'_allvar_'+ftype+'_'+str(flt)+'_pt'+str(int(twr*10)))
        plt.savefig('/content/drive/MyDrive/plots/'+vwl+'_allvar_'+ftype+'_'+str(flt)+'_pt'+str(int(twr*10)),dpi=300)
        plt.close()
        #
        accuracy=classify(input_feats,ind_vow_des[:,-1][selection],'rbf') #'rbf' 'linear'
        f=open('/content/drive/MyDrive/plots/accuracies.txt','a')
        f.write('\n'+vwl+','+ftype+','+str(flt)+','+str(accuracy)+',_pt'+str(int(twr*10)))
        f.close()

# compare filter ranges for accuracy
flt=128
rng=1+flt//3
ftype='logf'
twr=1
#for flt in [128]:#[32,128,64]:
for rn1 in range(3):
  for vwl in ['AE', 'AH', 'IH', 'IY', 'UW']:
    #for ftype in ['logf','mfcc']:#'fbnk' #logf mfcc
      #for twr in [0.5,1,2]:#
        selection=ind_vow_des[:,2]==vwl# AE=BAT
        #
        accuracy=classify(input_feats[selection][:,rng*rn1:rng*(rn1+1)],ind_vow_des[:,-1][selection],'rbf') #'rbf' 'linear'
        f=open('/content/drive/MyDrive/accuracies.txt','a')
        f.write('\n'+vwl+','+ftype+','+str(flt)+','+str(rng)+','+str(rn1)+','+str(accuracy)+',_pt'+str(int(twr*10)))
        f.close()

# compute mutual info for psf features

from sklearn.feature_selection import mutual_info_classif
for flt in [16,32]:#,64,128]:
  for vwl in ['AE', 'AH', 'IH', 'IY', 'UW']:
    for ftype in ['mfcc','logf']:#
      ncp=flt
      selection=ind_vow_des[:,2]==vwl# AE=BAT
      input_feats=compute_feats(ftype,ind_vow_aud[selection],ind_vow_des[:,1][selection],0.025,1200,flt,ncp) #ncp
      importances=mutual_info_classif(input_feats,ind_vow_des[:,-1][selection])
      plt.bar(np.arange(flt),importances)
      plt.title(vwl+'_'+ftype+'_'+str(flt))
      plt.savefig('/content/drive/MyDrive/plots/'+vwl+'_'+ftype+'_'+str(flt),dpi=300)
      plt.close()
      f=open('/content/drive/MyDrive/plots/feature_results.txt','a')
      f.write('\nmutual_info,'+vwl+','+ftype+','+str(flt)+','+','.join(str(x) for x in importances))
      f.close()

# compute chi-2 for psf features

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

for flt in [128]:#,64,128]:
  for ftype in ['logf','mfcc']:#
    for vwl in ['AE', 'AH', 'IH', 'IY', 'UW']:
      ncp=flt
      selection=ind_vow_des[:,2]==vwl# AE=BAT
      input_feats=compute_feats(ftype,ind_vow_aud[selection],ind_vow_des[:,1][selection],0.025,1200,flt,ncp) #ncp
      input_feats=scaler.fit_transform(input_feats)
      chi_best = SelectKBest(score_func=chi2, k='all')
      chi_fit = chi_best.fit(input_feats,ind_vow_des[:,-1][selection])
      chi_scores = chi_fit.scores_
      plt.bar(np.arange(flt),chi_scores)
      plt.title('chi_best_'+vwl+'_'+ftype+'_'+str(flt))
      plt.savefig('/content/drive/MyDrive/chi2_plots/chi_best_'+vwl+'_'+ftype+'_'+str(flt),dpi=300)
      plt.close()
      f=open('/content/drive/MyDrive/chi2_plots/chi_feature_results.txt','a')
      f.write('\nchi_best,'+vwl+','+ftype+','+str(flt)+','+','.join(str(x) for x in chi_scores))
      f.close()

def compute_fft(twr,audios,srates):
  features=[]
  dial_select=[]
  for x,y in zip(audios,srates):
    samps=int(0.025*twr*(int(y)//2))
    #if len(x)>2*samps:
    ft=np.fft.rfft(x[(len(x)//2)-samps:(len(x)//2)+samps],n=2*samps)#nfft=512 ,numcep=26
    features.append(ft)#[len(ft)//2]
    #dial_select.append(z)
  return np.array(features)#, dial_select

## plot fft for vowels
for vwl in ['AE', 'AH', 'IH', 'IY', 'UW']:
      for twr in [0.5,1,2]:#
        selection=ind_vow_des[:,2]==vwl# AE=BAT
        input_feats=compute_fft(twr,ind_vow_aud[selection],ind_vow_des[:,1][selection]) #ncp ,ind_vow_des[:,-1][selection]
        #
        for d in dialects:
          plt.plot(np.mean(input_feats[ind_vow_des[:,-1][selection]==d],axis=0),label=d)
        plt.title(vwl+'_fft_pt_'+str(int(twr*10)))
        plt.legend()
        plt.savefig('/content/drive/MyDrive/fft/'+vwl+'_fft_pt_'+str(int(twr*10)),dpi=300)
        plt.close()

def compute_librosa(type,audios,srates,frmsize,filts,ceps=None):# mfcc logfbank fbank
 features=[]
 for x,y in zip(audios,srates):
   samps=int(frmsize*(int(y)))
   if type=='lmsp':
    if (len(x)//2)-(samps//2) <0:
      ft=lbf.melspectrogram(x.astype('float'), n_mels=filts,sr=int(y), n_fft=samps, hop_length=samps, win_length=samps)#nfft=512 ,numcep=26
    else:
      ft=lbf.melspectrogram(x[(len(x)//2)-(samps//2)+1:(len(x)//2)+(samps//2)].astype('float'), n_mels=filts,sr=int(y), n_fft=samps, hop_length=samps, win_length=samps)#nfft=512 ,numcep=26
   elif type=='lmfc':
    if (len(x)//2)-(samps//2) <0:
      ft=lbf.mfcc(x.astype('float'), n_mfcc=ceps,n_mels=filts,sr=int(y), n_fft=samps, hop_length=samps, win_length=samps)#nfft=512 ,numcep=26
    else:
      ft=lbf.mfcc(x[(len(x)//2)-(samps//2)+1:(len(x)//2)+(samps//2)].astype('float'), n_mfcc=ceps,n_mels=filts,sr=int(y), n_fft=samps, hop_length=samps, win_length=samps)#nfft=512 ,numcep=26
   features.append(ft)#[len(ft)//2]
 return np.squeeze(features)

# compare librosa features
for flt in [128]:#[32,64,128]:
  for vwl in ['AE', 'AH','IH', 'IY', 'UW']:#
    for ftype in ['lmfc','lmsp']:#'fbnk' #logf mfcc
      for twr in [0.5,2]:#[0.5,1,2]:#[1]:#[0.5,2]:# replace with 1 and re run
        selection=ind_vow_des[:,2]==vwl# AE=BAT
        input_feats=compute_librosa(ftype,ind_vow_aud[selection],ind_vow_des[:,1][selection],0.025*twr,flt,flt) #ncp
        for d in dialects:
          plt.plot(np.mean(input_feats[ind_vow_des[selection][:,-1]==d],axis=0),label=d)
        plt.title(vwl+'_mean_'+ftype+'_'+str(flt)+'_pt'+str(int(twr*10)))
        plt.legend()
        plt.savefig('/content/drive/MyDrive/plots/'+vwl+'_mean_'+ftype+'_'+str(flt)+'_pt'+str(int(twr*10)),dpi=300)
        plt.close()
        for d in dialects:
          plt.plot(np.var(input_feats[ind_vow_des[selection][:,-1]==d],axis=0),label=d)
        plt.title(vwl+'_var_'+ftype+'_'+str(flt)+'_pt'+str(int(twr*10)))
        plt.legend()
        plt.savefig('/content/drive/MyDrive/plots/'+vwl+'_var_'+ftype+'_'+str(flt)+'_pt'+str(int(twr*10)),dpi=300)
        plt.close()
        #
        plt.bar(np.arange(flt),np.var(input_feats,axis=0))
        plt.title(vwl+'_allvar_'+ftype+'_'+str(flt)+'_pt'+str(int(twr*10)))
        plt.savefig('/content/drive/MyDrive/plots/'+vwl+'_allvar_'+ftype+'_'+str(flt)+'_pt'+str(int(twr*10)),dpi=300)
        plt.close()
        #
        accuracy=classify(input_feats,ind_vow_des[:,-1][selection],'rbf') #'rbf' 'linear'
        f=open('/content/drive/MyDrive/plots/accuracies.txt','a')
        f.write('\n'+vwl+','+ftype+','+str(flt)+','+str(accuracy)+',_pt'+str(int(twr*10)))
        f.close()

## plot raw wave for mean vowel audio segments
lnts=[len(x) for x in ind_vow_aud]
samps=min(lnts)
aud_trm=np.array([x[(len(x)//2)-(samps//2):(len(x)//2)+(samps//2)] for x in ind_vow_aud])
for vwl in ['AE', 'AH','IH', 'IY', 'UW']:#
  selection=ind_vow_des[:,2]==vwl# AE=BAT
  for d in dialects:
    plt.plot(np.mean(aud_trm[selection][ind_vow_des[selection][:,-1]==d],axis=0),label=d)
    plt.title(vwl+'_'+d)
    plt.legend()
    plt.savefig('/content/drive/MyDrive/plots/'+vwl+'_'+d,dpi=300)
    plt.close()

audfbk.shape

# plot complete raw audio and features for samples

fig, axs = plt.subplots(3, 2,figsize=(10,15),dpi=300)#,sharex=True,sharey=True
smp=10
frmsize=0.025
filts=32
y=ind_vow_des[:,1][smp]
samps=int(frmsize*(int(y)//2))
aud=ind_vow_aud[smp]
audseg=aud[(len(aud)//2)-samps:(len(aud)//2)+samps]
audfft=np.fft.rfft(audseg,n=2*samps)
audfbk=psf.fbank(audseg, samplerate=int(y), winlen=frmsize, winstep=frmsize, nfilt=filts, nfft=1200)[0][0]#
audlfb=psf.logfbank(audseg, samplerate=int(y), winlen=frmsize, winstep=frmsize, nfilt=filts,nfft=1200 )[0]#
audmfc=psf.mfcc(audseg, samplerate=int(y), winlen=frmsize, winstep=frmsize, nfilt=filts, nfft=1200, numcep=filts)[0]
axs[0, 0].plot(aud)
axs[0, 1].plot(audseg)
axs[1, 0].plot(audfft)
axs[1, 1].bar(list(range(filts)),audfbk)
axs[2, 0].bar(list(range(filts)),audlfb)
axs[2, 1].bar(list(range(filts)),audmfc)
fig.suptitle(ind_vow_des[:,2][smp])
#plt.title(vwl+'_full')
plt.savefig('/content/'+ind_vow_des[:,2][smp],dpi=300)
#plt.close()
int(y)
