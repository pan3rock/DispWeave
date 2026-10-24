"""Independent NumPy log-density evaluator for frozen physical-coordinate mixtures."""
import numpy as np
class FrozenDensities:
    def __init__(self, config):
        self.models={}
        for model in config.get('rj_correlated_models',[]):
            components=model['components'];weights=np.array([c.get('weight',1.) for c in components],float);weights/=weights.sum();prepared=[]
            for weight,c in zip(weights,components):
                mean=np.array(c['mean']);cov=np.array(c['covariance']);sign,det=np.linalg.slogdet(cov);assert sign==1
                prepared.append((mean,np.linalg.inv(cov),np.log(weight)-.5*(len(mean)*np.log(2*np.pi)+det)))
            self.models[model['layers']]=prepared
    def logq(self,row):
        values=[];x=np.array(row[4:]);k=int(row[1])
        for mean,precision,constant in self.models[k]:
            delta=x-mean;values.append(constant-.5*delta@precision@delta)
        values=np.array(values);maximum=values.max();return float(maximum+np.log(np.exp(values-maximum).sum()))
