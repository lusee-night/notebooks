import numpy as np


# global settings, fine for notebook
fcomb = np.arange(50e3,51.2e6,100e3)
Nb = len(fcomb)
kcomb = np.arange(1,2*Nb,2)
print (fcomb[0], fcomb[-1], Nb)
# true response
true_resp = np.exp(-(fcomb-30e6)**2/(2*25e6**2))
# our transmission code
code = np.exp(2*np.pi*1j*np.random.uniform(0,2*np.pi,Nb))
code_resp = code #* true_resp
dt = 4096/102.4e6  # time between frames 
phase_drift_per_ppm = fcomb*dt*(1/1e6)*2*np.pi
Nnotch = 16 #
alpha_to_pdrift = (Nnotch*phase_drift_per_ppm/kcomb)[0]




def produce_data(alpha = -0.5, dalpha_dt = +0.0, sA=1, sc=None, ssig=None, Nintg = 64, add_noise = True):
    istart = 0
    last_phase = np.zeros(Nb)
    Nblock = Nnotch*Nintg
    while True:
        iend = istart+Nblock
        tar = np.arange(istart, iend)*dt
        alpha_ar = -(alpha+dalpha_dt*tar) #silly signs
        ampl_ar = sA*np.exp(-(tar-sc)**2/(2*ssig**2)) if sc is not None else sA*np.ones_like(tar)
        phase = last_phase + np.cumsum(np.outer(alpha_ar,phase_drift_per_ppm),axis=0)
        #print(alpha_ar,phase_drift_per_ppm)
        #print (phase[Nnotch//2::Nnotch,0])
        rot = np.exp(-1j*phase)
        data = code_resp[None,:]*ampl_ar[:,None]*rot
        
        t = tar.mean()
        calpha = alpha_ar.mean()
        dB = np.log10(ampl_ar.mean())*20 ## power is square of amplitude
        if add_noise:
            data += np.random.normal(0,1.0/np.sqrt(Nnotch), data.shape)
        ## integrate here
        data = data.reshape(Nintg,Nnotch,data.shape[1]).mean(axis=1)
        
        yield (t, calpha, dB, data)
        istart += Nblock
        last_phase = phase[-1,:]


def use_refspec_data(filename, Nintg=64, add_noise=False, SNR_dB=0):
    
    d=np.fromfile(open(filename,'rb'),dtype=np.csingle)
    d = d.reshape((-1,2048)) 
    # and now take just 50, 150, etc
    d = d[:,1::4]*1e7
    istart = 0
    if add_noise:
        rms_signal = np.sqrt((np.abs(d)**2).mean())
        rms_noise = rms_signal/Nnotch*10**(-SNR_dB/20)
        #print (rms_noise)
    while True:
        iend = istart+Nintg
        if iend > d.shape[0]:
            break
        t = (istart+iend)/2*dt*Nnotch
        calpha = 0 
        data = d[istart:iend,:]
        if add_noise:
            
            data += np.random.normal(0,rms_noise, data.shape)
            
        yield (t,calpha, SNR_dB, data)
        istart = iend
        

def analyze_data (data_gen, tmax=30, countmax=None, Nintg = 64, alpha_start = 0, force_detect=False, maxdriftndx=None ):
    # Now we do the relatively 
    init_phase = np.ones(Nb, complex)
    detect = False
    pdrift = alpha_start*alpha_to_pdrift
    drift = []
    nullw = (-1)**np.arange(Nintg)
    Nblock = Nnotch*Nintg
    count = 0 
    for t, alpha, dB, data in data_gen:

        if t>tmax:
            break
        count += 1
        if countmax is not None:
            if count>countmax:
                break
        assert(data.shape[0]==Nintg)
        kar = np.outer(np.arange(Nintg+1),kcomb)
        
        if True:
        ## the actual statement
            phase_cor = np.exp(-1j*pdrift*kar)
        ## the "fast" statement
        else:
            tab = pdrift*kcomb
            phase_step = 1-(tab)**2/2 + 1j*tab
            phase_cor = np.array([phase_step**i for i in range(Nintg+1)])
        phase1 = init_phase*phase_cor
        init_phase = phase1[-1,:]
        data *= phase1[:-1,:]
        kar = kar[:-1,:]
        
        sum0 = data.sum(axis=0)
        sum0null = (data*nullw[:,None]).sum(axis=0)
        sum1 = (1j*kar*data).sum(axis=0)
        sum2 = (-kar**2*data).sum(axis=0)
        FD = np.real(sum1*np.conj(sum0))
        SD = np.real(sum2*np.conj(sum0)+sum1*np.conj(sum1))
        sig2 = np.abs(sum0**2)
        noise2 = np.abs(sum0null**2)
        SNRdB = np.log10(sig2.sum()/noise2.sum())*10
        #print (sum0[2],sum1[2],sum2[2],'sums')
        #print (FD[2], SD[2],'FDSD')
        yield t, alpha, dB, pdrift/alpha_to_pdrift, SNRdB, detect
        if maxdriftndx is not None:
            FD = FD[:maxdriftndx]
            SD = SD[:maxdriftndx]
        delta_drift = (FD.sum()/SD.sum())
        if force_detect:
            pdrift += delta_drift    
            print ('new pdrift = ', pdrift/alpha_to_pdrift)
        else:
            if np.abs(delta_drift)>0.05*alpha_to_pdrift:
                delta_drift = +0.05*alpha_to_pdrift
            pdrift = pdrift+delta_drift
            if np.abs(pdrift)>1.2*alpha_to_pdrift:
                pdrift = np.sign(pdrift)*1.2*alpha_to_pdrift*(-1)
       
        

    return 



def analyze_data_corr (data_gen, tmax=30, countmax=None, Nintg = 64, ofs_start = 0, force_detect=False,maxdriftndx=500):
    # Now we do the relatively 
    ofs = ofs_start
    nullw = (-1)**np.arange(len(kcomb))[:maxdriftndx]
    codeconj = np.conj(code)
    count =0 
    for t, alpha, dB, data in data_gen:
        if t>tmax:
            break
        count += 1
        if countmax is not None:
            if count>countmax:
                break
        assert(data.shape[0]==Nintg)
        tSNR2 = 0
        detect = 0 
        for d in data:
            SNRsave=0
            for dofs in [0]:#,-1e-4, 1e-4]:
                ds = (d*np.exp(-1j*kcomb*(ofs+dofs))*codeconj)[:maxdriftndx]
                sig = np.real(ds).sum()
                noise = np.real(ds*nullw).sum()
                
                SNR = np.abs(sig)/np.abs(noise)          
                if SNR>SNRsave:      
                    sum1 = (1j*kcomb[:maxdriftndx]*ds).sum()
                    sum2 = (-kcomb[:maxdriftndx]**2*ds).sum()
                    #FDsave = np.real(sum1*np.conj(sig))
                    #SDsave = np.real(sum2*np.conj(sig)+sum1*np.conj(sum1))
                    FDsave = np.real(sum1)
                    SDsave = np.real(sum2)
                    sigsave = sig
                    noisesave = noise
                    dofsave = dofs
                    SNRsave=SNR
            #print (sigsave, noisesave, ofs, dofsave, 'sig noise')
            if SDsave<0 and np.abs(FDsave/SDsave)<0.01:
                ofs += dofsave+(FDsave/SDsave)#/4
                tSNR2 += sigsave
                detect +=1
            else:
                ofs += (3*5*7*11/2048)*np.pi
                if ofs>np.pi/2:
                    ofs -= np.pi
        SNRdB = 10*np.log10(sigsave)
        yield t, alpha, dB, ofs, SNRdB, detect
