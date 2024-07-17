import numpy as np

class Comb:

    def __init__ (self,Nstart=181, Nend = 983, response = lambda x:1, noise = lambda x:1, pilot_idx=None, pilot_boost=3):
        # Above is 401 combs starting at 9.05
        # response and noise are functions of frequency in Hz

        delta_freq = 100e3
        base_freq = 50e3

        self.kcomb = np.arange(Nstart,Nend,2,dtype=np.float64)
        self.fcomb = self.kcomb*base_freq # in Hz
        self.Nb = (Nend-Nstart)//2
 
        self.true_resp = response(self.fcomb)
        self.noise_level = noise(self.fcomb)

        if pilot_idx is not None:
            Npilots = len(pilot_idx)
            Nothers = self.Nb-Npilots
            non_pilot_boost = np.sqrt((self.Nb-pilot_boost**2*Npilots)/Nothers)
            if np.isnan(non_pilot_boost):
                print ("pilot_boost too gigantic")
                assert(False)       
            print ('pilot_boost', pilot_boost)
            print ('non_pilot_boost', non_pilot_boost)
            for i in range(len(self.true_resp)): 
                if i in pilot_idx:
                    self.true_resp[i] *= pilot_boost
                else:
                    self.true_resp[i] *= non_pilot_boost
        self.pilot_idx = pilot_idx

        # our transmission code
        self.code = np.exp(2*np.pi*1j*np.random.uniform(0,2*np.pi,self.Nb))
        self.code_resp = self.code * self.true_resp
        self.dt = 4096/102.4e6  # time between frames in s
        self.phase_drift_per_ppm = self.fcomb*self.dt*(1/1e6)*2*np.pi
        self.alpha_to_pdrift0 = (base_freq*self.dt*(1/1e6)*2*np.pi)
        


        #self.Nnotch = 16
        #alpha_to_pdrift = (self.Nnotch*phase_drift_per_ppm/kcomb)[0]


class Calibrator:

    def __init__ (self, comb, alpha = 0.0, dalpha_dt=0.0, Nnotch = 16, Nintg = 64, add_noise = False, 
                  ndx_start = 0, ndx_end = 512,
                  sc=None, ssig=None, max_shift = 0.05, max_alpha = 1.2):
        # Nnotch is the primary blind integration
        # Nintg is what is Navg2 in firmware
        # sc and ssig are the center and width of the gaussian amplitude modulation (if not None) in units of secods
        self.comb=comb
        self.dt = comb.dt
        self.alpha = alpha
        self.dalpha_dt = dalpha_dt
        self.Nnotch = Nnotch
        self.Nintg = Nintg
        self.add_noise = add_noise
        self.istart = 0
        self.last_phase = np.zeros(self.comb.Nb)
        self.Nblock = Nnotch*Nintg
        self.sc = sc
        self.ssig = ssig
        self.max_shift = max_shift
        self.max_alpha = max_alpha
        self.ndx_start = ndx_start
        self.ndx_end = ndx_end

    def produce_data_block(self):
        iend = self.istart+self.Nblock
        tar = np.arange(self.istart, iend)*self.dt
        alpha_ar = -(self.alpha+self.dalpha_dt*tar) #silly signs
        ampl_ar = np.exp(-(tar-self.sc)**2/(2*self.ssig**2)) if self.sc is not None else np.ones_like(tar)
        phase = self.last_phase + np.cumsum(np.outer(alpha_ar,self.comb.phase_drift_per_ppm),axis=0)
        rot = np.exp(-1j*phase)
        data = self.comb.code_resp[None,:]*ampl_ar[:,None]*rot
        self.current_t = tar.mean()
        self.current_alpha = alpha_ar.mean()
        self.current_dB = np.log10(ampl_ar.mean())*20 ## power is square of amplitude
        if self.add_noise:
            noise_real = np.random.normal(0,1.0/np.sqrt(2), data.shape)
            noise_imag = np.random.normal(0,1.0/np.sqrt(2), data.shape)
            noise = noise_real+1j*noise_imag
            data += noise*self.comb.noise_level[None,:]
        ## integrate here
        self.istart += self.Nblock
        self.last_phase = phase[-1,:]
        return data

    def analyze_data_incoherent(self, tmax=30):
        wide_bin = []
        narrow_bin = []
        while True:
            data = self.produce_data_block()
            t = self.current_t        
            if t>tmax:
                break
            wide_bin.append(np.abs(data**2).mean(axis=0))
            data = data.reshape(self.Nintg,self.Nnotch,data.shape[1]).mean(axis=1)
            narrow_bin.append(np.abs(data**2).mean(axis=0))
        wide_bin = np.array(wide_bin).mean(axis=0)
        narrow_bin = np.array(narrow_bin).mean(axis=0)*self.Nnotch
        return wide_bin, narrow_bin
    
            
        

    def analyze_data (self, tmax=30, countmax=None, alpha_start = 0, force_detect=False):
        # Now we do the relatively 
        init_phase = np.ones(self.comb.Nb, complex)
        detect = False
        alpha_to_pdrift = self.Nnotch*self.comb.alpha_to_pdrift0
        pdrift = alpha_start*alpha_to_pdrift
        drift = []
        nullw = (-1)**np.arange(self.Nintg)
        Nblock = self.Nnotch*self.Nintg
        count = 0 
        t_ret = []
        alpha_ret = []        
        alphadet_ret = []
        SNRdB_ret = []
        SNRdBdet_ret = []
        detect_ret = []
        data_ret = []
        FD_check = [] #db
        SD_check = [] #db
        i = 1 #db
        while True:
            data = self.produce_data_block()
            # Average over Nnotch
            data = data.reshape(self.Nintg,self.Nnotch,data.shape[1]).mean(axis=1)

            alpha = self.current_alpha
            dB = self.current_dB
            t = self.current_t        
            if t>tmax:
                break
            count += 1
            if countmax is not None:
                if count>countmax:
                    break
            assert(data.shape[0]==self.Nintg)
            kar = np.outer(np.arange(self.Nintg+1),self.comb.kcomb)
            phase_cor = np.exp(-1j*pdrift*kar)
            phase1 = init_phase*phase_cor
            init_phase = phase1[-1,:]
            data *= phase1[:-1,:]
            kar = kar[:-1,:]            
            sum0 = data.sum(axis=0)
            sum0null = (data*nullw[:,None]).sum(axis=0)
            sum1 = (1j*kar*data).sum(axis=0)
            sum2 = (-kar**2*data).sum(axis=0)

            # it begins
            FD = np.real(sum1*np.conj(sum0)) #problematic
            SD = np.real(sum2*np.conj(sum0)+sum1*np.conj(sum1)) #less problematic
            sig2 = np.abs(sum0**2)
            noise2 = np.abs(sum0null**2)
            SNR = sig2.sum()/noise2.sum() #you're doing great
            SNRdB = np.log10(SNR)*10
            if self.comb.pilot_idx is None:
                FD_sum = sum(FD)
                SD_sum = sum(SD)
            else:
                FD_sum = sum(FD[i] for i in self.comb.pilot_idx)
                SD_sum = sum(SD[i] for i in self.comb.pilot_idx)
            FD_check.append(FD_sum) #db
            SD_check.append(SD_sum) #db
            delta_drift = (FD_sum / SD_sum)
            if force_detect:
                pdrift += delta_drift
                detect = True
                # print('new pdrift = ', pdrift / alpha_to_pdrift)
            else:
                # Ensure SD_sum is scalar for logical comparison
                if (np.abs(delta_drift) < self.max_shift * alpha_to_pdrift) and (SD_sum < 0):
                    # sticky detect, need SNR first time, then ok
                    if not detect:
                        if SNR>1.0:
                            detect=True
                else:
                    detect=False
                    delta_drift = self.max_shift*alpha_to_pdrift
                pdrift = pdrift+delta_drift
                if np.abs(pdrift)>self.max_alpha*alpha_to_pdrift:
                    pdrift = np.sign(pdrift)*self.max_alpha*alpha_to_pdrift*(-1)
            print(FD_sum, SD_sum, detect) #db
            i += 1 #db
            t_ret.append(t)
            alpha_ret.append(alpha)
            alphadet_ret.append(pdrift/alpha_to_pdrift)
            SNRdB_ret.append(dB)
            SNRdBdet_ret.append(SNRdB)
            detect_ret.append(detect)
            data_ret.append(sum0)
    
        print("FD mean:",np.mean(FD_check), "SD mean:", np.mean(SD_check)) #db
        print(self.max_shift * alpha_to_pdrift) #db
        return {'t':np.array(t_ret), 
                'alpha':np.array(alpha_ret), 
                'alphadet':np.array(alphadet_ret), 
                'SNRdB':np.array(SNRdB_ret), 
                'SNRdBdet':np.array(SNRdBdet_ret), 
                'detect':np.array(detect_ret), 
                'data':np.array(data_ret),
                'FD':np.array(FD_check),
                'SD':np.array(SD_check)}
    
                



#
# Code below does it with correlation -- less efficient since we don't have resources for a full search
#

# def analyze_data_corr (data_gen, tmax=30, countmax=None, Nintg = 64, ofs_start = 0, force_detect=False,maxdriftndx=500):
#     # Now we do the relatively 
#     ofs = ofs_start
#     nullw = (-1)**np.arange(len(kcomb))[:maxdriftndx]
#     codeconj = np.conj(code)
#     count =0 
#     for t, alpha, dB, data in data_gen:
#         if t>tmax:
#             break
#         count += 1
#         if countmax is not None:
#             if count>countmax:
#                 break
#         assert(data.shape[0]==Nintg)
#         tSNR2 = 0
#         detect = 0 
#         for d in data:
#             SNRsave=0
#             for dofs in [0]:#,-1e-4, 1e-4]:
#                 ds = (d*np.exp(-1j*kcomb*(ofs+dofs))*codeconj)[:maxdriftndx]
#                 sig = np.real(ds).sum()
#                 noise = np.real(ds*nullw).sum()
                
#                 SNR = np.abs(sig)/np.abs(noise)          
#                 if SNR>SNRsave:      
#                     sum1 = (1j*kcomb[:maxdriftndx]*ds).sum()
#                     sum2 = (-kcomb[:maxdriftndx]**2*ds).sum()
#                     #FDsave = np.real(sum1*np.conj(sig))
#                     #SDsave = np.real(sum2*np.conj(sig)+sum1*np.conj(sum1))
#                     FDsave = np.real(sum1)
#                     SDsave = np.real(sum2)
#                     sigsave = sig
#                     noisesave = noise
#                     dofsave = dofs
#                     SNRsave=SNR
#             #print (sigsave, noisesave, ofs, dofsave, 'sig noise')
#             if SDsave<0 and np.abs(FDsave/SDsave)<0.01:
#                 ofs += dofsave+(FDsave/SDsave)#/4
#                 tSNR2 += sigsave
#                 detect +=1
#             else:
#                 ofs += (3*5*7*11/2048)*np.pi
#                 if ofs>np.pi/2:
#                     ofs -= np.pi
#         SNRdB = 10*np.log10(sigsave)
#         yield t, alpha, dB, ofs, SNRdB, detect


# def use_refspec_data(filename, Nintg=64, add_noise=False, SNR_dB=0):
    
#     d=np.fromfile(open(filename,'rb'),dtype=np.csingle)
#     d = d.reshape((-1,2048)) 
#     # and now take just 50, 150, etc
#     d = d[:,1::4]*1e7
#     istart = 0
#     if add_noise:
#         rms_signal = np.sqrt((np.abs(d)**2).mean())
#         rms_noise = rms_signal/Nnotch*10**(-SNR_dB/20)
#         #print (rms_noise)
#     while True:
#         iend = istart+Nintg
#         if iend > d.shape[0]:
#             break
#         t = (istart+iend)/2*dt*Nnotch
#         calpha = 0 
#         data = d[istart:iend,:]
#         if add_noise:
            
#             data += np.random.normal(0,rms_noise, data.shape)
            
#         yield (t,calpha, SNR_dB, data)
#         istart = iend
