import numpy as np

class Comb:

    def __init__ (self,kar, response, noise, weight_bits = 8):
        # Above is 401 combs starting at 9.05
        # response and noise are functions of frequency in Hz

        delta_freq = 100e3
        base_freq = 50e3

        self.kcomb = kar
        self.fcomb = self.kcomb*base_freq # in Hz
        self.Nb = len(self.fcomb)
 
        self.true_resp = response
        self.noise_level = noise

        # our transmission code
        self.code = np.exp(2*np.pi*1j*np.random.uniform(0,2*np.pi,self.Nb))
        self.code_resp = self.code * self.true_resp
        self.dt = 4096/102.4e6  # time between frames in s
        self.phase_drift_per_ppm = self.fcomb*self.dt*(1/1e6)*2*np.pi
        self.alpha_to_pdrift0 = (base_freq*self.dt*(1/1e6)*2*np.pi)
        self.weights = response**2/noise**4
        self.weights /= self.weights.max()
        if weight_bits is not None:
            self.weights = np.round(self.weights*2**weight_bits)#/2**weight_bits
        print ('Non zero weights', np.sum(self.weights>0))   


class Calibrator:

    def __init__ (self, comb, alpha = 0.0, dalpha_dt=0.0, Nnotch = 16, Nintg = 256, Nstage3=8, add_noise = False, 
                  sc=None, ssig=None, sA=1.0, max_shift = 0.05, max_alpha = 1.2, Nsettle=3, SNRon = 3, SNRoff=2, delta_drift_cor_A = 2, delta_drift_cor_B = 20, export=None):
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
        self.last_phase = np.zeros(self.comb.Nb)
        self.Nblock = Nnotch*Nintg
        self.sc = sc
        self.ssig = ssig
        self.sA = sA
        self.max_shift = max_shift
        self.max_alpha = max_alpha
        self.kar = np.outer(np.arange(self.Nintg+1),self.comb.kcomb)
        self.istart = 0
        self.Nstage3 = Nstage3  
        self.delta_drift_cor_A = delta_drift_cor_A
        self.delta_drift_cor_B = delta_drift_cor_B
        self.SNRon = SNRon
        self.SNRoff = SNRoff
        self.Nsettle = Nsettle
        self.export = (export is not None)
        if self.export:
            self.export = True
            self.export_input = open(export+".input",'wb')
            self.export_noise = open(export+".noise",'wb')
            self.export_output = open(export+".output",'w')
            weights = np.zeros(512)
            Nw = len(self.comb.weights)
            weights[90:90+Nw] = self.comb.weights
            fw = open(export+'.weights','w')
            for i,w in enumerate(weights):
                fw.write (f"{i} {0.050+i*0.100:5.4f} {int(w)}\n")
            fw.close()

        
    def write_input(self, data):
        
        Nd = data.shape[1]
        for line in data:
            noise_real = np.random.normal(0,1.0/np.sqrt(2*self.Nnotch), 2048)
            noise_imag = np.random.normal(0,1.0/np.sqrt(2*self.Nnotch), 2048)
            noise = noise_real+1j*noise_imag
            self.export_noise.write(noise.astype(np.complex64).tobytes())

            dataf = np.zeros(2048, complex)  ## in principle, this could be noise
            dataf[361:361+4*Nd:4] = line
            self.export_input.write(dataf.astype(np.complex64).tobytes()) # this already contains the noise

        

    def produce_data_block(self):
        iend = self.istart+self.Nblock
        tar = np.arange(self.istart, iend)*self.dt
        alpha_ar = -(self.alpha+self.dalpha_dt*tar) #silly signs
        ampl_ar = self.sA*np.exp(-(tar-self.sc)**2/(2*self.ssig**2)) if self.sc is not None else np.ones_like(tar)
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
        return data, rot

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
    
            
        
    def stage3_add (self, sum0, detect):
        if detect:
            prod = self.sum3*np.conj(sum0)
            print ("PROD",self.sum3[0], np.conj(sum0)[0])
            if np.any(np.isnan(prod)):
                print ("WTF")
                print (self.sum3)
                print (sum0)
                stop()
            sum1 = (1j*self.comb.kcomb*prod).sum()
            sum2 = (-self.comb.kcomb**2*prod).sum()
            FD = np.real(sum1) #problematic
            SD = np.real(sum2)
            ofs = FD/SD if (SD!=0) else 0
            print ("FDSD",FD,SD)
            if (self.count3<self.Nstage3):
                phase = self.comb.kcomb*ofs
                self.sum3 = self.sum3*np.exp(-1j*phase)+sum0
                print ("And now:", self.sum3[0], np.exp(-1j*phase)[0], sum0[0])
                self.debug.append((self.sum3,sum0))
                self.count3 += 1
        else:
            # we do this elsewhere now
            #    self.sum3 = np.zeros(self.comb.Nb, complex)
            #    self.count3 = 0
            ofs = 0
        return detect, ofs


    def analyze_data (self, tmax=30, countmax=None, alpha_start = 0, force_detect=0):
        # Now we do the relatively 
        init_phase = np.ones(self.comb.Nb, complex)
        
        detect = 0
        alpha_to_pdrift = self.Nnotch*self.comb.alpha_to_pdrift0
        pdrift = alpha_start*alpha_to_pdrift
        count = 0 
        t_ret = []
        alpha_ret = []        
        alphadet_ret = []
        SNRdB_ret = []
        SNRdBdet_ret = []
        SNR2_ret = []
        SNR2alt_ret = []
        detect_ret = []
        detect3_ret = []
        ofs3_ret = []
        stage3_ret = []
        stage3_Nacc = []
        stage3_time = []
        data_ret = []
        FD_check = [] #db
        SD_check = [] #db
        #phase_in_ret = []
        #phase_fit_ret = []
        self.sum3 = np.zeros(self.comb.Nb, complex)
        self.debug = []
        self.count3 = 0
        SNRbase = 1 #np.sum(self.comb.weights>0)
        i = 1 #db
        while True:
            data,phase_in = self.produce_data_block()
            # Average over Nnotch
            data= data.reshape(self.Nintg,self.Nnotch,data.shape[1]).mean(axis=1)
            # we output Nnotch output
            if self.export:
                self.write_input(data)
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
            phase_cor = np.exp(-1j*pdrift*self.kar)
            #if not detect:
            #    init_phase = np.ones(self.comb.Nb, complex)
            phase1 = init_phase*phase_cor

            init_phase = phase1[-1,:]
            data *= phase1[:-1,:]
            #phase_in_ret.append (phase_in[:,0])
            #phase_fit_ret.append (phase1[:-1,0])

            kar = self.kar[:-1,:]            
            sum0 = data.sum(axis=0)
    
            sum0null = (np.abs((data[::2]-data[1::2])**2)).sum(axis=0)

            sum1 = (1j*kar*data).sum(axis=0)
            sum2 = (-kar**2*data).sum(axis=0)

            # it begins
            FD = np.real(sum1*np.conj(sum0)) #problematic
            SD = np.real(sum2*np.conj(sum0)+sum1*np.conj(sum1)) #less problematic



            SNR2alt  = np.sum(np.abs(sum0**2)/sum0null*(self.comb.weights>0))
            SNRdB = np.log10(SNR2alt-SNRbase)*10
            
            SNR2 = np.sum(np.abs(sum0**2)*self.comb.weights)/np.sum(sum0null*self.comb.weights)

            FD_sum = np.sum(FD*self.comb.weights)
            SD_sum = np.sum(SD*self.comb.weights)

            FD_check.append(FD_sum) #db
            SD_check.append(SD_sum) #db
            delta_drift = (FD_sum / SD_sum)
            if force_detect:
                if force_detect==1:
                    detect = 1
                elif force_detect==2:
                    detect = 0
            else:
                if detect>0:
                    if SNR2<self.SNRoff:
                        detect = 0
                    else:
                        detect = detect+1
                else:
                    if (SNR2>self.SNRon) and (SD_sum<0):
                        detect = 1
            if detect:
                if self.delta_drift_cor_A is not None:
                    aha = (SNR2-self.delta_drift_cor_A)/self.delta_drift_cor_B 
                    aha = max(0,min(aha,2))
                    cor = aha-aha**2/4 if aha<2 else 1
                    delta_drift *= cor
                if delta_drift>self.max_shift*alpha_to_pdrift:
                    delta_drift = self.max_shift*alpha_to_pdrift
                if delta_drift<-self.max_shift*alpha_to_pdrift:
                    delta_drift = -self.max_shift*alpha_to_pdrift
                
                #delta_drift = (-0.3*alpha_to_pdrift-pdrift)
            else:
                aha = 0   
                cor = 0 
                delta_drift = self.max_shift*alpha_to_pdrift

            
            print ("ADA:", sum0[0]*1e6)
            detect3, ofs3 = self.stage3_add(sum0,detect>self.Nsettle)
            if (self.count3==self.Nstage3) or (self.count3>0 and not detect):
                stage3_ret.append(self.sum3)
                stage3_Nacc.append(self.count3)
                stage3_time.append(t)
                self.sum3 = np.zeros(self.comb.Nb, complex)
                self.count3 = 0
                ofs = 0


            if self.export:
                self.export_output.write(f"{t:6.2} {alpha:5.2} {pdrift/alpha_to_pdrift:5.4} {SNR2:5.4}  {detect} {detect3} {ofs3} {delta_drift} {aha} {cor}\n")

            detect3_ret.append(detect3)
            ofs3_ret.append(ofs3)

            pdrift = pdrift+delta_drift
            if np.abs(pdrift)>self.max_alpha*alpha_to_pdrift:
                pdrift = np.sign(pdrift)*self.max_alpha*alpha_to_pdrift*(-1)
            i += 1 #db
            t_ret.append(t)
            alpha_ret.append(alpha)
            alphadet_ret.append(pdrift/alpha_to_pdrift)
            SNRdB_ret.append(dB)
            SNRdBdet_ret.append(SNRdB)
            SNR2_ret.append(SNR2)
            SNR2alt_ret.append(SNR2alt)
            detect_ret.append(detect)
            data_ret.append(sum0)
    
        
        print(self.max_shift * alpha_to_pdrift) #db
        if self.export:
            self.export_input.close()
            self.export_noise.close()
            self.export_output.close()

        self.results =  {'t':np.array(t_ret), 
                'alpha':np.array(alpha_ret), 
                'alphadet':np.array(alphadet_ret), 
                'SNRdB':np.array(SNRdB_ret), 
                'SNRdBdet':np.array(SNRdBdet_ret), 
                'SNR2':np.array(SNR2_ret),
                'SNR2alt':np.array(SNR2alt_ret),
                'detect':np.array(detect_ret), 
                'detect3':np.array(detect3_ret),
                'ofs3':np.array(ofs3_ret),
                'stage3':np.array(stage3_ret),
                'stage3_time':np.array(stage3_time),
                'stage3_Nacc':np.array(stage3_Nacc),
                'data':np.array(data_ret),
                'FD':np.array(FD_check),
                'SD':np.array(SD_check)}
                #'phase_in':np.hstack(phase_in_ret),
                #'phase_fit':np.hstack(phase_fit_ret)}
        return self.results
    
    def cross_correlate (self):
        assert(hasattr(self,'results'))
        Nfft = 500000
        self.results['stage3_phased'] = []
        self.results['stage3_offsets'] = []
        for l in self.results["stage3"]:
            cross= l*np.conj(self.comb.code_resp)
            fi = np.zeros(Nfft,complex)
            fi[self.comb.kcomb] = cross
            xi = np.real(np.fft.fft(fi))
            phi = xi.argmax()*2*np.pi/len(xi)
            phased_l = l*np.exp(-1j*phi*self.comb.kcomb)*np.conj(self.comb.code)/self.comb.true_resp/(self.Nintg*self.Nstage3)
            self.results['stage3_phased'].append(phased_l)
            self.results['stage3_offsets'].append(phi)
        self.results['stage3_phased'] = np.array(self.results['stage3_phased'])
        self.results['stage3_offsets'] = np.array(self.results['stage3_offsets'])
     


    
    
                



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
