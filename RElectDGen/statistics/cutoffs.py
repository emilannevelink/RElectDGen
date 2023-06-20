import sys
import numpy as np
from scipy import stats

def calculate_CI(data,dataset_size_ratio=1):
    if isinstance(data,(int,float)):
        ndata = data
    elif isinstance(data,np.ndarray):
        ndata = len(data.flatten())
    elif isinstance(data,list):
        ndata = len(np.array(data).flatten())
    else:
        raise ValueError
    if dataset_size_ratio <= 0:
        print(dataset_size_ratio)
    if ndata <= 0:
        print('Ndata: ',ndata)
    CI = 1-2/ndata/dataset_size_ratio
    return CI

def get_statistics_cutoff(sampled_uncertainties,dist_dict,dataset_size_ratio=1):
    if dist_dict['name'] is None:
        return 0
    if len(dist_dict['data']) < 30:
        return np.mean(dist_dict['data'])
    CI = calculate_CI(sampled_uncertainties,dataset_size_ratio)
    CI = 1 if CI > 1 else CI
    dist = getattr(stats,dist_dict['name'])
    _, uncertainty_cutoff = dist.interval(CI,*dist_dict['args'])

    return uncertainty_cutoff

def get_max_dataset_ratio(target_cutoff,sampling_data,dist_dict):
    if 'cutoff' not in dist_dict:
        dist_dict['cutoff'] = get_statistics_cutoff(sampling_data,dist_dict)
    
    # func = lambda n: get_statistics_cutoff(sampling_data,dist_dict,dataset_size_ratio=(n)**2)-target_cutoff
    # out = optimize.fsolve(func,10)
    # max_dataset_ratio = int(out**2)


    # loss = lambda n: (get_statistics_cutoff(sampling_data,dist_dict,dataset_size_ratio=(n)**2)-target_cutoff)**2
    # res = optimize.minimize(loss,10)
    # print(res)
    # max_dataset_ratio = float(max([(res.x**2),1]))
    dist = getattr(stats,dist_dict['name'])
    mean = dist.stats(*dist_dict['args'],moments='m')
    max_dataset_ratio = (target_cutoff-mean)/(dist_dict['cutoff']-mean)

    print(max_dataset_ratio)
    return max_dataset_ratio

def get_max_cutoff(sampled_uncertainties, errors_dict, unc_dict, max_error=1.5):

    error_cutoff = max_error #min([2*errors_dict['cutoff'],max_error])
    # dist = getattr(stats,errors_dict['name'])
    # probability_density = dist.pdf(error_cutoff,*errors_dict['args'])
    
    max_dataset_ratio = get_max_dataset_ratio(error_cutoff,sampled_uncertainties,errors_dict)

    # max_cutoff = get_statistics_cutoff(sampled_uncertainties,unc_dict,max_dataset_ratio)
    dist = getattr(stats,unc_dict['name'])
    mean = dist.stats(*unc_dict['args'],moments='m')
    max_cutoff = max_dataset_ratio*(unc_dict['cutoff']-mean)+mean

    print(error_cutoff,max_cutoff)
    return max_cutoff

def get_base_cutoffs(dist_dict, sampling_data):
    # print(dist_dict)
    if not isinstance(dist_dict,dict):
        name, args, res = choose_distribution(dist_dict)
        dist_dict = {'data': dist_dict,'name':name,'args':args,'res': res}
    elif 'name' not in dist_dict:
        name, args, res = choose_distribution(dist_dict['data'])
        dist_dict['name'] = name
        dist_dict['args'] = args
        dist_dict['res'] = res
        
    cutoff = get_statistics_cutoff(sampling_data, dist_dict)
    AL_cutoff = get_statistics_cutoff(sampling_data,dist_dict,2)

    dist_dict['cutoff'] = cutoff
    dist_dict['AL_cutoff'] = AL_cutoff
    return dist_dict

def get_best_dict(train_dict,val_dict):
    best_dict = val_dict if val_dict['res'].pvalue>train_dict['res'].pvalue else train_dict
    return best_dict

def get_all_dists_cutoffs(
    sampling_data,
    train_error_dict,
    validation_error_dict,
    train_uncertainty_dict=None,
    validation_uncertainty_dict=None,
    sampling_type = 'uncertainty',
    max_error=1.5,
    **kwargs
    ):
    
    train_error_dict = get_base_cutoffs(train_error_dict,sampling_data)
    validation_error_dict = get_base_cutoffs(validation_error_dict,sampling_data)
    error_dict = validation_error_dict if validation_error_dict['res'].pvalue>train_error_dict['res'].pvalue else train_error_dict
    if sampling_type == 'uncertainty':
        train_uncertainty_dict = get_base_cutoffs(train_uncertainty_dict,sampling_data)
        validation_uncertainty_dict = get_base_cutoffs(validation_uncertainty_dict,sampling_data)
        uncertainty_dict = validation_uncertainty_dict if validation_uncertainty_dict['res'].pvalue>train_uncertainty_dict['res'].pvalue else train_uncertainty_dict
    else:
        uncertainty_dict = error_dict
    
    max_cutoff = get_max_cutoff(sampling_data,error_dict,uncertainty_dict,max_error)
    out_dict = {
        'train_error_dict': train_error_dict,
        'validation_error_dict': validation_error_dict,
        'cutoff': uncertainty_dict['cutoff'],
        'AL_cutoff': uncertainty_dict['AL_cutoff'],
        'max_cutoff': max_cutoff,
    }
    if sampling_type == 'uncertainty':
        out_dict['train_uncertainty_dict'] = train_uncertainty_dict
        out_dict['validation_uncertainty_dict'] = validation_uncertainty_dict
    
    return out_dict

def fix_unc_cutoff(error_dict, unc_dict):
    print(error_dict['max_cutoff'])
    print(error_dict['cutoff'])
    print(unc_dict['max_cutoff'])
    print(unc_dict['cutoff'])

    error_ratio = error_dict['max_cutoff']/error_dict['cutoff']
    unc_ratio = unc_dict['max_cutoff']/unc_dict['cutoff']

    unc_dict['max_cutoff'] = unc_dict['cutoff']*error_ratio if unc_ratio < error_ratio else unc_dict['max_cutoff']

    return unc_dict

def converge_args(vals,dist_name='lognorm',dataset_size_ratio=1):

    dist_arg_array = None
    dist = getattr(stats,dist_name)
    vals_flat = vals.flatten()
    initial_length = len(vals_flat)
    for i in range(3):
        vals_flat = truncate_extrema(vals_flat)
        if len(vals_flat) == 0:
            return vals_flat, None
    lower_bound = vals_flat.min()
    upper_bound = vals_flat.max()
    dbound = max((upper_bound-lower_bound)/len(vals_flat),0.001)

    pmax = 0
    truncated_length = len(vals_flat)
    if truncated_length < initial_length/10:
        return vals_flat, None
    for i in range(10):
        if len(vals_flat)<=truncated_length/3:
            vals_flat = vals.flatten()
            break
        if 'chi' in dist_name:
            dist_args = dist.fit(vals_flat,fdf=3,floc=0)
        elif 'gamma' == dist_name:
            dist_args = dist.fit(vals_flat,3)
        else:
            dist_args = dist.fit(vals_flat)
        res = stats.kstest(vals_flat,dist_name,args=dist_args)
        if dist_arg_array is None:
            dist_arg_array = np.empty((0,len(dist_args)))
        dist_arg_array = np.concatenate([dist_arg_array,[dist_args]],axis=0)
        CI = 1-2/len(vals_flat)/dataset_size_ratio
        try:
            imin, imax = dist.interval(CI,*dist_args)
        except Exception as e:
            pass
        
        # if 'gamma' not in dist_name:
        lower_bound = imin if imin > lower_bound else lower_bound+dbound
        upper_bound = imax #if imax < upper_bound else upper_bound-dbound

        try:
            mask = np.logical_and(vals_flat>lower_bound,vals_flat<upper_bound)
            if i<5:
                converged = False
            elif 'chi' in dist_name:
                converged = np.all(np.abs(dist_arg_array[-5:,2:].std(axis=0)/dist_arg_array[-5:,2:].mean(axis=0))<0.01)
            else:
                converged = np.all(np.abs(dist_arg_array[-5:].std(axis=0)/dist_arg_array[-5:].mean(axis=0))<0.01)
            # if sum(mask) == len(vals_flat) and False:
            #     # no values outside distribution
            #     break
            if converged:# and False:
                # converged
                best_args = dist_args
                break
            if res.pvalue > 0.05:
                pmax = res.pvalue
                best_args = dist_args
                break
            elif res.pvalue >= pmax:
                pmax = res.pvalue
                best_args = dist_args
        except Exception as e:
            print(e)
            print(mask)
            print(vals_flat)
            sys.exit()
        vals_flat = vals_flat[mask]
    
    return vals_flat, best_args

def truncate_by_probability(vals, dist, args, nbins=101):
    vals_flat = vals.flatten()
    vals, bins = np.histogram(vals,nbins,density=True)
    
    vals_keep = np.empty(0)
    for i, (v, bl, br) in enumerate(zip(vals,bins[:-1],bins[1:])):
        bin_center = np.mean([bl,br])
        probability = dist.pdf(bin_center,*args)

        keep_ratio = np.nan_to_num(probability/v,nan=1)
        between_inds = np.argwhere(np.logical_and(vals_flat>bl,vals_flat<br)).flatten()
        n_keep = int(len(between_inds)*keep_ratio)
        keep_inds = between_inds[np.random.permutation(len(between_inds))[:n_keep]]
        vals_keep = np.concatenate([vals_keep,vals_flat[keep_inds]])

    return vals_keep

class FakeFitResult():
    pvalue = -1

def choose_distribution(vals,pthreshold=0.05,truncate_decreasing=True):
    # print(len(vals))
    # pvalues = [
    #     res_lognormal.pvalue,
    #     res_chi.pvalue,
    #     res_gamma.pvalue,
    #     res_gengamma.pvalue,
    # ]
    
    try:
        chi_vals, chi_args = converge_args(vals,'maxwell')
        if chi_args is None or len(chi_vals)==0:
            res_chi = FakeFitResult()
        else:
            res_chi = stats.kstest(chi_vals,'maxwell',args=chi_args)
    except stats.FitError:
        res_chi = FakeFitResult()
    if res_chi.pvalue > pthreshold: # np.argmax(pvalues) in [1,2] and 
        return 'maxwell', chi_args, res_chi
    
    try:
        gamma_vals, gamma_args = converge_args(vals,'gamma')
        if gamma_args is None or len(gamma_vals)==0:
            res_gamma = FakeFitResult()
        else:
            res_gamma = stats.kstest(gamma_vals,'gamma',args=gamma_args)
    except stats.FitError:
        res_gamma = FakeFitResult()
    if res_gamma.pvalue > pthreshold: #np.argmax(pvalues) == 2 and 
        return 'gamma', gamma_args, res_gamma
    
    try:
        gengamma_vals, gengamma_args = converge_args(vals,'gengamma')
        if gengamma_args is None or len(gengamma_vals)==0:
            res_gengamma = FakeFitResult()
        else:
            res_gengamma = stats.kstest(gengamma_vals,'gengamma',args=gengamma_args)
    except stats.FitError:
        res_gengamma = FakeFitResult()
    if res_gengamma.pvalue > pthreshold: # np.argmax(pvalues) == 3 and 
        return 'gengamma', gengamma_args, res_gengamma
    
    if truncate_decreasing:
        vals_truncated = truncat_mode_decreasing(vals)
        name, args, res_truncated = choose_distribution(vals_truncated,pthreshold,truncate_decreasing=False)
    else:
        name = args = None
        
    try:
        lognorm_vals, lognorm_args = converge_args(vals,'lognorm')
        if lognorm_args is None or len(lognorm_vals)==0:
            res_lognormal = FakeFitResult()
        else:
            res_lognormal = stats.kstest(lognorm_vals,'lognorm',args=lognorm_args)
    except stats.FitError:
        res_lognormal = FakeFitResult()
    if res_lognormal.pvalue > pthreshold: # np.argmax(pvalues) == 0 and 
        return 'lognorm', lognorm_args, res_lognormal
    
    pvalues = [res_chi.pvalue,res_gamma.pvalue,res_gengamma.pvalue,0]# remove lognorm: res_lognormal.pvalue]
    if truncate_decreasing:
        pvalues += [res_truncated.pvalue]
    if np.argmax(pvalues) == 0:
        name = 'maxwell'
        args = chi_args
        res = res_chi
    elif np.argmax(pvalues) == 1:
        name = 'gamma'
        args = gamma_args
        res = res_gamma
    elif np.argmax(pvalues) == 2:
        name = 'gengamma'
        args = gengamma_args
        res = res_gengamma
    elif np.argmax(pvalues) == 3:
        name = 'lognorm'
        args = lognorm_args
        res = res_lognormal
    elif np.argmax(pvalues) == 4:
        res = res_truncated

    return name, args, res
    
def truncat_mode_decreasing(vals,nbins=101):
    vals_flat = truncate_extrema(vals,nbins = nbins)
    vals_flat = vals.flatten()
    hist, bins = np.histogram(vals_flat,bins=nbins)
    max_ind = np.argmax(hist)
    max_cutoff = bins[max_ind]
    mask = vals_flat>max_cutoff
    vals_truncated = vals_flat[mask]
    return vals_truncated

def truncate_extrema(vals,nbins = 101):
    vals_flat = vals.flatten()
    lower_bound = vals_flat.min()
    upper_bound = vals_flat.max()
    if len(vals_flat)<=nbins:
        return vals_flat
    hist_vals,bins = np.histogram(vals_flat,nbins)
    try:
        if np.argwhere(hist_vals>2).max() != len(hist_vals)-1:
            upper_bound = bins[np.argwhere(hist_vals>2).max()+1]
            mask = np.logical_and(vals_flat>lower_bound,vals_flat<upper_bound)
            vals_flat = vals_flat[mask]
    except:
        print(hist_vals)
    hist_vals, bins = np.histogram(vals_flat,nbins)
    if np.argmax(hist_vals) == 0:
        lower_bound = bins[1]
        mask = np.logical_and(vals_flat>lower_bound,vals_flat<upper_bound)
        vals_flat = vals_flat[mask]
    
    return vals_flat