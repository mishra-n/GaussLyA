from transforms import *


def nfgpa_param_check(data):
    F = tf.placeholder(tf.float64, shape=[None])
    #many of the outputs here are for debugging purposes (for example each transform also outputs the input variables)

    #reverse fgpa

    FGPA_params = FGPA_Layer('1_', .3, 1.6, 1.5, constant=True)

    nfgpa, A, alpha, sigma = NFGPA(F, FGPA_params)

    nfgpa_std = tf.math.reduce_std(nfgpa)
    nfgpa_mean = tf.math.reduce_mean(nfgpa)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        nfgpa_std_temp, nfgpa_mean_temp = sess.run([nfgpa_std, nfgpa_mean], {F:data})

        nfgpa_std = nfgpa_std_temp
        nfgpa_mean = nfgpa_mean_temp
        
    return nfgpa_std, nfgpa_mean

def optim(F, rand, eta_start, eps_start, eta2_start, eps2_start, mean_start, hess_start, beta_const):
    #many of the outputs here are for debugging purposes (for example each transform also outputs the input variables)
    recursion = 1
    #reverse fgpa

    FGPA_params = FGPA_Layer('1_', .3, 1.6, 1.5, constant=True)

    nfgpa, A, alpha, sigma = NFGPA(F, FGPA_params)

    #use this to transform the data after opt
    YJ_params = YJ_Layer(layer='1_', eta_input=eta_start, eps_input=eps_start, beta_input=beta_const, mean_input=mean_start, eta_constraint=[-3, 3-1e-50], eps_constraint=[-1+1e-50, 1-1e-50], beta_constraint=[-5, 5], vary_mean=True, vary_beta=False)
    YJ_params2 = YJ_Layer(layer='2_', eta_input=eta2_start, eps_input=eps2_start, beta_input=np.array([1.0]).astype(np.float64), mean_input=0.0, eta_constraint=[-3, 3-1e-50], eps_constraint=[-1+1e-50, 1-1e-50], beta_constraint=[-5, 5], vary_mean=False, vary_beta=False)

    r, eta, eps, beta, mean = YJ(nfgpa, YJ_params, recursion_num=recursion)
    
    r2, eta2, eps2, beta2, mean2 = YJ(r, YJ_params2, recursion_num=recursion)

    #likelihood
    likilihood_params = log_likilihood_Layer('1_', input_hess=hess_start, vary_hess=True)
    loss, outline, hess, r_g = log_likilihood(F, r2, likilihood_params)


    rand = tf.placeholder(tf.float64, shape=[None])

    rand_JY = JY(rand, YJ_params2, recursion_num=recursion)
    rand2_JY = JY(rand_JY, YJ_params, recursion_num=recursion)

    fgpa = FGPA(rand2_JY, FGPA_params)

    return A, alpha, sigma, nfgpa, eta, eps, beta, mean, r, eta2, eps2, beta2, r2, hess, outline, loss, fgpa

def optim3(F, rand, eta_start, eps_start, eta2_start, eps2_start, eta3_start, eps3_start,  mean_start, hess_start, beta_const):
    #many of the outputs here are for debugging purposes (for example each transform also outputs the input variables)
    recursion = 1
    #reverse fgpa

    FGPA_params = FGPA_Layer('1_', .3, 1.6, 1.5, constant=True)

    nfgpa, A, alpha, sigma = NFGPA(F, FGPA_params)

    #use this to transform the data after opt
    YJ_params = YJ_Layer(layer='1_', eta_input=eta_start, eps_input=eps_start, beta_input=beta_const, mean_input=mean_start, eta_constraint=[-3, 3-1e-50], eps_constraint=[-1+1e-50, 1-1e-50], beta_constraint=[-5, 5], vary_mean=True, vary_beta=False)
    YJ_params2 = YJ_Layer(layer='2_', eta_input=eta2_start, eps_input=eps2_start, beta_input=np.array([1.0]).astype(np.float64), mean_input=0.0, eta_constraint=[-3, 3-1e-50], eps_constraint=[-1+1e-50, 1-1e-50], beta_constraint=[-5, 5], vary_mean=False, vary_beta=False)
    YJ_params3 = YJ_Layer(layer='3_', eta_input=eta3_start, eps_input=eps3_start, beta_input=np.array([1.0]).astype(np.float64), mean_input=0.0, eta_constraint=[-3, 3-1e-50], eps_constraint=[-1+1e-50, 1-1e-50], beta_constraint=[-5, 5], vary_mean=False, vary_beta=False)
    
    r, eta, eps, beta, mean = YJ(nfgpa, YJ_params, recursion_num=recursion)
    
    r2, eta2, eps2, beta2, mean2 = YJ(r, YJ_params2, recursion_num=recursion)
    
    r3, eta3, eps3, beta3, mean3 = YJ(r2, YJ_params3, recursion_num=recursion)
    
    #likelihood
    likilihood_params = log_likilihood_Layer('1_', input_hess=hess_start, vary_hess=True)
    loss, outline, hess, r_g = log_likilihood(F, r3, likilihood_params)


    rand = tf.placeholder(tf.float64, shape=[None])

    rand_JY = JY(rand, YJ_params3, recursion_num=recursion)
    rand2_JY = JY(rand_JY, YJ_params2, recursion_num=recursion)
    rand3_JY = JY(rand2_JY, YJ_params, recursion_num=recursion)
    
    fgpa = FGPA(rand3_JY, FGPA_params)

    return A, alpha, sigma, nfgpa, eta, eps, beta, mean, r, eta2, eps2, beta2, r2, eta3, eps3, beta3, r3, hess, outline, loss, fgpa





def passthru(F, nfgpa_std, nfgpa_mean):
    # when running function in session pass through a start value for the following values
    #F:data_s8,
    #eta:float(value),
    #eps:float(value),
    #eta2:float(value),
    #eps2:float(value),
    #mean:float(value),
    #hess:float(value),
    #mean2:0.0                                                                                                       
    
    recursion = 1
    #reverse fgpa

    FGPA_params = FGPA_Layer('1_', .3, 1.6, 1.5, constant=True)

    nfgpa, A, alpha, sigma = NFGPA(F, FGPA_params)


    #use this to transform the data after opt
    YJ_params = YJ_PassThru_Layer(layer='1_', beta_input=nfgpa_std)
    YJ_params2 = YJ_PassThru_Layer(layer='2_', beta_input=np.array([1.0]).astype(np.float64))

    r, eta, eps, beta, mean = YJ(nfgpa, YJ_params, recursion_num=recursion)
    r2, eta2, eps2, beta2, mean2 = YJ(r, YJ_params2, recursion_num=recursion)

    #likelihood
    likilihood_params = log_likilihood_PassThru_Layer(layer='1_')
    loss, outline, hess, r_g = log_likilihood(F, r2, likilihood_params)


    rand = tf.placeholder(tf.float64, shape=[None])

    rand_JY = JY(rand, YJ_params2, recursion_num=recursion)
    rand2_JY = JY(rand_JY, YJ_params, recursion_num=recursion)

    fgpa = FGPA(rand2_JY, FGPA_params)
    
    return A, alpha, sigma, nfgpa, eta, eps, beta, mean, r, eta2, eps2, beta2, mean2, r2, hess, outline, loss, fgpa

def passthru3(F, nfgpa_std, nfgpa_mean):
    # when running function in session pass through a start value for the following values
    #F:data_s8,
    #eta:float(value),
    #eps:float(value),
    #eta2:float(value),
    #eps2:float(value),
    #mean:float(value),
    #hess:float(value),
    #mean2:0.0                                                                                                       
    
    recursion = 1
    #reverse fgpa

    FGPA_params = FGPA_Layer('1_', .3, 1.6, 1.5, constant=True)

    nfgpa, A, alpha, sigma = NFGPA(F, FGPA_params)


    #use this to transform the data after opt
    YJ_params = YJ_PassThru_Layer(layer='1_', beta_input=nfgpa_std)
    YJ_params2 = YJ_PassThru_Layer(layer='2_', beta_input=np.array([1.0]).astype(np.float64))
    YJ_params3 = YJ_PassThru_Layer(layer='3_', beta_input=np.array([1.0]).astype(np.float64))

    r, eta, eps, beta, mean = YJ(nfgpa, YJ_params, recursion_num=recursion)
    r2, eta2, eps2, beta2, mean2 = YJ(r, YJ_params2, recursion_num=recursion)
    r3, eta3, eps3, beta3, mean3 = YJ(r2, YJ_params3, recursion_num=recursion)

    #likelihood
    likilihood_params = log_likilihood_PassThru_Layer(layer='1_')
    loss, outline, hess, r_g = log_likilihood(F, r3, likilihood_params)


    rand = tf.placeholder(tf.float64, shape=[None])

    rand_JY = JY(rand, YJ_params3, recursion_num=recursion)
    rand2_JY = JY(rand_JY, YJ_params2, recursion_num=recursion)
    rand3_JY = JY(rand2_JY, YJ_params, recursion_num=recursion)

    fgpa = FGPA(rand3_JY, FGPA_params)
    
    return A, alpha, sigma, nfgpa, eta, eps, beta, mean, r, eta2, eps2, beta2, mean2, r2, eta3, eps3, beta3, mean3, r3, hess, outline, loss, fgpa
