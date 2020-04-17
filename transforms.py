from headers import *

def FGPA_Layer(layer, A_input, alpha_input, sigma_input, A_constraint=[0.+1e-50 , 5.], alpha_constraint=[1.6, 5.], sigma_constraint=[0+1e-50, 5.], constant=False):
    if constant:
        A = tf.constant(np.array([A_input]))
        alpha = tf.constant(np.array([alpha_input]))
        sigma = tf.constant(np.array([sigma_input]))
    else:
        A = tf.Variable(A_input, name=layer + '_A', dtype=tf.float64, constraint= lambda t: tf.clip_by_value(t, A_constraint[0] , A_constraint[1]))
        alpha = tf.Variable(alpha_input, name=layer + '_alpha', dtype=tf.float64, constraint= lambda t: tf.clip_by_value(t, alpha_constraint[0], alpha_constraint[1]))
        sigma = tf.Variable(sigma_input, name=layer + '_sigma', dtype=tf.float64, constraint= lambda t: tf.clip_by_value(t, sigma_constraint[0], sigma_constraint[1]))

    return np.array([A, alpha, sigma])

def NFGPA(F, params):
    #Defining Variables
    A = params[0]
    alpha = params[1]
    sigma = params[2]
    
    #Reverse FGPA
    nfgpa = ((sigma**2.)/2. + (tf.log(-tf.log(F)/A))/alpha)
    return nfgpa, A, alpha, sigma


def FGPA(delta, params):
    #Defining Variables
    A = params[0]
    alpha = params[1]
    sigma = params[2]
    
    #FGPA
    fgpa = tf.exp(-A*tf.exp(alpha*(delta-(sigma**2/2))))
    return fgpa

    
def YJ_Layer(layer, eta_input, eps_input, beta_input, mean_input, eta_constraint=[-5., 5.], eps_constraint=[-1.+1e-50, -1e-50], beta_constraint=[-5, -5], vary_mean=True, vary_beta=True):
    eta = tf.Variable(eta_input, name=layer + "_eta", dtype=tf.float64, constraint= lambda t: tf.clip_by_value(t, eta_constraint[0], eta_constraint[1]))
    eps = tf.Variable(eps_input, name=layer + "_eps", dtype=tf.float64, constraint= lambda t: tf.clip_by_value(t, eps_constraint[0], eps_constraint[1]))
    
    if vary_beta==True:
        beta = tf.Variable(beta_input, name=layer + "_beta", dtype=tf.float64, constraint= lambda t: tf.clip_by_value(t, beta_constraint[0], beta_constraint[1]))
    else:
        beta = tf.constant(beta_input)

    if vary_mean == True:
        mean = tf.Variable(mean_input, name=layer + "_mean", dtype=tf.float64, constraint= lambda t: tf.clip_by_value(t, -5., 5.))
    else:
        mean = tf.constant(np.array([mean_input]))
    
    return np.array([eta, eps, beta, mean])


def YJ(data, params, recursion_num):
    eta = params[0]
    eps = params[1]
    beta = params[2]
    mean = params[3]

    #yeo johnson positive and negative
    # #mask
    def yj_recursion(data, recursion_num):
        if recursion_num==0:

            return data
        else:
            less_data = tf.less(data, 0.)
            more_data = tf.greater_equal(data, 0.)

            zeroes = tf.zeros(tf.shape(data)[0], dtype=tf.float64)


            where_data = tf.where(more_data, data, zeroes)
            where_not = tf.where(less_data, data, zeroes)

            fp = ((where_data/beta + 1.)**(1. + eps) - 1.)/ (1. + eps)
            fn = -((-where_not/beta + 1.)**(1. - eps) - 1.) / (1. - eps)

            yj_start = tf.where(tf.greater_equal(data, 0.), fp, fn)

            return yj_recursion(yj_start, recursion_num-1)

    
    yj = yj_recursion(data, recursion_num)

    #seperating into pieces
    sa_o = (yj)*beta - mean
    sa_p = (tf.sinh(eta * yj) / eta)*beta - mean
    sa_n = (tf.asinh(eta * yj) / eta)*beta - mean

    #piecewise definition
    def f1(): return sa_o
    def f3(): return sa_p
    def f5(): return sa_n

    #logic pieces
    eta_eq_zero = tf.equal(eta, 0.)
    eta_greater_zero = tf.greater(eta, 0.)
    eta_less_zero = tf.less(eta, 0.)

    #logic for each part
    f1_logic = (eta_eq_zero)
    f3_logic = (eta_greater_zero)
    f5_logic = (eta_less_zero)

    #putting it together
    def casework():
        r = tf.case(((f1_logic, f1),
                 (f3_logic, f3),
                 (f5_logic, f5)),
             default=f1, exclusive=False)
        return r

    r = casework()
    r = tf.cast(r, tf.float64)
    
    return r, eta, eps, beta, mean

def JY(reverse, params, recursion_num):
    eta = params[0]
    eps = params[1]
    beta = params[2]
    mean = params[3]
    

    #seperating into pieces
    sa_o = (reverse+mean)/beta
    sa_p = (tf.sinh(eta * (reverse+mean)/beta) / eta)
    sa_n = (tf.asinh(eta * (reverse+mean)/beta) / eta)
    
    #here we are switching so the negative data is being sent to the positive bin and vice versa
    def f1(): return sa_o
    def f3(): return sa_p
    def f5(): return sa_n
    
    #logic pieces
    eta_eq_zero = tf.equal(eta, 0.)
    eta_greater_zero = tf.greater(eta, 0.)
    eta_less_zero = tf.less(eta, 0.)
    
    #logic for each part
    f1_logic = (eta_eq_zero)
    f3_logic = (eta_greater_zero)
    f5_logic = (eta_less_zero)
    

        #putting it together
    def casework():
        sa = tf.case(((f1_logic, f1),
                 (f3_logic, f5),
                 (f5_logic, f3)),
             default=f1, exclusive=False)
        return sa

    sa = casework()
    sa = tf.cast(sa, tf.float64)
    
    
    def jy_recursion(sa, recursion_num):
        
        if recursion_num == 0:
            return sa
        else:
            less_sa = tf.less(sa, 0.)
            more_sa = tf.greater_equal(sa, 0.)

            zeroes = tf.zeros(tf.shape(sa)[0], dtype=tf.float64)

            where_data = tf.where(more_sa, sa, zeroes)
            where_not = tf.where(less_sa, sa, zeroes)

            fp = ((1 + eps) * where_data + 1)**(1 / (1 + eps)) - 1
            fn = -(-(1 - eps) * where_not + 1)**(1 / (1 - eps)) + 1

            r_inv_start = tf.where(tf.greater_equal(sa, 0.), fp*beta, fn*beta)
            
            return jy_recursion(r_inv_start, recursion_num-1)
        
    r_inv = jy_recursion(sa, recursion_num)
    
    return r_inv


def log_likilihood_Layer(layer, input_hess, vary_hess=True):
    if vary_hess:
        hess = tf.Variable(input_hess, name=layer + 'hess', dtype=tf.float64, constraint= lambda t: tf.clip_by_value(t, 1./(100)**2, 1./(1e-50)**2))
    else:
         hess = tf.constant(np.array([input_hess]))
    return hess

def log_likilihood(F, r, params):
    #derivative of r for the log likelihood
    r_g = tf.gradients(r, F)

    #likelihood
    pi = np.array([np.pi])
    pi = pi.astype((np.float64))

    hess = params

    norm_logpdf = -.5 * r * hess * r + .5 * tf.log(hess) - .5 * tf.log(2 * pi)
    grad_abs_log = tf.log(tf.abs(r_g))

    loss_piece = -norm_logpdf-grad_abs_log
    
    outline = tf.exp(-loss_piece)
    
    loss = tf.reduce_sum(loss_piece)
    
    return loss, outline, hess, r_g
