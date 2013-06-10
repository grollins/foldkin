import numpy
from foldkin.base.parameter_set import ParameterSet
from foldkin.util import convert_beta_to_T, convert_T_to_beta,\
                         change_log10x_to_lnx, change_lnx_to_log10x

class CoopModelParameterSet(ParameterSet):
    """docstring for CoopModelParameterSet"""
    def __init__(self):
        super(CoopModelParameterSet, self).__init__()
        self.parameter_dict = {'N':5, 'log_K_ss':-1.0,
                               'log_K_ter':0.3, 'log_K_f':6.0,
                               'log_k0':5.7, 'beta':convert_T_to_beta(300.)}
        self.bounds_dict = {'log_K_ss':(None, None),
                            'log_K_ter':(None, None),
                            'log_K_f':(None, None),
                            'log_k0':(None, None)}

    def __str__(self):
        my_array = self.as_array()
        return "%s" % (my_array)

    def __iter__(self):
        for param_name, param_value in self.parameter_dict.iteritems():
            yield param_name, param_value

    def set_parameter(self, param_name, param_value):
        if param_name in self.parameter_dict.keys():
            self.parameter_dict[param_name] = param_value
        else:
            assert False, "No such parameter: %s" % param_name

    def get_parameter(self, param_name):
        if param_name in ['log_k1', 'log_K_ss', 'log_K_ter', 'log_K_f']:
            assert False, "Try computing rather than getting %s" % param_name
        else:
            return self.parameter_dict[param_name]

    def compute_log_k1_at_beta(self, beta):
        log10_k0 = self.parameter_dict['log_k0']
        return log10_k0

    def compute_log_K_ss_at_beta(self, beta):
        log10_K_ss = self.parameter_dict['log_K_ss']
        return log10_K_ss

    def compute_log_K_ter_at_beta(self, beta):
        log10_K_ter = self.parameter_dict['log_K_ter']
        return log10_K_ter

    def compute_log_K_f_at_beta(self, beta):
        log10_K_f = self.parameter_dict['log_K_f']
        return log10_K_f

    def as_array(self):
        beta = self.get_parameter('beta')
        log_K_ss = self.compute_log_K_ss_at_beta(beta)
        log_K_ter = self.compute_log_K_ter_at_beta(beta)
        log_K_f = self.compute_log_K_f_at_beta(beta)
        log_k0 = self.get_parameter('log_k0')
        N = self.get_parameter('N')
        return numpy.array([log_K_ss, log_K_ter, log_K_f, log_k0, N, beta])

    def as_array_for_scipy_optimizer(self):
        beta = self.get_parameter('beta')
        log_K_ss = self.compute_log_K_ss_at_beta(beta)
        log_K_ter = self.compute_log_K_ter_at_beta(beta)
        log_K_f = self.compute_log_K_f_at_beta(beta)
        log_k0 = self.get_parameter('log_k0')
        return numpy.array([log_K_ss, log_K_ter, log_K_f, log_k0])

    def update_from_array(self, parameter_array):
        """Expected order of parameters in array:
           log_K_ss, log_K_ter, log_K_f, log_k0
           N will not be updated
        """
        parameter_array = numpy.atleast_1d(parameter_array)
        self.set_parameter('log_K_ss', parameter_array[0])
        self.set_parameter('log_K_ter', parameter_array[1])
        self.set_parameter('log_K_f', parameter_array[2])
        self.set_parameter('log_k0', parameter_array[3])

    def set_parameter_bounds(self, parameter_name, min_value, max_value):
        self.bounds_dict[parameter_name] = (min_value, max_value)

    def get_parameter_bounds(self, parameter_name):
        return self.bounds_dict[parameter_name]

    def get_parameter_bounds_list(self):
        '''N will not be updated'''
        log_K_ss_bounds = self.get_parameter_bounds('log_K_ss')
        log_K_ter_bounds = self.get_parameter_bounds('log_K_ter')
        log_K_f_bounds = self.get_parameter_bounds('log_K_f')
        log_k0_bounds = self.get_parameter_bounds('log_k0')
        bounds = [log_K_ss_bounds, log_K_ter_bounds, log_K_f_bounds,
                  log_k0_bounds]
        return bounds


class SimpleTParameterSet(ParameterSet):
    """docstring for SimpleTParameterSet"""
    def __init__(self):
        super(SimpleTParameterSet, self).__init__()
        self.parameter_dict = {'N':5, 'G_ss':-1.0,
                               'G_ter':0.3, 'G_f':6.0, 'G_act':0.0,
                               'log_k0':5.7, 'a':0.0,
                               'beta':convert_T_to_beta(300.)}
        self.bounds_dict = {'G_ss':(None, None),
                            'G_ter':(None, None),
                            'G_f':(None, None),
                            'G_act':(None, None),
                            'log_k0':(None, None),
                            'a':(None,None)}

    def __str__(self):
        my_array = self.as_array()
        return "%s" % (my_array)

    def __iter__(self):
        for param_name, param_value in self.parameter_dict.iteritems():
            yield param_name, param_value

    def set_parameter(self, param_name, param_value):
        if param_name in self.parameter_dict.keys():
            self.parameter_dict[param_name] = param_value
        else:
            assert False, "No such parameter: %s" % param_name

    def get_parameter(self, param_name):
        if param_name in ['log_k1', 'log_K_ss', 'log_K_ter', 'log_K_f']:
            assert False, "Try computing rather than getting %s" % param_name
        else:
            return self.parameter_dict[param_name]

    def compute_log_k1_at_beta(self, beta):
        log10_k0 = self.parameter_dict['log_k0']
        G_act = self.parameter_dict['G_act']
        ln_k0 = change_log10x_to_lnx(log10_k0)
        ln_k1 = ln_k0 - (G_act * beta)
        log10_k1 = change_lnx_to_log10x(ln_k1)
        return log10_k1

    def compute_log_K_ss_at_beta(self, beta):
        G_ss = self.parameter_dict['G_ss']
        ln_K_ss = -G_ss * beta
        log10_K_ss = change_lnx_to_log10x(ln_K_ss)
        return log10_K_ss

    def compute_log_K_ter_at_beta(self, beta):
        G_ter = self.parameter_dict['G_ter']
        ln_K_ter = -G_ter * beta
        log10_K_ter = change_lnx_to_log10x(ln_K_ter)
        return log10_K_ter

    def compute_log_K_f_at_beta(self, beta):
        G_f = self.parameter_dict['G_f']
        ln_K_f = -G_f * beta
        log10_K_f = change_lnx_to_log10x(ln_K_f)
        return log10_K_f

    def as_array(self):
        beta = self.get_parameter('beta')
        G_ss = self.get_parameter('G_ss')
        G_ter = self.get_parameter('G_ter')
        G_f = self.get_parameter('G_f')
        G_act = self.get_parameter('G_act')
        log_k0 = self.get_parameter('log_k0')
        N = self.get_parameter('N')
        a = self.get_parameter('a')
        return numpy.array([G_ss, G_ter, G_f, G_act, log_k0, N, beta, a])

    def as_array_for_scipy_optimizer(self):
        G_ss = self.get_parameter('G_ss')
        G_ter = self.get_parameter('G_ter')
        G_f = self.get_parameter('G_f')
        G_act = self.get_parameter('G_act')
        log_k0 = self.get_parameter('log_k0')
        a = self.get_parameter('a')
        return numpy.array([G_ss, G_ter, G_f, G_act, log_k0, a])

    def update_from_array(self, parameter_array):
        """Expected order of parameters in array:
           G_ss, G_ter, G_f, G_act, log_k0, a
           N will not be updated
        """
        parameter_array = numpy.atleast_1d(parameter_array)
        self.set_parameter('G_ss', parameter_array[0])
        self.set_parameter('G_ter', parameter_array[1])
        self.set_parameter('G_f', parameter_array[2])
        self.set_parameter('G_act', parameter_array[3])
        self.set_parameter('log_k0', parameter_array[4])
        self.set_parameter('a', parameter_array[5])

    def set_parameter_bounds(self, parameter_name, min_value, max_value):
        self.bounds_dict[parameter_name] = (min_value, max_value)

    def get_parameter_bounds(self, parameter_name):
        return self.bounds_dict[parameter_name]

    def get_parameter_bounds_list(self):
        '''N will not be updated'''
        G_ss_bounds = self.get_parameter_bounds('G_ss')
        G_ter_bounds = self.get_parameter_bounds('G_ter')
        G_f_bounds = self.get_parameter_bounds('G_f')
        G_act_bounds = self.get_parameter_bounds('G_act')
        log_k0_bounds = self.get_parameter_bounds('log_k0')
        a_bounds = self.get_parameter_bounds('a')
        bounds = [G_ss_bounds, G_ter_bounds, G_f_bounds, G_act_bounds,
                  log_k0_bounds, a_bounds]
        return bounds


class FixedK_SS_TER_TempDependenceParameterSet(ParameterSet):
    """docstring for FixedK_SS_TER_TempDependenceParameterSet"""
    def __init__(self):
        super(FixedK_SS_TER_TempDependenceParameterSet, self).__init__()
        self.parameter_dict = {'N':5, 'log_K_ss':-1.0, 'log_K_ter':0.3,
                               'H_ss':0.0, 'H_ter':-0.5,
                               'S_ss':-1e-3, 'S_ter':-1e-3,
                               'G_f':-1.5, 'G_act':0.0, 'log_k0':5.0,
                               'beta':convert_T_to_beta(300.)}
        self.bounds_dict = {'log_K_ss':(None, None),
                            'log_K_ter':(None, None),
                            'H_ss':(None, None),
                            'H_ter':(None, None),
                            'G_f':(None, None),
                            'G_act':(None, None),
                            'log_k0':(None, None),
                            'beta':(None, None)}
        # beta of log_K's
        self.logK_beta = convert_T_to_beta(300.)
        # initialize S values
        self.parameter_dict['S_ss'] = self.compute_S_ss()
        self.parameter_dict['S_ter'] = self.compute_S_ter()

    def __str__(self):
        my_array = self.as_array()
        return "%s" % (my_array)

    def __iter__(self):
        for param_name, param_value in self.parameter_dict.iteritems():
            yield param_name, param_value

    def set_parameter(self, param_name, param_value):
        if param_name == 'H_ss':
            self.parameter_dict['H_ss'] = param_value
            S_ss = self.compute_S_ss()
            self.parameter_dict['S_ss'] = S_ss
        elif param_name == 'H_ter':
            self.parameter_dict['H_ter'] = param_value
            S_ter = self.compute_S_ter()
            self.parameter_dict['S_ter'] = S_ter
        elif param_name == 'log_K_ss':
            self.parameter_dict['log_K_ss'] = param_value
            S_ss = self.compute_S_ss()
            self.parameter_dict['S_ss'] = S_ss
        elif param_name == 'log_K_ter':
            self.parameter_dict['log_K_ter'] = param_value
            S_ter = self.compute_S_ter()
            self.parameter_dict['S_ter'] = S_ter
        elif param_name in ['S_ss', 'S_ter']:
            print "Cannot set S values directly."
        elif param_name in self.parameter_dict.keys():
            self.parameter_dict[param_name] = param_value
        else:
            assert False, "No such parameter: %s" % param_name

    def get_parameter(self, param_name):
        if param_name in ['log_K_ss', 'log_K_ter']:
            assert False, "Try computing rather than getting %s" % param_name
        elif param_name in self.parameter_dict.keys():
            return self.parameter_dict[param_name]
        else:
            assert False, "No such parameter: %s" % param_name

    def compute_log_k1_at_beta(self, beta):
        log10_k0 = self.parameter_dict['log_k0']
        G_act = self.parameter_dict['G_act']
        ln_k0 = change_log10x_to_lnx(log10_k0)
        ln_k1 = ln_k0 - (G_act * beta)
        log10_k1 = change_lnx_to_log10x(ln_k1)
        return log10_k1

    def compute_log_K_ss_at_beta(self, beta):
        H_ss = self.get_parameter('H_ss')
        S_ss = self.get_parameter('S_ss')
        T = convert_beta_to_T(beta)
        ln_K_ss = -(H_ss - T * S_ss) * beta
        log10_K_ss = change_lnx_to_log10x(ln_K_ss)
        return log10_K_ss

    def compute_log_K_ter_at_beta(self, beta):
        H_ter = self.get_parameter('H_ter')
        S_ter = self.get_parameter('S_ter')
        T = convert_beta_to_T(beta)
        ln_K_ter = -(H_ter - T * S_ter) * beta
        log10_K_ter = change_lnx_to_log10x(ln_K_ter)
        return log10_K_ter

    def compute_log_K_f_at_beta(self, beta):
        G_f = self.parameter_dict['G_f']
        ln_K_f = -G_f * beta
        log10_K_f = change_lnx_to_log10x(ln_K_f)
        return log10_K_f

    def compute_S_ss(self):
        '''
        K = exp((-H+TS)*beta)
        lnK = -H*beta + T*beta*S
        lnK + H*beta = T * beta * S
        S = lnK/(T*beta) + H/T
        '''
        beta = self.logK_beta
        H_ss = self.get_parameter('H_ss')
        log10_K_ss = self.parameter_dict['log_K_ss']
        lnK_ss = change_log10x_to_lnx(log10_K_ss)
        T = convert_beta_to_T(beta)
        S_ss = lnK_ss / (T * beta) + H_ss/T
        return S_ss

    def compute_S_ter(self):
        '''
        K = exp((-H+TS)*beta)
        lnK = -H*beta + T*beta*S
        lnK + H*beta = T * beta * S
        S = lnK/(T*beta) + H/T
        '''
        beta = self.logK_beta
        H_ter = self.get_parameter('H_ter')
        log10_K_ter = self.parameter_dict['log_K_ter']
        lnK_ter = change_log10x_to_lnx(log10_K_ter)
        T = convert_beta_to_T(beta)
        S_ter = lnK_ter / (T * beta) + H_ter/T
        return S_ter

    def as_array(self):
        log_K_ss = self.compute_log_K_ss_at_beta(self.logK_beta)
        log_K_ter = self.compute_log_K_ter_at_beta(self.logK_beta)
        # log_K_ss = self.parameter_dict['log_K_ss']
        # log_K_ter = self.parameter_dict['log_K_ter']
        H_ss = self.get_parameter('H_ss')
        H_ter = self.get_parameter('H_ter')
        S_ss = self.get_parameter('S_ss')
        S_ter = self.get_parameter('S_ter')
        G_f = self.get_parameter('G_f')
        G_act = self.get_parameter('G_act')
        log_k0 = self.get_parameter('log_k0')
        beta = self.get_parameter('beta')
        N = self.get_parameter('N')
        return numpy.array([log_K_ss, log_K_ter, H_ss, H_ter, S_ss, S_ter, G_f,
                            G_act, log_k0, beta, N])

    def as_array_for_scipy_optimizer(self):
        log_K_ss = self.compute_log_K_ss_at_beta(self.logK_beta)
        log_K_ter = self.compute_log_K_ter_at_beta(self.logK_beta)
        H_ss = self.get_parameter('H_ss')
        H_ter = self.get_parameter('H_ter')
        G_f = self.get_parameter('G_f')
        G_act = self.get_parameter('G_act')
        log_k0 = self.get_parameter('log_k0')
        return numpy.array([log_K_ss, log_K_ter, H_ss, H_ter,
                            G_f, G_act, log_k0])

    def update_from_array(self, parameter_array):
        """Expected order of parameters in array:
           log_K_ss, log_K_ter, H_ss, H_ter, G_f, G_act, log_k0
           N and beta will not be set this way
        """
        parameter_array = numpy.atleast_1d(parameter_array)
        self.set_parameter('log_K_ss', parameter_array[0])
        self.set_parameter('log_K_ter', parameter_array[1])
        self.set_parameter('H_ss', parameter_array[2])
        self.set_parameter('H_ter', parameter_array[3])
        self.set_parameter('G_f', parameter_array[4])
        self.set_parameter('G_act', parameter_array[5])
        self.set_parameter('log_k0', parameter_array[6])

    def set_parameter_bounds(self, parameter_name, min_value, max_value):
        self.bounds_dict[parameter_name] = (min_value, max_value)

    def get_parameter_bounds(self, parameter_name):
        return self.bounds_dict[parameter_name]

    def get_parameter_bounds_list(self):
        '''N and beta will not be varied, so no bounds'''
        log_K_ss_bounds = self.bounds_dict['log_K_ss']
        log_K_ter_bounds = self.bounds_dict['log_K_ter']
        H_ss_bounds = self.bounds_dict['H_ss']
        H_ter_bounds = self.bounds_dict['H_ter']
        G_f_bounds = self.bounds_dict['G_f']
        G_act_bounds = self.bounds_dict['G_act']
        log_k0_bounds = self.bounds_dict['log_k0']
        bounds = [log_K_ss_bounds, log_K_ter_bounds, H_ss_bounds, H_ter_bounds,
                  G_f_bounds, G_act_bounds, log_k0_bounds]
        return bounds


class FixedK_SS_TER_k1_ParameterSet(ParameterSet):
    """docstring for FixedK_SS_TER_k1_ParameterSet"""
    def __init__(self):
        super(FixedK_SS_TER_k1_ParameterSet, self).__init__()
        self.parameter_dict = {'N':5, 'log_K_ss':-1.0, 'log_K_ter':0.3,
                               'H_ss':0.0, 'H_ter':-0.5,
                               'S_ss':-1e-3, 'S_ter':-1e-3,
                               'G_f':-1.5, 'G_act':0.0, 'log_k0':5.6,
                               'log_k1':5.6, 'beta':convert_T_to_beta(300.)}
        self.bounds_dict = {'log_K_ss':(None, None),
                            'log_K_ter':(None, None),
                            'H_ss':(None, None),
                            'H_ter':(None, None),
                            'G_f':(None, None),
                            'G_act':(None, None),
                            'log_k1':(None, None),
                            'beta':(None, None)}
        # beta of log_K's
        self.logK_beta = convert_T_to_beta(300.)
        # initialize S values
        self.parameter_dict['S_ss'] = self.compute_S_ss()
        self.parameter_dict['S_ter'] = self.compute_S_ter()

    def __str__(self):
        my_array = self.as_array()
        return "%s" % (my_array)

    def __iter__(self):
        for param_name, param_value in self.parameter_dict.iteritems():
            yield param_name, param_value

    def set_parameter(self, param_name, param_value):
        if param_name == 'H_ss':
            self.parameter_dict['H_ss'] = param_value
            S_ss = self.compute_S_ss()
            self.parameter_dict['S_ss'] = S_ss
        elif param_name == 'H_ter':
            self.parameter_dict['H_ter'] = param_value
            S_ter = self.compute_S_ter()
            self.parameter_dict['S_ter'] = S_ter
        elif param_name == 'log_K_ss':
            self.parameter_dict['log_K_ss'] = param_value
            S_ss = self.compute_S_ss()
            self.parameter_dict['S_ss'] = S_ss
        elif param_name == 'log_K_ter':
            self.parameter_dict['log_K_ter'] = param_value
            S_ter = self.compute_S_ter()
            self.parameter_dict['S_ter'] = S_ter
        elif param_name in ['S_ss', 'S_ter']:
            print "Cannot set S values directly."
        elif param_name == 'log_k0':
            print "Cannot set log_k0 directly."
        elif param_name == 'G_act':
            self.parameter_dict['G_act'] = param_value
            log_k0 = self.compute_log_k0()
            self.parameter_dict['log_k0'] = log_k0
        elif param_name in self.parameter_dict.keys():
            self.parameter_dict[param_name] = param_value
        else:
            assert False, "No such parameter: %s" % param_name

    def get_parameter(self, param_name):
        if param_name in ['log_K_ss', 'log_K_ter', 'log_k1']:
            assert False, "Try computing rather than getting %s" % param_name
        elif param_name in self.parameter_dict.keys():
            return self.parameter_dict[param_name]
        else:
            assert False, "No such parameter: %s" % param_name

    def compute_log_k1_at_beta(self, beta):
        log10_k0 = self.parameter_dict['log_k0']
        G_act = self.parameter_dict['G_act']
        ln_k0 = change_log10x_to_lnx(log10_k0)
        ln_k1 = ln_k0 - (G_act * beta)
        log10_k1 = change_lnx_to_log10x(ln_k1)
        return log10_k1

    def compute_log_K_ss_at_beta(self, beta):
        H_ss = self.get_parameter('H_ss')
        S_ss = self.get_parameter('S_ss')
        T = convert_beta_to_T(beta)
        ln_K_ss = -(H_ss - T * S_ss) * beta
        log10_K_ss = change_lnx_to_log10x(ln_K_ss)
        return log10_K_ss

    def compute_log_K_ter_at_beta(self, beta):
        H_ter = self.get_parameter('H_ter')
        S_ter = self.get_parameter('S_ter')
        T = convert_beta_to_T(beta)
        ln_K_ter = -(H_ter - T * S_ter) * beta
        log10_K_ter = change_lnx_to_log10x(ln_K_ter)
        return log10_K_ter

    def compute_log_K_f_at_beta(self, beta):
        G_f = self.parameter_dict['G_f']
        ln_K_f = -G_f * beta
        log10_K_f = change_lnx_to_log10x(ln_K_f)
        return log10_K_f

    def compute_log_k0(self):
        beta = self.logK_beta
        G_act = self.get_parameter('G_act')
        log_k1 = self.get_parameter('log_k1')
        # k1 = k0 * exp(-G_act * beta)
        # lnk1 = lnk0 - G_act * beta
        # lnk0 = lnk1 + G_act * beta
        lnk1 = change_log10x_to_lnx(log_k1)
        lnk0 = lnk1 + G_act * beta
        log_k0 = change_lnx_to_log10x(lnk0)
        return log_k0

    def compute_S_ss(self):
        '''
        K = exp((-H+TS)*beta)
        lnK = -H*beta + T*beta*S
        lnK + H*beta = T * beta * S
        S = lnK/(T*beta) + H/T
        '''
        beta = self.logK_beta
        H_ss = self.get_parameter('H_ss')
        log10_K_ss = self.parameter_dict['log_K_ss']
        lnK_ss = change_log10x_to_lnx(log10_K_ss)
        T = convert_beta_to_T(beta)
        S_ss = lnK_ss / (T * beta) + H_ss/T
        return S_ss

    def compute_S_ter(self):
        '''
        K = exp((-H+TS)*beta)
        lnK = -H*beta + T*beta*S
        lnK + H*beta = T * beta * S
        S = lnK/(T*beta) + H/T
        '''
        beta = self.logK_beta
        H_ter = self.get_parameter('H_ter')
        log10_K_ter = self.parameter_dict['log_K_ter']
        lnK_ter = change_log10x_to_lnx(log10_K_ter)
        T = convert_beta_to_T(beta)
        S_ter = lnK_ter / (T * beta) + H_ter/T
        return S_ter

    def as_array(self):
        log_K_ss = self.compute_log_K_ss_at_beta(self.logK_beta)
        log_K_ter = self.compute_log_K_ter_at_beta(self.logK_beta)
        # log_K_ss = self.parameter_dict['log_K_ss']
        # log_K_ter = self.parameter_dict['log_K_ter']
        H_ss = self.get_parameter('H_ss')
        H_ter = self.get_parameter('H_ter')
        S_ss = self.get_parameter('S_ss')
        S_ter = self.get_parameter('S_ter')
        G_f = self.get_parameter('G_f')
        G_act = self.get_parameter('G_act')
        log_k0 = self.get_parameter('log_k0')
        log_k1 = self.compute_log_k1_at_beta(self.logK_beta)
        beta = self.get_parameter('beta')
        N = self.get_parameter('N')
        return numpy.array([log_K_ss, log_K_ter, H_ss, H_ter, S_ss, S_ter, G_f,
                            G_act, log_k0, log_k1, beta, a, N])

    def as_array_for_scipy_optimizer(self):
        log_K_ss = self.compute_log_K_ss_at_beta(self.logK_beta)
        log_K_ter = self.compute_log_K_ter_at_beta(self.logK_beta)
        H_ss = self.get_parameter('H_ss')
        H_ter = self.get_parameter('H_ter')
        G_f = self.get_parameter('G_f')
        G_act = self.get_parameter('G_act')
        log_k1 = self.compute_log_k1_at_beta(self.logK_beta)
        return numpy.array([log_K_ss, log_K_ter, H_ss, H_ter, G_f,
                            G_act, log_k1, a])

    def update_from_array(self, parameter_array):
        """Expected order of parameters in array:
           log_K_ss, log_K_ter, H_ss, H_ter, G_f, G_act, log_k1
           N and beta will not be set this way
        """
        parameter_array = numpy.atleast_1d(parameter_array)
        self.set_parameter('log_K_ss', parameter_array[0])
        self.set_parameter('log_K_ter', parameter_array[1])
        self.set_parameter('H_ss', parameter_array[2])
        self.set_parameter('H_ter', parameter_array[3])
        self.set_parameter('G_f', parameter_array[4])
        self.set_parameter('G_act', parameter_array[5])
        self.set_parameter('log_k1', parameter_array[6])

    def set_parameter_bounds(self, parameter_name, min_value, max_value):
        self.bounds_dict[parameter_name] = (min_value, max_value)

    def get_parameter_bounds(self, parameter_name):
        return self.bounds_dict[parameter_name]

    def get_parameter_bounds_list(self):
        '''N and beta will not be varied, so no bounds'''
        log_K_ss_bounds = self.bounds_dict['log_K_ss']
        log_K_ter_bounds = self.bounds_dict['log_K_ter']
        H_ss_bounds = self.bounds_dict['H_ss']
        H_ter_bounds = self.bounds_dict['H_ter']
        G_f_bounds = self.bounds_dict['G_f']
        G_act_bounds = self.bounds_dict['G_act']
        log_k1_bounds = self.bounds_dict['log_k1']
        bounds = [log_K_ss_bounds, log_K_ter_bounds, H_ss_bounds, H_ter_bounds,
                  G_f_bounds, G_act_bounds, log_k1_bounds]
        return bounds


class ManyParamTempDependenceParameterSet(ParameterSet):
    """docstring for ManyParamTempDependenceParameterSet"""
    def __init__(self):
        super(ManyParamTempDependenceParameterSet, self).__init__()
        self.parameter_dict = {'N':5, 'log_K_ss':-1.0, 'log_K_ter':0.3,
                               'H_ss':0.0, 'H_ter':-0.5,
                               'S_ss':-1e-3, 'S_ter':-1e-3,
                               'G_f':-1.5, 'H_act':0.0, 'S_act':0.0,
                               'log_k0':5.0, 'beta':convert_T_to_beta(300.)}
        self.bounds_dict = {'log_K_ss':(None, None),
                            'log_K_ter':(None, None),
                            'H_ss':(None, None),
                            'H_ter':(None, None),
                            'G_f':(None, None),
                            'H_act':(None, None),
                            'S_act':(None, None),
                            'log_k0':(None, None),
                            'beta':(None, None)}
        # beta of log_K's
        self.logK_beta = convert_T_to_beta(300.)
        # initialize S values
        self.parameter_dict['S_ss'] = self.compute_S_ss()
        self.parameter_dict['S_ter'] = self.compute_S_ter()

    def __str__(self):
        my_array = self.as_array()
        return "%s" % (my_array)

    def __iter__(self):
        for param_name, param_value in self.parameter_dict.iteritems():
            yield param_name, param_value

    def set_parameter(self, param_name, param_value):
        if param_name == 'H_ss':
            self.parameter_dict['H_ss'] = param_value
            S_ss = self.compute_S_ss()
            self.parameter_dict['S_ss'] = S_ss
        elif param_name == 'H_ter':
            self.parameter_dict['H_ter'] = param_value
            S_ter = self.compute_S_ter()
            self.parameter_dict['S_ter'] = S_ter
        elif param_name == 'log_K_ss':
            self.parameter_dict['log_K_ss'] = param_value
            S_ss = self.compute_S_ss()
            self.parameter_dict['S_ss'] = S_ss
        elif param_name == 'log_K_ter':
            self.parameter_dict['log_K_ter'] = param_value
            S_ter = self.compute_S_ter()
            self.parameter_dict['S_ter'] = S_ter
        elif param_name in ['S_ss', 'S_ter']:
            print "Cannot set S values directly."
        elif param_name in self.parameter_dict.keys():
            self.parameter_dict[param_name] = param_value
        else:
            assert False, "No such parameter: %s" % param_name

    def get_parameter(self, param_name):
        if param_name in ['log_K_ss', 'log_K_ter']:
            assert False, "Try computing rather than getting %s" % param_name
        elif param_name in self.parameter_dict.keys():
            return self.parameter_dict[param_name]
        else:
            assert False, "No such parameter: %s" % param_name

    def compute_log_k1_at_beta(self, beta):
        log10_k0 = self.parameter_dict['log_k0']
        H_act = self.parameter_dict['H_act']
        S_act = self.parameter_dict['S_act']
        ln_k0 = change_log10x_to_lnx(log10_k0)
        T = convert_beta_to_T(beta)
        ln_k1 = ln_k0 - ((H_act - T*S_act) * beta)
        log10_k1 = change_lnx_to_log10x(ln_k1)
        return log10_k1

    def compute_log_K_ss_at_beta(self, beta):
        H_ss = self.get_parameter('H_ss')
        S_ss = self.get_parameter('S_ss')
        T = convert_beta_to_T(beta)
        ln_K_ss = -(H_ss - T * S_ss) * beta
        log10_K_ss = change_lnx_to_log10x(ln_K_ss)
        return log10_K_ss

    def compute_log_K_ter_at_beta(self, beta):
        H_ter = self.get_parameter('H_ter')
        S_ter = self.get_parameter('S_ter')
        T = convert_beta_to_T(beta)
        ln_K_ter = -(H_ter - T * S_ter) * beta
        log10_K_ter = change_lnx_to_log10x(ln_K_ter)
        return log10_K_ter

    def compute_log_K_f_at_beta(self, beta):
        G_f = self.parameter_dict['G_f']
        ln_K_f = -G_f * beta
        log10_K_f = change_lnx_to_log10x(ln_K_f)
        return log10_K_f

    def compute_S_ss(self):
        '''
        K = exp((-H+TS)*beta)
        lnK = -H*beta + T*beta*S
        lnK + H*beta = T * beta * S
        S = lnK/(T*beta) + H/T
        '''
        beta = self.logK_beta
        H_ss = self.get_parameter('H_ss')
        log10_K_ss = self.parameter_dict['log_K_ss']
        lnK_ss = change_log10x_to_lnx(log10_K_ss)
        T = convert_beta_to_T(beta)
        S_ss = lnK_ss / (T * beta) + H_ss/T
        return S_ss

    def compute_S_ter(self):
        '''
        K = exp((-H+TS)*beta)
        lnK = -H*beta + T*beta*S
        lnK + H*beta = T * beta * S
        S = lnK/(T*beta) + H/T
        '''
        beta = self.logK_beta
        H_ter = self.get_parameter('H_ter')
        log10_K_ter = self.parameter_dict['log_K_ter']
        lnK_ter = change_log10x_to_lnx(log10_K_ter)
        T = convert_beta_to_T(beta)
        S_ter = lnK_ter / (T * beta) + H_ter/T
        return S_ter

    def as_array(self):
        log_K_ss = self.compute_log_K_ss_at_beta(self.logK_beta)
        log_K_ter = self.compute_log_K_ter_at_beta(self.logK_beta)
        # log_K_ss = self.parameter_dict['log_K_ss']
        # log_K_ter = self.parameter_dict['log_K_ter']
        H_ss = self.get_parameter('H_ss')
        H_ter = self.get_parameter('H_ter')
        S_ss = self.get_parameter('S_ss')
        S_ter = self.get_parameter('S_ter')
        G_f = self.get_parameter('G_f')
        H_act = self.get_parameter('H_act')
        S_act = self.get_parameter('S_act')
        log_k0 = self.get_parameter('log_k0')
        beta = self.get_parameter('beta')
        N = self.get_parameter('N')
        return numpy.array([log_K_ss, log_K_ter, H_ss, H_ter, S_ss, S_ter, G_f,
                            H_act, S_act, log_k0, beta, N])

    def as_array_for_scipy_optimizer(self):
        log_K_ss = self.compute_log_K_ss_at_beta(self.logK_beta)
        log_K_ter = self.compute_log_K_ter_at_beta(self.logK_beta)
        H_ss = self.get_parameter('H_ss')
        H_ter = self.get_parameter('H_ter')
        G_f = self.get_parameter('G_f')
        H_act = self.get_parameter('H_act')
        S_act = self.get_parameter('S_act')
        log_k0 = self.get_parameter('log_k0')
        return numpy.array([log_K_ss, log_K_ter, H_ss, H_ter, G_f,
                            H_act, S_act, log_k0])

    def update_from_array(self, parameter_array):
        """Expected order of parameters in array:
           log_K_ss, log_K_ter, H_ss, H_ter, G_f, H_act, S_act, log_k0
           N and beta will not be set this way
        """
        parameter_array = numpy.atleast_1d(parameter_array)
        self.set_parameter('log_K_ss', parameter_array[0])
        self.set_parameter('log_K_ter', parameter_array[1])
        self.set_parameter('H_ss', parameter_array[2])
        self.set_parameter('H_ter', parameter_array[3])
        self.set_parameter('G_f', parameter_array[4])
        self.set_parameter('H_act', parameter_array[5])
        self.set_parameter('S_act', parameter_array[6])
        self.set_parameter('log_k0', parameter_array[7])

    def set_parameter_bounds(self, parameter_name, min_value, max_value):
        self.bounds_dict[parameter_name] = (min_value, max_value)

    def get_parameter_bounds(self, parameter_name):
        return self.bounds_dict[parameter_name]

    def get_parameter_bounds_list(self):
        '''N and beta will not be varied, so no bounds'''
        log_K_ss_bounds = self.bounds_dict['log_K_ss']
        log_K_ter_bounds = self.bounds_dict['log_K_ter']
        H_ss_bounds = self.bounds_dict['H_ss']
        H_ter_bounds = self.bounds_dict['H_ter']
        G_f_bounds = self.bounds_dict['G_f']
        H_act_bounds = self.bounds_dict['H_act']
        S_act_bounds = self.bounds_dict['S_act']
        log_k0_bounds = self.bounds_dict['log_k0']
        bounds = [log_K_ss_bounds, log_K_ter_bounds, H_ss_bounds, H_ter_bounds,
                  G_f_bounds, H_act_bounds, S_act_bounds, log_k0_bounds]
        return bounds


class TempDependenceParameterSet(ParameterSet):
    """docstring for TempDependenceParameterSet"""
    def __init__(self):
        super(TempDependenceParameterSet, self).__init__()
        self.parameter_dict = {'N':5, 'G_ss':0.6, 'G_ter':-0.6,
                               'G_f':-1.5, 'H_act':0.0, 'S_act':0.0,
                               'log_k0':5.0, 'beta':convert_T_to_beta(300.)}
        self.bounds_dict = {'G_ss':(None, None),
                            'G_ter':(None, None),
                            'G_f':(None, None),
                            'H_act':(None, None),
                            'S_act':(None, None),
                            'log_k0':(None, None),
                            'beta':(None, None)}
        # beta of log_K's
        self.logK_beta = convert_T_to_beta(300.)

    def __str__(self):
        my_array = self.as_array()
        return "%s" % (my_array)

    def __iter__(self):
        for param_name, param_value in self.parameter_dict.iteritems():
            yield param_name, param_value

    def set_parameter(self, param_name, param_value):
        if param_name in self.parameter_dict.keys():
            self.parameter_dict[param_name] = param_value
        else:
            assert False, "No such parameter: %s" % param_name

    def get_parameter(self, param_name):
        if param_name in ['log_K_ss', 'log_K_ter']:
            assert False, "Try computing rather than getting %s" % param_name
        elif param_name in self.parameter_dict.keys():
            return self.parameter_dict[param_name]
        else:
            assert False, "No such parameter: %s" % param_name

    def compute_log_k1_at_beta(self, beta):
        log10_k0 = self.parameter_dict['log_k0']
        H_act = self.parameter_dict['H_act']
        S_act = self.parameter_dict['S_act']
        ln_k0 = change_log10x_to_lnx(log10_k0)
        T = convert_beta_to_T(beta)
        ln_k1 = ln_k0 - ((H_act - T*S_act) * beta)
        log10_k1 = change_lnx_to_log10x(ln_k1)
        return log10_k1

    def compute_log_K_ss_at_beta(self, beta):
        G_ss = self.parameter_dict['G_ss']
        ln_K_ss = -G_ss * beta
        log10_K_ss = change_lnx_to_log10x(ln_K_ss)
        return log10_K_ss

    def compute_log_K_ter_at_beta(self, beta):
        G_ter = self.parameter_dict['G_ter']
        ln_K_ter = -G_ter * beta
        log10_K_ter = change_lnx_to_log10x(ln_K_ter)
        return log10_K_ter

    def compute_log_K_f_at_beta(self, beta):
        G_f = self.parameter_dict['G_f']
        ln_K_f = -G_f * beta
        log10_K_f = change_lnx_to_log10x(ln_K_f)
        return log10_K_f

    def as_array(self):
        G_ss = self.get_parameter('G_ss')
        G_ter = self.get_parameter('G_ter')
        G_f = self.get_parameter('G_f')
        H_act = self.get_parameter('H_act')
        S_act = self.get_parameter('S_act')
        log_k0 = self.get_parameter('log_k0')
        beta = self.get_parameter('beta')
        N = self.get_parameter('N')
        return numpy.array([G_ss, G_ter, G_f, H_act, S_act, log_k0, beta, N])

    def as_array_for_scipy_optimizer(self):
        G_ss = self.get_parameter('G_ss')
        G_ter = self.get_parameter('G_ter')
        G_f = self.get_parameter('G_f')
        H_act = self.get_parameter('H_act')
        S_act = self.get_parameter('S_act')
        log_k0 = self.get_parameter('log_k0')
        return numpy.array([G_ss, G_ter, G_f, H_act, S_act, log_k0])

    def update_from_array(self, parameter_array):
        """Expected order of parameters in array:
           G_ss, G_ter, G_f, H_act, S_act, log_k0
           N and beta will not be set this way
        """
        parameter_array = numpy.atleast_1d(parameter_array)
        self.set_parameter('G_ss', parameter_array[0])
        self.set_parameter('G_ter', parameter_array[1])
        self.set_parameter('G_f', parameter_array[2])
        self.set_parameter('H_act', parameter_array[3])
        self.set_parameter('S_act', parameter_array[4])
        self.set_parameter('log_k0', parameter_array[5])

    def set_parameter_bounds(self, parameter_name, min_value, max_value):
        self.bounds_dict[parameter_name] = (min_value, max_value)

    def get_parameter_bounds(self, parameter_name):
        return self.bounds_dict[parameter_name]

    def get_parameter_bounds_list(self):
        '''N and beta will not be varied, so no bounds'''
        G_ss_bounds = self.bounds_dict['G_ss']
        G_ter_bounds = self.bounds_dict['G_ter']
        G_f_bounds = self.bounds_dict['G_f']
        H_act_bounds = self.bounds_dict['H_act']
        S_act_bounds = self.bounds_dict['S_act']
        log_k0_bounds = self.bounds_dict['log_k0']
        bounds = [G_ss_bounds, G_ter_bounds, G_f_bounds,
                  H_act_bounds, S_act_bounds, log_k0_bounds]
        return bounds
