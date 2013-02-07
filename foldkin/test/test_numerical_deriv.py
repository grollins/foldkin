import nose.tools
import numpy
from foldkin.util import compute_dy_at_x, compute_ddy_at_x

EPSILON = 1e-3

@nose.tools.nottest
class PS(object):
    def __init__(self):
        self.x = 0.0
        self.x0 = 0.0
        self.y0 = 0.0
    def set_parameter(self, id_str, value):
        if id_str == 'x':
            self.x = value
        elif id_str == 'x0':
            self.x0 = value
        elif id_str =='y0':
            self.y0 = value
        else:
            print "Unknown parameter", id_str
    def get_parameter(self, id_str):
        if id_str == 'x':
            return self.x
        elif id_str == 'x0':
            return self.x0
        elif id_str =='y0':
            return self.y0
        else:
            print "Unknown parameter", id_str
            return None

class ModelFactory(object):
    def __init__(self):
        pass
    def create_model(self, id_str, ps):
        return Model(ps)

class Model(object):
    def __init__(self, ps):
        self.ps = ps
    def get_parameter(self, id_str):
        return self.ps.get_parameter(id_str)

def y_fcn(model):
    x = model.get_parameter('x')
    x0 = model.get_parameter('x0')
    y0 = model.get_parameter('y0')
    return y0 + 3 * (x - x0)**2

@nose.tools.istest
class TestComputeDerivOfCurve(object):

    @nose.tools.istest
    def return_correct_first_deriv(self):
        '''This example numerically computes derivative of
           y = y0 + 3(x-x0)^2
           dy_dx should equal 0
        '''
        x0 = 3.0
        y0 = 6.0
        ps = PS()
        ps.set_parameter('x0', x0)
        ps.set_parameter('y0', y0)
        x = x0
        model_factory = ModelFactory()
        dydx = compute_dy_at_x(x, 'x', ps, model_factory, y_fcn)
        expected_dydx = 0.0
        dydx_diff = expected_dydx - dydx
        error_msg = "%.1f  %.1f" % (expected_dydx, dydx)
        nose.tools.ok_(dydx_diff < EPSILON, error_msg)
        print error_msg

    @nose.tools.istest
    def compute_correct_second_deriv(self):
        '''This example numerically computes second derivative of
           y = y0 + 3(x-x0)^2
           ddy_ddx should equal 6
        '''
        x0 = 3.0
        y0 = 6.0
        ps = PS()
        ps.set_parameter('x0', x0)
        ps.set_parameter('y0', y0)
        x = x0
        model_factory = ModelFactory()
        ddy_ddx = compute_ddy_at_x(x, 'x', ps, model_factory, y_fcn)
        expected_ddy_ddx = 6.0
        ddydx_diff = expected_ddy_ddx - ddy_ddx
        error_msg = "%.1f  %.1f" % (expected_ddy_ddx, ddy_ddx)
        nose.tools.ok_(ddydx_diff < EPSILON, error_msg)
        print error_msg
