from numpy.distutils.misc_util import Configuration

def configuration(parent_package='', top_path=None):
    config = Configuration('foldkin', parent_package, top_path)
    config.add_subpackage('base')
    config.add_subpackage('kings')
    config.add_subpackage('coop')
    config.add_subpackage('one_param_curve')
    config.add_subpackage('simple')
    config.add_subpackage('test')
    config.add_subpackage('quadratic_curve')
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
